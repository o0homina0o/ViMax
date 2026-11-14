"""Local LTX-2.5 video inference server for ViMax (runs on your GPU box).

Serves ``Lightricks/LTX-2.5-Diffusers`` via diffusers over a tiny JSON HTTP
API consumed by ``tools.VideoGeneratorLTXLocal``:

    0 reference images -> text-to-video           (LTX2Pipeline)
    1 reference image  -> first-frame i2v         (LTX2ImageToVideoPipeline)
    2 reference images-> first+last conditioning  (LTX2ConditionPipeline)

    python ltx_video_server.py                        # 127.0.0.1:8702

Endpoints:
    GET  /health    -> model/device status (also reports accepted call params)
    POST /generate  -> {prompt, width, height, num_frames, fps, reference_images: [...]}

Env knobs:
    LTX_MODEL                default Lightricks/LTX-2.5-Diffusers
    LTX_DTYPE / LTX_DEVICE   default bfloat16 / cuda
    LTX_HOST / LTX_PORT      default 127.0.0.1 / 8702
    LTX_SERVER_TOKEN         optional bearer token
    LTX_ENABLE_CPU_OFFLOAD   1 -> pipeline.enable_model_cpu_offload()
    LTX_KEEP_ALL_LOADED      1 -> keep mode pipelines resident after load
    LTX_STEPS / LTX_GUIDANCE sampling defaults (40 / 3.0)
"""

import base64
import gc
import inspect
import io
import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List

import torch
from diffusers.utils import export_to_video
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

import diffusers
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_REPO = os.environ.get("LTX_MODEL", "Lightricks/LTX-2.5-Diffusers")
_DTYPE = getattr(torch, os.environ.get("LTX_DTYPE", "bfloat16"))
_DEVICE = os.environ.get("LTX_DEVICE", "cuda")
_HOST = os.environ.get("LTX_HOST", "127.0.0.1")
_PORT = int(os.environ.get("LTX_PORT", "8702"))
_TOKEN = os.environ.get("LTX_SERVER_TOKEN", "")
_KEEP_ALL = os.environ.get("LTX_KEEP_ALL_LOADED", "0") == "1"
_ENABLE_CPU_OFFLOAD = os.environ.get("LTX_ENABLE_CPU_OFFLOAD", "0") == "1"
_DEFAULT_STEPS = int(os.environ.get("LTX_STEPS", "40"))
_DEFAULT_GUIDANCE = float(os.environ.get("LTX_GUIDANCE", "3.0"))

# mode (number of reference frames) -> preferred diffusers pipeline class
_MODE_CLASSES = {0: "LTX2Pipeline", 1: "LTX2ImageToVideoPipeline", 2: "LTX2ConditionPipeline"}

_pipelines: Dict[int, Any] = {}
_pipe_lock = threading.Lock()

app = FastAPI(title="ViMax local LTX-2.5 server")


@app.middleware("http")
async def _auth_middleware(request, call_next):
    if _TOKEN and request.url.path != "/health":
        if request.headers.get("Authorization") != f"Bearer {_TOKEN}":
            return JSONResponse(status_code=401, content={"ok": False, "error": "unauthorized"})
    return await call_next(request)


def _decode_image(data_url: str) -> Image.Image:
    if data_url.startswith("data:") and "," in data_url:
        data_url = data_url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(data_url))).convert("RGB")


def _default_class_name() -> str:
    model_index = json.loads(Path(hf_hub_download(_REPO, "model_index.json")).read_text())
    return model_index.get("_class_name", "LTX2Pipeline")


def _pipeline_class(mode: int):
    """Preferred mode class, falling back to the repo's own default class."""
    name = _MODE_CLASSES[mode]
    cls = getattr(diffusers, name, None)
    if cls is not None:
        return cls
    fallback = _default_class_name()
    logging.warning("diffusers %s has no '%s'; falling back to '%s' from %s model_index "
                    "(mode %d conditioning may be limited)",
                    diffusers.__version__, name, fallback, _REPO, mode)
    cls = getattr(diffusers, fallback, None)
    if cls is None:
        raise RuntimeError(f"diffusers {diffusers.__version__} cannot serve {_REPO}: "
                           f"neither '{name}' nor '{fallback}' exist. Upgrade diffusers.")
    return cls


def _load(mode: int):
    if mode in _pipelines:
        return _pipelines[mode]
    if not _KEEP_ALL and _pipelines:
        for other in list(_pipelines):
            if other != mode:
                logging.info("Unloading pipeline mode %d to free VRAM for mode %d", other, mode)
                del _pipelines[other]
        gc.collect()
        torch.cuda.empty_cache()
    logging.info("Loading %s (mode %d) on %s ...", _REPO, mode, _DEVICE)
    cls = _pipeline_class(mode)
    pipe = cls.from_pretrained(_REPO, torch_dtype=_DTYPE)
    if _ENABLE_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(_DEVICE)
    _pipelines[mode] = pipe
    logging.info("Loaded %s via %s (mode %d)", _REPO, type(pipe).__name__, mode)
    return pipe


def _filter_kwargs(pipe, wanted: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(type(pipe).__call__)
    params = list(sig.parameters.values())
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        return wanted
    accepted = {p.name for p in params}
    dropped = sorted(set(wanted) - accepted)
    if dropped:
        logging.info("Dropping kwargs unsupported by %s: %s", type(pipe).__name__, dropped)
    return {k: v for k, v in wanted.items() if k in accepted}


class GenerateRequest(BaseModel):
    prompt: str = ""
    width: int = 1280
    height: int = 704
    num_frames: int = 121
    fps: int = 24
    reference_images: List[str] = []
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    negative_prompt: str | None = None
    seed: int | None = None
    generate_audio: bool = False


@app.get("/health")
def health():
    info = {
        "ok": True,
        "server": "ltx_video",
        "model": _REPO,
        "modes": {str(m): _MODE_CLASSES[m] for m in _MODE_CLASSES},
        "loaded_modes": sorted(_pipelines.keys()),
        "device": _DEVICE,
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
    }
    try:
        info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


@app.post("/generate")
def generate(req: GenerateRequest):
    try:
        return _generate(req)
    except Exception as exc:
        logging.exception("generation failed")
        return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})


def _snap_frames(n: int) -> int:
    return max(9, ((max(9, int(n)) - 1) // 8) * 8 + 1)


def _snap_dim(value: int) -> int:
    return max(256, (int(value) // 32) * 32)


def _generate(req: GenerateRequest):
    with _pipe_lock:
        mode = min(len(req.reference_images), 2)
        pipe = _load(mode)

        width, height = _snap_dim(req.width), _snap_dim(req.height)
        if height > width:
            width, height = height, width
        num_frames = _snap_frames(req.num_frames)
        generator = (torch.Generator(device=_DEVICE).manual_seed(int(req.seed))
                     if req.seed is not None else None)
        guidance = req.guidance_scale if req.guidance_scale is not None else _DEFAULT_GUIDANCE

        wanted: Dict[str, Any] = {
            "prompt": req.prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "num_inference_steps": req.num_inference_steps or _DEFAULT_STEPS,
            "guidance_scale": guidance,
            "generator": generator,
            "generate_audio": req.generate_audio,
        }
        if req.negative_prompt is not None:
            wanted["negative_prompt"] = req.negative_prompt

        if mode >= 1:
            first = _decode_image(req.reference_images[0]).resize((width, height), Image.LANCZOS)
            wanted["image"] = first
        if mode == 2:
            last = _decode_image(req.reference_images[1]).resize((width, height), Image.LANCZOS)
            wanted["last_image"] = last  # dropped by the filter if this version lacks it

        kwargs = _filter_kwargs(pipe, wanted)
        with torch.inference_mode():
            output = pipe(**kwargs)

        frames = getattr(output, "frames", None)
        if frames is None:
            frames = getattr(output, "videos", None)
        if isinstance(frames, (list, tuple)):
            frames = frames[0]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
            video_path = handle.name
        try:
            if req.generate_audio:
                audio = getattr(output, "audio", None)
                try:
                    export_to_video(frames, video_path, fps=req.fps, audio=audio)
                except TypeError:
                    export_to_video(frames, video_path, fps=req.fps)
            else:
                export_to_video(frames, video_path, fps=req.fps)
            data = Path(video_path).read_bytes()
        finally:
            try:
                os.unlink(video_path)
            except OSError:
                pass

        return {
            "ok": True,
            "data_b64": base64.b64encode(data).decode("ascii"),
            "ext": "mp4",
            "num_frames": num_frames,
            "fps": req.fps,
            "width": width,
            "height": height,
            "mode": mode,
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=_HOST, port=_PORT)
