"""Local Qwen-Image inference server for ViMax (runs on your GPU box).

Serves ``Qwen/Qwen-Image-2.1`` (text-to-image) and
``Qwen/Qwen-Image-Edit-2511`` (reference-conditioned editing) via diffusers
over a tiny JSON HTTP API consumed by ``tools.ImageGeneratorQwenLocal``.

    python qwen_image_server.py                       # 127.0.0.1:8701
    QWEN_DEVICE=cuda:1 QWEN_PORT=8701 python qwen_image_server.py

Endpoints:
    GET  /health    -> model/device status (also reports accepted call params)
    POST /generate  -> {prompt, width, height, reference_images: [data-url|b64], ...}

Env knobs:
    QWEN_T2I_MODEL           default Qwen/Qwen-Image-2.1
    QWEN_EDIT_MODEL          default Qwen/Qwen-Image-Edit-2511
    QWEN_DTYPE               default bfloat16
    QWEN_DEVICE              default cuda
    QWEN_HOST / QWEN_PORT    default 127.0.0.1 / 8701
    QWEN_SERVER_TOKEN        optional bearer token (clients set init_args: token)
    QWEN_ENABLE_CPU_OFFLOAD  1 -> pipeline.enable_model_cpu_offload()
    QWEN_KEEP_ALL_LOADED     1 -> keep both pipelines resident after load
    QWEN_EDIT_MAX_SIDE       reference images downscaled to this (default 1024)
    QWEN_STEPS / QWEN_GUIDANCE   sampling defaults (40 / 4.0)
"""

import base64
import gc
import inspect
import io
import json
import logging
import os
import threading
from typing import Any, Dict, List

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

import diffusers
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_T2I = os.environ.get("QWEN_T2I_MODEL", "Qwen/Qwen-Image-2.1")
_EDIT = os.environ.get("QWEN_EDIT_MODEL", "Qwen/Qwen-Image-Edit-2511")
_DTYPE = getattr(torch, os.environ.get("QWEN_DTYPE", "bfloat16"))
_DEVICE = os.environ.get("QWEN_DEVICE", "cuda")
_HOST = os.environ.get("QWEN_HOST", "127.0.0.1")
_PORT = int(os.environ.get("QWEN_PORT", "8701"))
_TOKEN = os.environ.get("QWEN_SERVER_TOKEN", "")
_KEEP_ALL = os.environ.get("QWEN_KEEP_ALL_LOADED", "0") == "1"
_ENABLE_CPU_OFFLOAD = os.environ.get("QWEN_ENABLE_CPU_OFFLOAD", "0") == "1"
_EDIT_MAX_SIDE = int(os.environ.get("QWEN_EDIT_MAX_SIDE", "1024"))
_DEFAULT_STEPS = int(os.environ.get("QWEN_STEPS", "40"))
_DEFAULT_GUIDANCE = float(os.environ.get("QWEN_GUIDANCE", "4.0"))

_pipelines: Dict[str, Any] = {}
_pipe_lock = threading.Lock()  # single GPU: serialize loading + inference

app = FastAPI(title="ViMax local Qwen-Image server")


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


def _pipeline_class(repo_id: str):
    """Resolve the diffusers pipeline class from the repo's model_index.json (version-proof)."""
    model_index = json.loads(__import__("pathlib").Path(
        hf_hub_download(repo_id, "model_index.json")).read_text())
    cls_name = model_index.get("_class_name")
    cls = getattr(diffusers, cls_name, None)
    if cls is None:
        raise RuntimeError(
            f"diffusers {diffusers.__version__} has no class '{cls_name}' required by {repo_id}. "
            "Upgrade diffusers (pip install -U diffusers) or use a matching version.")
    return cls


def _load(kind: str):
    """Load and cache the pipeline for `kind`; unload the other kind unless QWEN_KEEP_ALL_LOADED=1."""
    if kind in _pipelines:
        return _pipelines[kind]
    if not _KEEP_ALL and _pipelines:
        for other in list(_pipelines):
            if other != kind:
                logging.info("Unloading pipeline '%s' to free VRAM for '%s'", other, kind)
                del _pipelines[other]
        gc.collect()
        torch.cuda.empty_cache()
    repo_id = _EDIT if kind == "edit" else _T2I
    logging.info("Loading %s (%s) on %s ...", repo_id, kind, _DEVICE)
    cls = _pipeline_class(repo_id)
    pipe = cls.from_pretrained(repo_id, torch_dtype=_DTYPE)
    if _ENABLE_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(_DEVICE)
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass
    _pipelines[kind] = pipe
    logging.info("Loaded %s (%s)", repo_id, kind)
    return pipe


def _filter_kwargs(pipe, wanted: Dict[str, Any]) -> Dict[str, Any]:
    """Drop kwargs the installed pipeline version does not accept (version-proof calls)."""
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
    prompt: str
    width: int = 1600
    height: int = 900
    reference_images: List[str] = []
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    negative_prompt: str | None = None
    seed: int | None = None


@app.get("/health")
def health():
    info = {
        "ok": True,
        "server": "qwen_image",
        "models": {"t2i": _T2I, "edit": _EDIT},
        "loaded": sorted(_pipelines.keys()),
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


def _generate(req: GenerateRequest):
    with _pipe_lock:
        kind = "edit" if req.reference_images else "t2i"
        pipe = _load(kind)

        width = max(256, (int(req.width) // 16) * 16)
        height = max(256, (int(req.height) // 16) * 16)
        generator = (torch.Generator(device=_DEVICE).manual_seed(int(req.seed))
                     if req.seed is not None else None)
        guidance = req.guidance_scale if req.guidance_scale is not None else _DEFAULT_GUIDANCE

        wanted: Dict[str, Any] = {
            "prompt": req.prompt,
            "width": width,
            "height": height,
            "num_inference_steps": req.num_inference_steps or _DEFAULT_STEPS,
            # Qwen pipelines name this differently across versions; set both, filter drops
            # whichever the installed class does not accept.
            "guidance_scale": guidance,
            "true_cfg_scale": guidance,
            "generator": generator,
        }
        if req.negative_prompt is not None:
            wanted["negative_prompt"] = req.negative_prompt

        if kind == "edit":
            images = []
            for data_url in req.reference_images:
                img = _decode_image(data_url)
                img.thumbnail((_EDIT_MAX_SIDE, _EDIT_MAX_SIDE))
                images.append(img)
            wanted["image"] = images[0] if len(images) == 1 else images

        kwargs = _filter_kwargs(pipe, wanted)
        with torch.inference_mode():
            output = pipe(**kwargs)
        image = output.images[0]

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return {
            "ok": True,
            "b64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "ext": "png",
            "width": image.width,
            "height": image.height,
            "mode": kind,
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=_HOST, port=_PORT)
