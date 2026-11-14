"""Local LTX-2.5 video backend for ViMax.

``VideoGeneratorLTXLocal`` talks to a self-hosted GPU inference server
(``local_servers/ltx_video_server.py``) that serves diffusers
``Lightricks/LTX-2.5-Diffusers`` over HTTP: text-to-video, first-frame
image-to-video, and first+last-frame conditioned rendering. No cloud API.

Frame/size rules are enforced server-side too, but the client snaps them
up front so failed requests never reach the GPU: frame counts are snapped
to the 8k+1 form LTX requires and dimensions to multiples of 32.
"""

import asyncio
import logging
import os
from typing import List, Optional

import aiohttp

from interfaces.video_output import VideoOutput
from utils.image import image_path_to_b64
from utils.rate_limiter import RateLimiter


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return max(1.0, float(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _snap_frames(n: int) -> int:
    """LTX requires num_frames == 8k + 1."""
    return max(9, ((max(1, n) - 1) // 8) * 8 + 1)


def _snap_dim(value: int) -> int:
    """LTX VAE requires dimensions divisible by 32."""
    return max(256, (int(value) // 32) * 32)


def _emit_progress(progress, stage: str, message: str, metadata: dict | None = None) -> None:
    if progress is not None:
        progress(stage, message, metadata or {})


class VideoGeneratorLTXLocal:
    """Generate videos with a local LTX-2.5 server (t2v / i2v / first+last frame)."""

    def __init__(
        self,
        url: str = "http://127.0.0.1:8702",
        model: str = "Lightricks/LTX-2.5-Diffusers",
        token: str = "",
        num_frames: Optional[int] = None,
        fps: Optional[int] = None,
        resolution: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.url = url.rstrip("/")
        self.model = model
        self.token = token
        self.num_frames = _snap_frames(num_frames or _env_int("VIMAX_LTX_NUM_FRAMES", 121))
        self.fps = fps or _env_int("VIMAX_LTX_FPS", 24)
        self.resolution = resolution or os.environ.get("VIMAX_LTX_RESOLUTION", "1280x704")
        self.timeout_seconds = timeout_seconds or _env_float("VIMAX_LOCAL_VIDEO_TIMEOUT_SECONDS", 7200.0)
        self.rate_limiter = rate_limiter

    async def generate_single_video(
        self,
        prompt: str = "",
        reference_image_paths: List[str] = [],
        aspect_ratio: str = "16:9",
        **kwargs,
    ) -> VideoOutput:
        if len(reference_image_paths) > 2:
            raise ValueError("Local LTX video generation supports at most first and last frame images")

        logging.info("Calling local LTX-2.5 server at %s (mode=%s) ...",
                     self.url, ["t2v", "i2v", "first+last"][min(len(reference_image_paths), 2)])

        if self.rate_limiter:
            await self.rate_limiter.acquire()

        width, height = None, None
        size = kwargs.get("size") or os.environ.get("VIMAX_LTX_RESOLUTION_OVERRIDE")
        if isinstance(size, str) and "x" in size.lower():
            try:
                left, right = size.lower().split("x", 1)
                width, height = _snap_dim(int(left)), _snap_dim(int(right))
            except ValueError:
                width, height = None, None
        if width is None:
            try:
                left, right = self.resolution.lower().split("x", 1)
                width, height = _snap_dim(int(left)), _snap_dim(int(right))
            except ValueError:
                width, height = 1280, 704
        if height > width:  # keep landscape, like the cloud backends
            width, height = height, width

        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_frames": _snap_frames(kwargs.get("num_frames") or self.num_frames),
            "fps": self.fps,
            "reference_images": [image_path_to_b64(p, mime=True) for p in reference_image_paths],
        }
        for key in ("num_inference_steps", "guidance_scale", "negative_prompt", "seed",
                    "generate_audio"):
            value = kwargs.get(key)
            if value is None and key == "generate_audio":
                value = os.environ.get("VIMAX_LTX_GENERATE_AUDIO", "0").lower() in {"1", "true", "yes", "on"}
            if value is not None:
                payload[key] = value

        progress = kwargs.get("progress")
        _emit_progress(progress, "video_create",
                       f"Rendering video locally with {self.model} ({payload['num_frames']} frames @ {width}x{height})",
                       {"model": self.model, "num_frames": payload["num_frames"],
                        "width": width, "height": height, "frame_count": len(reference_image_paths)})

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{self.url}/generate", json=payload,
                                    headers=self._headers()) as response:
                body = await response.json(content_type=None)
                if response.status >= 400:
                    raise RuntimeError(
                        f"Local LTX server failed (HTTP {response.status}): {str(body)[:500]}")

        import base64
        data = base64.b64decode(body.get("data_b64") or "")
        if not data:
            raise RuntimeError(f"Local LTX server returned no video: {str(body)[:500]}")

        _emit_progress(progress, "video_completed", "Local LTX-2.5 rendering completed",
                       {"frames": body.get("num_frames"), "fps": body.get("fps")})
        return VideoOutput(fmt="bytes", ext=body.get("ext", "mp4"), data=data)

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers
