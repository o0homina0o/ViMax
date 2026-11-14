"""Local Qwen-Image backend for ViMax.

``ImageGeneratorQwenLocal`` talks to a self-hosted GPU inference server
(``local_servers/qwen_image_server.py``) that serves diffusers
``Qwen/Qwen-Image-2.1`` (text-to-image) and ``Qwen/Qwen-Image-Edit-2511``
(reference-conditioned editing) over HTTP. No cloud API is involved.

Landscape semantics intentionally mirror the built-in generators: portrait
requests are converted to landscape up front and returned images are
validated with ``ensure_not_portrait`` so the pipeline's retry logic behaves
identically to the cloud backends.
"""

import base64
import logging
import os
from io import BytesIO
from typing import List, Optional

import aiohttp
from PIL import Image

from interfaces.image_output import ImageOutput
from tools.image_orientation import ensure_not_portrait, landscape_guard_requested
from utils.image import image_path_to_b64
from utils.rate_limiter import RateLimiter


# aspect_ratio -> default pixel size (landscape-first, mirrors ViMax defaults)
_ASPECT_SIZES = {
    "16:9": (1600, 900),
    "4:3": (1440, 1080),
    "3:2": (1536, 1024),
    "1:1": (1024, 1024),
    "3:4": (1080, 1440),
    "9:16": (900, 1600),
    "21:9": (1680, 720),
}


def _env_float(name: str, default: float) -> float:
    try:
        return max(1.0, float(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _emit_progress(progress, stage: str, message: str, metadata: dict | None = None) -> None:
    if progress is not None:
        progress(stage, message, metadata or {})


class ImageGeneratorQwenLocal:
    """Generate images with a local Qwen-Image-2.1 / Qwen-Image-Edit-2511 server."""

    def __init__(
        self,
        url: str = "http://127.0.0.1:8701",
        token: str = "",
        timeout_seconds: Optional[float] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.url = url.rstrip("/")
        self.token = token
        self.model = "Qwen/Qwen-Image-2.1+Qwen-Image-Edit-2511 (local)"
        self.timeout_seconds = timeout_seconds or _env_float("VIMAX_LOCAL_IMAGE_TIMEOUT_SECONDS", 1800.0)
        self.rate_limiter = rate_limiter

    async def generate_single_image(
        self,
        prompt: str,
        reference_image_paths: List[str] = [],
        aspect_ratio: Optional[str] = "16:9",
        **kwargs,
    ) -> ImageOutput:
        logging.info("Calling local Qwen-Image server at %s ...", self.url)

        if self.rate_limiter:
            await self.rate_limiter.acquire()

        # --- resolve output size -------------------------------------------------
        size = kwargs.get("size")
        width, height = None, None
        if isinstance(size, str) and "x" in size.lower():
            try:
                left, right = size.lower().split("x", 1)
                width, height = int(left.strip()), int(right.strip())
            except ValueError:
                width, height = None, None
        if width is None:
            width, height = _ASPECT_SIZES.get(aspect_ratio or "", _ASPECT_SIZES["16:9"])

        enforce = kwargs.get("enforce_landscape", True)
        allow_portrait = kwargs.get("allow_portrait", False)
        if landscape_guard_requested(size=f"{width}x{height}", aspect_ratio=aspect_ratio,
                                     enforce_landscape=enforce, allow_portrait=allow_portrait) \
                and height > width:
            width, height = height, width  # request landscape up front, like the built-ins

        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "reference_images": [image_path_to_b64(p, mime=True) for p in reference_image_paths],
        }
        for key in ("num_inference_steps", "guidance_scale", "negative_prompt", "seed"):
            value = kwargs.get(key)
            if value is not None:
                payload[key] = value

        progress = kwargs.get("progress")
        _emit_progress(progress, "image_generate_start",
                       f"Generating image with local Qwen-Image ({width}x{height})",
                       {"width": width, "height": height, "references": len(reference_image_paths)})

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{self.url}/generate", json=payload,
                                    headers=self._headers()) as response:
                body = await response.json(content_type=None)
                if response.status >= 400:
                    raise RuntimeError(
                        f"Local Qwen-Image server failed (HTTP {response.status}): {str(body)[:500]}")

        b64 = body.get("b64")
        ext = body.get("ext", "png")
        if not b64:
            raise RuntimeError(f"Local Qwen-Image server returned no image: {str(body)[:500]}")

        image = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
        if landscape_guard_requested(size=f"{width}x{height}", aspect_ratio=aspect_ratio,
                                     enforce_landscape=enforce, allow_portrait=allow_portrait):
            ensure_not_portrait(image)  # raises -> pipeline retries, same as built-ins

        _emit_progress(progress, "image_generate_done", "Local Qwen-Image generation completed",
                       {"width": image.width, "height": image.height})
        return ImageOutput(fmt="b64", ext=ext, data=b64)

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers
