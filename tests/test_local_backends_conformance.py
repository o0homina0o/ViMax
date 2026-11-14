"""Offline conformance tests for the local ViMax backends (no GPU required).

Starts in-process mock servers that mimic local_servers/*.py's JSON envelope,
then verifies ``tools.ImageGeneratorQwenLocal`` / ``tools.VideoGeneratorLTXLocal``
against ViMax's real contracts: protocol compliance, output saving, the
landscape guard, frame/dimension snapping, and RenderBackend wiring.

Run from the repo root:
    python tests/test_local_backends_conformance.py
"""

import asyncio
import base64
import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aiohttp import web
from PIL import Image

from interfaces.image_output import ImageOutput
from interfaces.video_output import VideoOutput
from tools.local_image_generator import ImageGeneratorQwenLocal
from tools.local_video_generator import VideoGeneratorLTXLocal
from tools.protocols import ImageGenerator, VideoGenerator
from tools.render_backend import RenderBackend


def _png_bytes(width: int, height: int) -> bytes:
    image = Image.new("RGB", (width, height), (120, 140, 180))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _make_image_app(portrait: bool = False) -> web.Application:
    w, h = (64, 128) if portrait else (128, 64)

    async def generate(request):
        data = await request.json()
        assert "prompt" in data and "width" in data and "height" in data
        return web.json_response({
            "ok": True,
            "b64": base64.b64encode(_png_bytes(w, h)).decode(),
            "ext": "png",
        })

    app = web.Application()
    app.router.add_post("/generate", generate)
    return app


def _make_video_app() -> web.Application:
    async def generate(request):
        data = await request.json()
        assert data["num_frames"] % 8 == 1, f"frames not 8k+1: {data['num_frames']}"
        assert data["width"] % 32 == 0 and data["height"] % 32 == 0, "dims not /32"
        return web.json_response({
            "ok": True,
            "data_b64": base64.b64encode(b"FAKE_MP4_BYTES").decode(),
            "ext": "mp4",
            "num_frames": data["num_frames"],
            "fps": data["fps"],
        })

    app = web.Application()
    app.router.add_post("/generate", generate)
    return app


async def _start(app):
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


async def main() -> bool:
    results = []

    def check(name, condition):
        results.append(bool(condition))
        print(("PASS  " if condition else "FAIL  ") + name)

    image_runner, image_url = await _start(_make_image_app())
    portrait_runner, portrait_url = await _start(_make_image_app(portrait=True))
    video_runner, video_url = await _start(_make_video_app())
    try:
        tmp = Path(tempfile.mkdtemp())
        reference = tmp / "ref.png"
        Image.new("RGB", (32, 32), (200, 10, 10)).save(reference)

        image_gen = ImageGeneratorQwenLocal(url=image_url)
        video_gen = VideoGeneratorLTXLocal(url=video_url)

        check("ImageGeneratorQwenLocal satisfies ImageGenerator protocol",
              isinstance(image_gen, ImageGenerator))
        check("VideoGeneratorLTXLocal satisfies VideoGenerator protocol",
              isinstance(video_gen, VideoGenerator))

        # --- image: landscape request -> b64 output -> saveable -------------------
        output = await image_gen.generate_single_image(
            prompt="a test frame", reference_image_paths=[str(reference)], size="1600x900")
        check("image generation returns ImageOutput(fmt='b64')",
              isinstance(output, ImageOutput) and output.fmt == "b64")
        image_path = tmp / "out.png"
        output.save(str(image_path))
        check("ImageOutput.save writes a decodable PNG at requested size",
              image_path.stat().st_size > 0 and Image.open(image_path).size == (128, 64))

        # --- image: portrait response must be rejected (built-in retry semantics) --
        try:
            await ImageGeneratorQwenLocal(url=portrait_url).generate_single_image(
                prompt="p", reference_image_paths=[], size="900x1600")
            rejected = False
        except ValueError:
            rejected = True
        check("portrait image rejected with ValueError (landscape guard)", rejected)

        permissive = await ImageGeneratorQwenLocal(url=image_url).generate_single_image(
            prompt="p", reference_image_paths=[], size="900x1600", allow_portrait=True)
        check("allow_portrait=True bypasses the guard", permissive is not None)

        # --- video: 1 reference frame, snapping, bytes output ---------------------
        video = await video_gen.generate_single_video(
            prompt="camera pushes in", reference_image_paths=[str(reference)],
            num_frames=100)  # snapped to 97 (8k+1)
        check("video generation returns VideoOutput(fmt='bytes')",
              isinstance(video, VideoOutput) and video.fmt == "bytes")
        video_path = tmp / "out.mp4"
        video.save(str(video_path))
        check("VideoOutput.save writes non-empty mp4", video_path.stat().st_size > 0)

        # --- video: >2 reference frames must be rejected --------------------------
        try:
            await video_gen.generate_single_video(
                prompt="p", reference_image_paths=[str(reference)] * 3)
            too_many_rejected = False
        except ValueError:
            too_many_rejected = True
        check("more than two reference frames rejected", too_many_rejected)

        # --- RenderBackend wiring --------------------------------------------------
        config = {
            "image_generator": {"class_path": "tools.ImageGeneratorQwenLocal",
                                "init_args": {"url": image_url}},
            "video_generator": {"class_path": "tools.VideoGeneratorLTXLocal",
                                "init_args": {"url": video_url}},
        }
        backend = RenderBackend.from_config(config)
        check("RenderBackend.from_config instantiates both local classes",
              isinstance(backend.image_generator, ImageGeneratorQwenLocal)
              and isinstance(backend.video_generator, VideoGeneratorLTXLocal))

        print()
        print(f"{sum(results)}/{len(results)} checks passed")
        return all(results)
    finally:
        await image_runner.cleanup()
        await portrait_runner.cleanup()
        await video_runner.cleanup()


if __name__ == "__main__":
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)
