"""Run ViMax pipelines from a JSON input file -- pairs with the local GPU backends.

Usage:
    python run_local.py idea2video   --input my_idea.json
    python run_local.py script2video --input my_script.json
    python run_local.py idea2video   --input my_idea.json --config configs/idea2video.yaml

JSON input schema:
    idea2video:   {"idea": "...", "user_requirement": "...", "style": "..."}
    script2video: {"script": "...", "user_requirement": "...", "style": "..."}

Config defaults to configs/<pipeline>.local-gpu.yaml when present, so the
local LTX-2.5 / Qwen-Image servers on 127.0.0.1 are used automatically.
"""

import argparse
import asyncio
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent


async def _run(pipeline: str, spec: dict, config: str):
    if pipeline == "idea2video":
        from pipelines.idea2video_pipeline import Idea2VideoPipeline
        pipe = Idea2VideoPipeline.init_from_config(config_path=config)
        await pipe(idea=spec["idea"],
                   user_requirement=spec.get("user_requirement", ""),
                   style=spec.get("style", ""))
    elif pipeline == "script2video":
        from pipelines.script2video_pipeline import Script2VideoPipeline
        pipe = Script2VideoPipeline.init_from_config(config_path=config)
        await pipe(script=spec["script"],
                   user_requirement=spec.get("user_requirement", ""),
                   style=spec.get("style", ""))
    else:
        raise SystemExit(f"unknown pipeline: {pipeline}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pipeline", choices=["idea2video", "script2video"])
    parser.add_argument("--input", required=True,
                        help="JSON file with idea/script + user_requirement + style")
    parser.add_argument("--config", default=None,
                        help="ViMax YAML config (default: configs/<pipeline>.local-gpu.yaml if present)")
    args = parser.parse_args()

    config = args.config
    if config is None:
        local = REPO / "configs" / f"{args.pipeline}.local-gpu.yaml"
        config = str(local if local.exists() else REPO / "configs" / f"{args.pipeline}.yaml")
    print(f"pipeline={args.pipeline} config={config}")

    spec = json.loads(Path(args.input).read_text(encoding="utf-8"))
    asyncio.run(_run(args.pipeline, spec, config))


if __name__ == "__main__":
    main()
