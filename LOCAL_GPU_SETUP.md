# ViMax Local GPU Backends — RTX PRO 6000 (Blackwell)

This branch adds **fully local** image and video generation to ViMax. No cloud
API is used for pixels — only the chat/keyframe-selection model runs remotely
(free OpenRouter tier by default).

```
ViMax pipeline (your GPU box)
 ├─ chat / keyframe selection -> OpenRouter google/gemma-4-31b-it:free (free, cloud)
 ├─ images  -> tools.ImageGeneratorQwenLocal ──HTTP──> local_servers/qwen_image_server.py :8701
 │                     Qwen/Qwen-Image-2.1 (t2i) + Qwen/Qwen-Image-Edit-2511 (character refs)
 └─ video   -> tools.VideoGeneratorLTXLocal  ──HTTP──> local_servers/ltx_video_server.py  :8702
                       Lightricks/LTX-2.5-Diffusers (t2v / first-frame i2v / first+last frame)
```

The servers run in their **own venv** (torch cu128 for Blackwell sm_120); ViMax
itself stays torch-free. Both servers handle the ugly parts for you:
pipeline-version kwargs filtering, VRAM release between modes, LTX 8k+1 frame
rules, /32 dimension alignment, landscape enforcement, and rate-limiter wiring.

## 0. Get this branch

```bash
git clone -b local-backends https://github.com/o0homina0o/ViMax.git vimax-local
cd vimax-local
```

Already have a clone? `git remote add local https://github.com/o0homina0o/ViMax.git
&& git fetch local local-backends && git checkout local-backends`

## 1. ViMax runtime environment

```bash
uv sync        # ViMax + pipeline deps (no torch required here)
```

## 2. GPU server environment (dedicated venv)

```bash
uv venv .venv-gpu
uv pip install --python .venv-gpu/bin/python torch \
    --index-url https://download.pytorch.org/whl/cu128   # Blackwell = cu128 or newer
uv pip install --python .venv-gpu/bin/python -r local_servers/requirements.txt
```

## 3. Hugging Face authentication

```bash
export HF_TOKEN=hf_xxx           # token of an account that accepted the Lightricks license
export HF_HOME=/path/to/big/disk # first run downloads ~100+ GB of weights
```

## 4. Start the servers

```bash
.venv-gpu/bin/python local_servers/qwen_image_server.py &   # :8701
.venv-gpu/bin/python local_servers/ltx_video_server.py  &   # :8702

curl -s localhost:8701/health | python3 -m json.tool
curl -s localhost:8702/health | python3 -m json.tool
```

`/health` reports the resolved pipeline classes and accepted call parameters —
check it first if anything misbehaves.

## 5. Configure chat, then create a video

Paste your OpenRouter key into `configs/idea2video.local-gpu.yaml` and
`configs/script2video.local-gpu.yaml` (`api_key:` under `chat_model`). Any
OpenAI-compatible chat endpoint works.

```bash
.venv/bin/python run_local.py idea2video   --input my_idea.json
.venv/bin/python run_local.py script2video --input my_script.json
```

Results land under `.working_dir/idea2video/` / `.working_dir/script2video/`.

## Tuning knobs

| Where | Variable / init_arg | Default | Meaning |
|---|---|---|---|
| image server | `QWEN_STEPS` / `QWEN_GUIDANCE` | 40 / 4.0 | sampling defaults |
| image server | `QWEN_ENABLE_CPU_OFFLOAD=1` | off | keep weights in RAM, stream to GPU |
| image server | `QWEN_KEEP_ALL_LOADED=1` | off | keep both t2i+edit pipelines resident |
| image server | `QWEN_EDIT_MAX_SIDE` | 1024 | reference images downscaled to this |
| video server | `LTX_STEPS` / `LTX_GUIDANCE` | 40 / 3.0 | sampling defaults |
| video server | `LTX_ENABLE_CPU_OFFLOAD=1` | off | VRAM relief |
| video client | `VIMAX_LTX_NUM_FRAMES` | 121 | snapped to 8k+1 |
| video client | `VIMAX_LTX_FPS` | 24 | export fps |
| video client | `VIMAX_LTX_RESOLUTION` | 1280x704 | snapped to /32 |
| video client | `VIMAX_LTX_GENERATE_AUDIO=1` | off | LTX-2 audio track when supported |
| both clients | `VIMAX_LOCAL_{IMAGE,VIDEO}_TIMEOUT_SECONDS` | 1800 / 7200 | first call includes model load |
| both servers | `QWEN_SERVER_TOKEN` / `LTX_SERVER_TOKEN` | unset | bearer token (set `token:` in configs too) |

## Remote GPU box?

Run the servers with `QWEN_HOST=0.0.0.0` / `LTX_HOST=0.0.0.0`, set a
`*_SERVER_TOKEN`, and point the config `url:` at `http://<box-ip>:8701|8702`
with matching `token:` values.

## Troubleshooting

- **"diffusers has no class 'LTX2Pipeline'"** — LTX-2 support is recent:
  `uv pip install --python .venv-gpu/bin/python -U diffusers`.
- **first+last frame mode falls back with a warning** — your diffusers lacks
  `LTX2ConditionPipeline`; the video server falls back to the repo's default
  class (first-frame conditioning). Upgrade diffusers for true first+last.
- **CUDA out of memory** — enable `*_ENABLE_CPU_OFFLOAD=1`, lower
  resolution/frames, or leave `QWEN_KEEP_ALL_LOADED`/`LTX_KEEP_ALL_LOADED` off
  (default unloads the unused pipeline between modes).
- **Frequent portrait retries** — normal ViMax retry semantics; tune
  `VIMAX_IMAGE_PORTRAIT_RETRY_TOLERANCE` if needed.
- **Offline sanity check (no GPU needed):**
  `.venv/bin/python tests/test_local_backends_conformance.py` — 10 checks
  covering protocols, output saving, guards, and RenderBackend wiring.
