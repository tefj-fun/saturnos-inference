# saturnos-inference

Inference worker for SaturnOS. This service:
- pulls `training_runs.trained_model_url` from Supabase,
- downloads/caches YOLOv8 `.pt` models locally,
- serves `/models/{run_id}/predict` for the UI,
- optionally writes predictions to `predicted_annotations`.

Default backend is Ultralytics YOLO (no ONNX conversion required). Triton is optional and assumes you already exported models to ONNX/TensorRT and configured Triton to serve them.

## Requirements
- Python 3.10+
- `pip install fastapi uvicorn[standard] ultralytics supabase python-dotenv pillow numpy`
- Jetson or CUDA GPU recommended (CPU works for small models)

## Environment
Required:
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`

Common options:
- `MODEL_ROOT` (default `./models`)
- `MAX_MODELS` (default `2`)
- `DEVICE` (default `0`)
- `DEFAULT_CONFIDENCE` (default `0.25`)
- `DEFAULT_IOU` (default `0.45`)
- `DEFAULT_IMAGE_SIZE` (default `640`)
- `DEFAULT_MAX_DETECTIONS` (default `300`)
- `MODEL_FUSE` (default `1`) to fuse model layers for faster inference
- `MODEL_HALF` (default `0`) to use FP16 on CUDA devices
- `MAX_IMAGE_BYTES` (default `10485760`)
- `IMAGE_FETCH_TIMEOUT` (default `15`)
- `HEARTBEAT_INTERVAL` (default `10`) to upsert worker status into `inference_workers`
- `SAVE_PREDICTIONS` (default `0`) to store predictions in `predicted_annotations`
- `ALLOW_NULL_STEP_IMAGE` (default `1`) to allow writes without a known `step_image_id`

Deployment registration (optional):
- `PUBLIC_BASE_URL` (e.g. `http://jetson.local:8001`)
- `REGISTER_DEPLOYMENTS` (default `1`)
- `POLL_DEPLOYMENTS` (default `1`)
- `DEPLOYMENT_POLL_INTERVAL` (default `20`)
- `OVERWRITE_DEPLOYMENT_URL` (default `0`)

Worker metadata (optional, shown in Results & Analysis status panel):
- `WORKER_ID` (default hostname)
- `WORKER_DEVICE_TYPE`, `WORKER_COMPUTE_TYPE`
- `WORKER_GPU_NAME`, `WORKER_GPU_MODEL`, `WORKER_GPU`
- `WORKER_GPU_MEMORY_GB`, `WORKER_GPU_VRAM_GB`
- `WORKER_GPU_MEMORY_MB`, `WORKER_GPU_VRAM_MB`
- `WORKER_CPU_MODEL`, `WORKER_CPU`, `WORKER_RAM_GB`

Auto-update (optional):
- `AUTO_UPDATE_ENABLED` (default `0`) to enable repo polling
- `AUTO_UPDATE_INTERVAL` (default `300`) seconds between checks
- `AUTO_UPDATE_REMOTE` (default `origin`) git remote to fetch/pull
- `AUTO_UPDATE_BRANCH` (default current branch) branch to track
- `AUTO_UPDATE_ALLOW_DIRTY` (default `0`) allow updates with local changes
- `AUTO_UPDATE_WAIT_FOR_IDLE` (default `1`) wait for no active requests
- `AUTO_UPDATE_MAX_WAIT` (default `60`) seconds to wait before updating anyway
- `AUTO_UPDATE_RESTART_COMMAND` (optional) command to restart the service after pull
  - if unset, the process re-execs itself after pulling
- Requires a git checkout with a valid upstream or `AUTO_UPDATE_REMOTE` + branch.

Supabase schema note:
- Create `public.inference_workers` (see SaturnOS migration `0020_add_inference_workers.sql`)
  to enable the worker heartbeat status panel in the UI.

Triton (optional):
- `INFERENCE_BACKEND=triton`
- `TRITON_URL` (e.g. `localhost:8000`)
- `TRITON_MODEL_NAME_TEMPLATE` (default `{run_id}`)
- `TRITON_INPUT_NAME` (default `images`)
- `TRITON_OUTPUT_NAME` (default `output`)
- `TRITON_INPUT_DTYPE` (default `uint8`)
- `TRITON_INPUT_FORMAT` (default `nhwc`)
- `TRITON_OUTPUT_FORMAT` (default `xyxy`)

## Run
```bash
python inference_service.py
```

## Docker (Jetson)
1) Create `.env.local` with `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` (plus any optional settings).
2) Build and start:
```bash
docker compose build
docker compose up -d
```
3) Verify:
```bash
curl http://localhost:8001/health
```

Note: Docker uses host networking on Jetson to avoid iptables raw-table issues,
so the service binds directly to the host at port 8001.

Auto-start on reboot:
```bash
sudo systemctl enable --now docker
```

If you want a different base image, set `BASE_IMAGE` in `docker-compose.yml`
(default is `nvcr.io/nvidia/pytorch:24.05-py3-igpu`).

## Endpoints
- `GET /health` -> service status
- `GET /models` -> list deployed/deploying runs
- `POST /models/{run_id}/predict` -> run inference
- `POST /models/{run_id}/reload` -> drop cached model

## Admin web interface
This service also serves a lightweight admin dashboard:
- `GET /admin` -> HTML dashboard (polls the API endpoints below)
- `GET /admin/api/status` -> worker status (uptime, backend, cache, heartbeat/poll state)
- `GET /admin/api/logs` -> recent in-memory logs (`?after=<seq>` for incremental tail, `?format=text` for plain text)
- `GET /admin/api/requests` -> recent prediction request history
- `POST /admin/api/models/{run_id}/reload` -> evict a cached model (admin-only)

To protect the admin API, set `ADMIN_TOKEN`. The dashboard stores the token in `localStorage` and sends:
`Authorization: Bearer <token>`.

### Predict examples
JSON body with image URL:
```bash
curl -X POST "http://localhost:8001/models/<run_id>/predict" \
  -H "Content-Type: application/json" \
  -d '{"image_url":"https://...","confidence":0.3,"iou":0.5,"imgsz":640}'
```

Multipart upload:
```bash
curl -X POST "http://localhost:8001/models/<run_id>/predict" \
  -F "image=@/path/to/image.jpg"
```

## Notes on ONNX/Triton
This worker does not convert models to ONNX. If you want Triton:
1) export `.pt` -> ONNX/TensorRT separately,
2) serve via Triton,
3) set `INFERENCE_BACKEND=triton` and the `TRITON_*` vars.
