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
- `MAX_IMAGE_BYTES` (default `10485760`)
- `IMAGE_FETCH_TIMEOUT` (default `15`)
- `SAVE_PREDICTIONS` (default `0`) to store predictions in `predicted_annotations`
- `ALLOW_NULL_STEP_IMAGE` (default `1`) to allow writes without a known `step_image_id`

Deployment registration (optional):
- `PUBLIC_BASE_URL` (e.g. `http://jetson.local:8001`)
- `REGISTER_DEPLOYMENTS` (default `1`)
- `POLL_DEPLOYMENTS` (default `1`)
- `DEPLOYMENT_POLL_INTERVAL` (default `20`)
- `OVERWRITE_DEPLOYMENT_URL` (default `0`)

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

## Endpoints
- `GET /health` -> service status
- `GET /models` -> list deployed/deploying runs
- `POST /models/{run_id}/predict` -> run inference
- `POST /models/{run_id}/reload` -> drop cached model

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
