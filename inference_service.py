import io
import os
import socket
import threading
import time
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from supabase import create_client

try:
    from supabase import ClientOptions
except Exception:
    ClientOptions = None

from PIL import Image
import numpy as np
from ultralytics import YOLO

RUNS_TABLE = "training_runs"
PREDICTIONS_TABLE = "predicted_annotations"
STEP_IMAGES_TABLE = "step_images"
WORKERS_TABLE = "inference_workers"

ENV_PATH = os.path.join(os.path.dirname(__file__), ".env.local")
load_dotenv(ENV_PATH)

SUPABASE_URL = os.getenv("SUPABASE_URL")
if SUPABASE_URL and not SUPABASE_URL.endswith("/"):
    SUPABASE_URL = f"{SUPABASE_URL}/"
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_ARTIFACTS_BUCKET = os.getenv("SUPABASE_ARTIFACTS_BUCKET", "training-artifacts")

MODEL_ROOT = os.getenv("MODEL_ROOT", "./models")
MAX_MODELS = int(os.getenv("MAX_MODELS", "2"))
DEFAULT_CONFIDENCE = float(os.getenv("DEFAULT_CONFIDENCE", "0.25"))
DEFAULT_IOU = float(os.getenv("DEFAULT_IOU", "0.45"))
DEFAULT_IMAGE_SIZE = int(os.getenv("DEFAULT_IMAGE_SIZE", "640"))
DEFAULT_MAX_DETECTIONS = int(os.getenv("DEFAULT_MAX_DETECTIONS", "300"))
DEVICE = os.getenv("DEVICE", "0")
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
IMAGE_FETCH_TIMEOUT = int(os.getenv("IMAGE_FETCH_TIMEOUT", "15"))
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "10"))

SAVE_PREDICTIONS = os.getenv("SAVE_PREDICTIONS", "0").lower() in ("1", "true", "yes")
ALLOW_NULL_STEP_IMAGE = os.getenv("ALLOW_NULL_STEP_IMAGE", "1").lower() in ("1", "true", "yes")

CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "")
CORS_ALLOW_ORIGIN_REGEX = os.getenv("CORS_ALLOW_ORIGIN_REGEX", "")
CORS_ALLOW_METHODS = os.getenv("CORS_ALLOW_METHODS", "GET,POST,OPTIONS")
CORS_ALLOW_HEADERS = os.getenv("CORS_ALLOW_HEADERS", "Authorization,Content-Type,Accept")
CORS_ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "0").lower() in ("1", "true", "yes")
CORS_MAX_AGE = int(os.getenv("CORS_MAX_AGE", "600"))

POLL_DEPLOYMENTS = os.getenv("POLL_DEPLOYMENTS", "1").lower() in ("1", "true", "yes")
DEPLOYMENT_POLL_INTERVAL = int(os.getenv("DEPLOYMENT_POLL_INTERVAL", "20"))
REGISTER_DEPLOYMENTS = os.getenv("REGISTER_DEPLOYMENTS", "1").lower() in ("1", "true", "yes")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
OVERWRITE_DEPLOYMENT_URL = os.getenv("OVERWRITE_DEPLOYMENT_URL", "0").lower() in ("1", "true", "yes")

INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "ultralytics").lower()
TRITON_URL = os.getenv("TRITON_URL")
TRITON_MODEL_NAME_TEMPLATE = os.getenv("TRITON_MODEL_NAME_TEMPLATE", "{run_id}")
TRITON_INPUT_NAME = os.getenv("TRITON_INPUT_NAME", "images")
TRITON_OUTPUT_NAME = os.getenv("TRITON_OUTPUT_NAME", "output")
TRITON_INPUT_DTYPE = os.getenv("TRITON_INPUT_DTYPE", "uint8")
TRITON_INPUT_FORMAT = os.getenv("TRITON_INPUT_FORMAT", "nhwc")
TRITON_OUTPUT_FORMAT = os.getenv("TRITON_OUTPUT_FORMAT", "xyxy")

WORKER_ID = os.getenv("WORKER_ID", socket.gethostname())
WORKER_DEVICE_TYPE = os.getenv("WORKER_DEVICE_TYPE")
WORKER_COMPUTE_TYPE = os.getenv("WORKER_COMPUTE_TYPE")
WORKER_GPU_NAME = os.getenv("WORKER_GPU_NAME")
WORKER_GPU_MODEL = os.getenv("WORKER_GPU_MODEL")
WORKER_GPU = os.getenv("WORKER_GPU")
WORKER_GPU_MEMORY_GB = os.getenv("WORKER_GPU_MEMORY_GB")
WORKER_GPU_VRAM_GB = os.getenv("WORKER_GPU_VRAM_GB")
WORKER_GPU_MEMORY_MB = os.getenv("WORKER_GPU_MEMORY_MB")
WORKER_GPU_VRAM_MB = os.getenv("WORKER_GPU_VRAM_MB")
WORKER_CPU_MODEL = os.getenv("WORKER_CPU_MODEL")
WORKER_CPU = os.getenv("WORKER_CPU")
WORKER_RAM_GB = os.getenv("WORKER_RAM_GB")

WORKER_STATUS_LOCK = threading.Lock()
ACTIVE_REQUESTS_LOCK = threading.Lock()
WORKER_STATUS = "online"
ACTIVE_REQUESTS = 0

app = FastAPI(title="SaturnOS Inference Worker")
cors_origins = [origin.strip() for origin in CORS_ALLOW_ORIGINS.split(",") if origin.strip()]
cors_methods = [method.strip() for method in CORS_ALLOW_METHODS.split(",") if method.strip()]
cors_headers = [header.strip() for header in CORS_ALLOW_HEADERS.split(",") if header.strip()]
if cors_origins or CORS_ALLOW_ORIGIN_REGEX:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_origin_regex=CORS_ALLOW_ORIGIN_REGEX or None,
        allow_credentials=CORS_ALLOW_CREDENTIALS,
        allow_methods=cors_methods or ["GET", "POST", "OPTIONS"],
        allow_headers=cors_headers or ["Authorization", "Content-Type", "Accept"],
        max_age=CORS_MAX_AGE,
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def normalize_float_env(value: Optional[str]):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def set_worker_status(status: str) -> None:
    global WORKER_STATUS
    with WORKER_STATUS_LOCK:
        WORKER_STATUS = status


def get_worker_status() -> str:
    with WORKER_STATUS_LOCK:
        return WORKER_STATUS


def mark_request_start() -> None:
    global ACTIVE_REQUESTS
    with ACTIVE_REQUESTS_LOCK:
        ACTIVE_REQUESTS += 1
        set_worker_status("busy")


def mark_request_end() -> None:
    global ACTIVE_REQUESTS
    with ACTIVE_REQUESTS_LOCK:
        ACTIVE_REQUESTS = max(0, ACTIVE_REQUESTS - 1)
        if ACTIVE_REQUESTS == 0:
            set_worker_status("online")


def build_worker_payload(status: Optional[str] = None) -> Dict[str, Any]:
    payload = {
        "worker_id": WORKER_ID,
        "status": status or get_worker_status(),
        "last_seen": utc_now(),
    }
    optional_fields = {
        "device_type": WORKER_DEVICE_TYPE,
        "compute_type": WORKER_COMPUTE_TYPE,
        "gpu_name": WORKER_GPU_NAME,
        "gpu_model": WORKER_GPU_MODEL,
        "gpu": WORKER_GPU,
        "cpu_model": WORKER_CPU_MODEL,
        "cpu": WORKER_CPU,
    }
    for key, value in optional_fields.items():
        if value:
            payload[key] = value
    numeric_fields = {
        "gpu_memory_gb": normalize_float_env(WORKER_GPU_MEMORY_GB),
        "gpu_vram_gb": normalize_float_env(WORKER_GPU_VRAM_GB),
        "gpu_memory_mb": normalize_float_env(WORKER_GPU_MEMORY_MB),
        "gpu_vram_mb": normalize_float_env(WORKER_GPU_VRAM_MB),
        "ram_gb": normalize_float_env(WORKER_RAM_GB),
    }
    for key, value in numeric_fields.items():
        if value is not None:
            payload[key] = value
    return payload


def upsert_worker_heartbeat(supabase, status: Optional[str] = None) -> None:
    payload = build_worker_payload(status)
    supabase.table(WORKERS_TABLE).upsert(payload, on_conflict="worker_id").execute()


def start_heartbeat_thread(supabase):
    stop_event = threading.Event()

    def loop():
        while not stop_event.is_set():
            try:
                upsert_worker_heartbeat(supabase)
            except Exception as exc:
                log(f"Failed to update inference worker heartbeat: {exc}")
            stop_event.wait(HEARTBEAT_INTERVAL)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return stop_event, thread


def extract_storage_path(value: str):
    if not value:
        return None, None
    if value.startswith("storage:"):
        raw = value.split("storage:", 1)[1].lstrip("/")
        if "/" in raw:
            bucket, path = raw.split("/", 1)
            return bucket, path
        return SUPABASE_ARTIFACTS_BUCKET, raw
    if value.startswith("http"):
        parsed = urllib.parse.urlparse(value)
        prefixes = [
            "/storage/v1/object/public/",
            "/storage/v1/object/sign/",
            "/storage/v1/object/",
        ]
        for prefix in prefixes:
            if prefix in parsed.path:
                remainder = parsed.path.split(prefix, 1)[1]
                parts = remainder.split("/", 1)
                if len(parts) == 2:
                    return parts[0], urllib.parse.unquote(parts[1])
        return None, None
    return None, None


def download_storage_file(supabase, bucket: str, remote_path: str, local_path: str) -> str:
    result = supabase.storage.from_(bucket).download(remote_path)
    content = getattr(result, "data", result)
    if isinstance(content, dict) and "data" in content:
        content = content["data"]
    if not isinstance(content, (bytes, bytearray)):
        raise RuntimeError("Unexpected storage download response type.")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as handle:
        handle.write(content)
    return local_path


def download_http_file(url: str, local_path: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "saturnos-inference"})
    with urllib.request.urlopen(request, timeout=IMAGE_FETCH_TIMEOUT) as response:
        content = response.read()
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as handle:
        handle.write(content)
    return local_path


def download_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "saturnos-inference"})
    with urllib.request.urlopen(request, timeout=IMAGE_FETCH_TIMEOUT) as response:
        content = response.read(MAX_IMAGE_BYTES + 1)
    if len(content) > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds MAX_IMAGE_BYTES limit.")
    return content


def load_image_from_bytes(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as image:
        image = image.convert("RGB")
        return np.array(image)


def load_image_from_url(url: str) -> np.ndarray:
    data = download_bytes(url)
    return load_image_from_bytes(data)


def normalize_predictions(
    boxes_xyxy: np.ndarray,
    confidences: np.ndarray,
    class_ids: np.ndarray,
    class_names: Dict[int, str],
) -> List[Dict[str, Any]]:
    predictions = []
    for (x1, y1, x2, y2), conf, cls_id in zip(boxes_xyxy, confidences, class_ids):
        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)
        predictions.append(
            {
                "class_id": int(cls_id),
                "class": class_names.get(int(cls_id), str(int(cls_id))),
                "confidence": float(conf),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
            }
        )
    return predictions


class ModelEntry:
    def __init__(self, model: YOLO, source_url: str):
        self.model = model
        self.source_url = source_url
        self.last_used = time.time()


class ModelCache:
    def __init__(self):
        self._lock = threading.Lock()
        self._models: Dict[str, ModelEntry] = {}
        self._loading: Dict[str, threading.Event] = {}

    def _evict_if_needed(self):
        if len(self._models) <= MAX_MODELS:
            return
        oldest = sorted(self._models.items(), key=lambda item: item[1].last_used)
        while len(oldest) > 0 and len(self._models) > MAX_MODELS:
            run_id, _ = oldest.pop(0)
            self._models.pop(run_id, None)

    def get(self, run_id: str):
        with self._lock:
            entry = self._models.get(run_id)
            if entry:
                entry.last_used = time.time()
                return entry.model
        return None

    def get_or_load(self, run_id: str, run_record: Dict[str, Any], supabase):
        with self._lock:
            entry = self._models.get(run_id)
            if entry:
                entry.last_used = time.time()
                return entry.model
            if run_id in self._loading:
                event = self._loading[run_id]
            else:
                event = threading.Event()
                self._loading[run_id] = event
        if event.is_set():
            return self.get(run_id)
        if run_id in self._loading and self._loading[run_id] is not event:
            event = self._loading[run_id]
        if event.is_set():
            return self.get(run_id)
        if run_id in self._loading and self._loading[run_id] is event:
            if not event.is_set():
                try:
                    model, source_url = self._load_model(run_record, supabase)
                except Exception:
                    with self._lock:
                        event.set()
                        self._loading.pop(run_id, None)
                    raise
                with self._lock:
                    self._models[run_id] = ModelEntry(model, source_url)
                    self._models[run_id].last_used = time.time()
                    event.set()
                    self._loading.pop(run_id, None)
                    self._evict_if_needed()
                    return model
        event.wait()
        model = self.get(run_id)
        if not model:
            raise RuntimeError("Model failed to load.")
        return model

    def _load_model(self, run_record: Dict[str, Any], supabase):
        trained_model_url = run_record.get("trained_model_url")
        if not trained_model_url:
            raise RuntimeError("Run has no trained_model_url.")
        local_path = resolve_model_path(trained_model_url, run_record, supabase)
        log(f"Loading model for run {run_record['id']} from {local_path}")
        model = YOLO(local_path)
        return model, trained_model_url


class TritonBackend:
    def __init__(self, url: str):
        try:
            from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
        except Exception as exc:
            raise RuntimeError("tritonclient is required for Triton backend.") from exc
        self.client = InferenceServerClient(url=url, verbose=False)
        self.InferInput = InferInput
        self.InferRequestedOutput = InferRequestedOutput

    def infer(self, run_id: str, image: np.ndarray) -> Dict[str, Any]:
        model_name = TRITON_MODEL_NAME_TEMPLATE.format(run_id=run_id)
        input_format = TRITON_INPUT_FORMAT.lower()
        tensor = image
        if input_format == "nchw":
            tensor = np.transpose(image, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        dtype = TRITON_INPUT_DTYPE.lower()
        if dtype in ("fp32", "float32"):
            tensor = tensor.astype(np.float32) / 255.0
        elif dtype in ("fp16", "float16"):
            tensor = tensor.astype(np.float16) / 255.0
        else:
            tensor = tensor.astype(np.uint8)
        infer_input = self.InferInput(TRITON_INPUT_NAME, tensor.shape, tensor.dtype.name)
        infer_input.set_data_from_numpy(tensor)
        outputs = [self.InferRequestedOutput(TRITON_OUTPUT_NAME)]
        result = self.client.infer(model_name, inputs=[infer_input], outputs=outputs)
        output = result.as_numpy(TRITON_OUTPUT_NAME)
        return {"raw_output": output}


def parse_triton_output(output: Any, class_names: Dict[int, str]) -> List[Dict[str, Any]]:
    if output is None:
        return []
    if isinstance(output, dict):
        if "predictions" in output and isinstance(output["predictions"], list):
            return output["predictions"]
        if "annotations" in output and isinstance(output["annotations"], list):
            return output["annotations"]
    if isinstance(output, np.ndarray):
        if output.ndim == 2 and output.shape[1] >= 6:
            predictions = []
            for row in output:
                x1, y1, x2, y2, score, cls_id = row[:6]
                if TRITON_OUTPUT_FORMAT.lower() == "xywh":
                    x1, y1, w, h = row[:4]
                    x2 = x1 + w
                    y2 = y1 + h
                predictions.append(
                    {
                        "class_id": int(cls_id),
                        "class": class_names.get(int(cls_id), str(int(cls_id))),
                        "confidence": float(score),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    }
                )
            return predictions
    return []


def resolve_model_path(model_url: str, run_record: Dict[str, Any], supabase) -> str:
    os.makedirs(MODEL_ROOT, exist_ok=True)
    run_id = run_record["id"]
    local_dir = os.path.join(MODEL_ROOT, run_id)
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, "best.pt")
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path
    bucket, path = extract_storage_path(model_url)
    if bucket and path:
        download_storage_file(supabase, bucket, path, local_path)
        return local_path
    download_http_file(model_url, local_path)
    return local_path


def get_run(supabase, run_id: str) -> Dict[str, Any]:
    response = supabase.table(RUNS_TABLE).select("*").eq("id", run_id).single().execute()
    error = getattr(response, "error", None)
    if error:
        raise RuntimeError(str(error))
    return response.data


def extract_class_names(model: YOLO) -> Dict[int, str]:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {idx: name for idx, name in enumerate(names)}
    return {}


def store_predictions(
    supabase,
    run_id: str,
    annotations: List[Dict[str, Any]],
    step_image_id: Optional[str],
) -> None:
    if not step_image_id and not ALLOW_NULL_STEP_IMAGE:
        return
    payload = {
        "run_id": run_id,
        "step_image_id": step_image_id,
        "annotations": annotations,
    }
    supabase.table(PREDICTIONS_TABLE).insert(payload).execute()


def lookup_step_image_id(supabase, image_url: Optional[str]) -> Optional[str]:
    if not image_url:
        return None
    for column in ("image_url", "display_url", "thumbnail_url"):
        response = supabase.table(STEP_IMAGES_TABLE).select("id").eq(column, image_url).limit(1).execute()
        data = response.data or []
        if data:
            return data[0].get("id")
    return None


def resolve_request_options(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "confidence": payload.get("confidence"),
        "iou": payload.get("iou"),
        "imgsz": payload.get("imgsz"),
        "max_det": payload.get("max_det"),
        "step_image_id": payload.get("step_image_id") or payload.get("image_id"),
    }


def run_ultralytics(model: YOLO, image: np.ndarray, options: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = model.predict(
        source=image,
        conf=options.get("confidence", DEFAULT_CONFIDENCE),
        iou=options.get("iou", DEFAULT_IOU),
        imgsz=options.get("imgsz", DEFAULT_IMAGE_SIZE),
        max_det=options.get("max_det", DEFAULT_MAX_DETECTIONS),
        device=DEVICE,
        verbose=False,
    )
    predictions: List[Dict[str, Any]] = []
    class_names = extract_class_names(model)
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        boxes_xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy()
        predictions.extend(normalize_predictions(boxes_xyxy, confs, cls_ids, class_names))
    return predictions


def run_triton(triton: TritonBackend, run_id: str, image: np.ndarray, class_names: Dict[int, str]):
    raw = triton.infer(run_id, image)
    predictions = parse_triton_output(raw.get("raw_output"), class_names)
    return predictions, raw.get("raw_output")


def build_supabase_client():
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required.")
    storage_url = f"{SUPABASE_URL}storage/v1/"
    os.environ.setdefault("SUPABASE_STORAGE_URL", storage_url)
    if ClientOptions:
        try:
            return create_client(
                SUPABASE_URL,
                SUPABASE_SERVICE_ROLE_KEY,
                options=ClientOptions(storage_url=storage_url),
            )
        except Exception:
            return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def infer_backend(model_cache: ModelCache, supabase, run_id: str, image: np.ndarray, options: Dict[str, Any]):
    if INFERENCE_BACKEND == "triton":
        if not TRITON_URL:
            raise RuntimeError("TRITON_URL is required for Triton backend.")
        triton = TritonBackend(TRITON_URL)
        class_names = {}
        predictions, raw_output = run_triton(triton, run_id, image, class_names)
        return predictions, raw_output
    run_record = get_run(supabase, run_id)
    model = model_cache.get_or_load(run_id, run_record, supabase)
    predictions = run_ultralytics(model, image, options)
    return predictions, None


def update_deployment_status(supabase, run_id: str, payload: Dict[str, Any]) -> None:
    supabase.table(RUNS_TABLE).update(payload).eq("id", run_id).execute()


def deployment_loop(model_cache: ModelCache, supabase) -> None:
    while True:
        if not POLL_DEPLOYMENTS:
            time.sleep(DEPLOYMENT_POLL_INTERVAL)
            continue
        try:
            response = (
                supabase.table(RUNS_TABLE)
                .select("id, trained_model_url, deployment_status, is_deployed")
                .in_("deployment_status", ["deploying", "deployed"])
                .execute()
            )
            runs = response.data or []
            for run in runs:
                run_id = run["id"]
                if not run.get("trained_model_url"):
                    continue
                try:
                    model_cache.get_or_load(run_id, run, supabase)
                except Exception as exc:
                    log(f"Deployment load failed for {run_id}: {exc}")
                    if REGISTER_DEPLOYMENTS and run.get("deployment_status") == "deploying":
                        update_deployment_status(
                            supabase,
                            run_id,
                            {"deployment_status": "deployment_failed", "error_message": str(exc)},
                        )
                    continue
                if REGISTER_DEPLOYMENTS and run.get("deployment_status") == "deploying":
                    deployment_url = None
                    if PUBLIC_BASE_URL:
                        deployment_url = f"{PUBLIC_BASE_URL}/models/{run_id}/predict"
                    updates = {
                        "deployment_status": "deployed",
                        "is_deployed": True,
                        "deployment_date": utc_now(),
                    }
                    if deployment_url:
                        updates["deployment_url"] = deployment_url
                    update_deployment_status(supabase, run_id, updates)
                elif REGISTER_DEPLOYMENTS and PUBLIC_BASE_URL and OVERWRITE_DEPLOYMENT_URL:
                    update_deployment_status(
                        supabase,
                        run_id,
                        {"deployment_url": f"{PUBLIC_BASE_URL}/models/{run_id}/predict"},
                    )
        except Exception as exc:
            log(f"Deployment poll error: {exc}")
        time.sleep(DEPLOYMENT_POLL_INTERVAL)


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": utc_now(), "worker_id": WORKER_ID}


@app.get("/models")
def list_models():
    supabase = build_supabase_client()
    response = (
        supabase.table(RUNS_TABLE)
        .select("id, run_name, base_model, deployment_status, is_deployed, trained_model_url")
        .in_("deployment_status", ["deploying", "deployed"])
        .execute()
    )
    return {"models": response.data or []}


@app.post("/models/{run_id}/predict")
async def predict(
    run_id: str,
    request: Request,
    image: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),
    save: Optional[bool] = Query(default=None),
):
    supabase = build_supabase_client()
    options: Dict[str, Any] = {}
    image_bytes = None
    image_url = None
    step_image_id = None
    if image or file:
        upload = image or file
        image_bytes = await upload.read()
    else:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            payload = await request.json()
            image_url = payload.get("image_url") or payload.get("url")
            options = resolve_request_options(payload)
            step_image_id = options.get("step_image_id")
        else:
            payload = await request.form()
            image_url = payload.get("image_url") or payload.get("url")
            step_image_id = payload.get("step_image_id") or payload.get("image_id")
    if image_bytes:
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds MAX_IMAGE_BYTES limit.")
        image_array = load_image_from_bytes(image_bytes)
    elif image_url:
        image_array = load_image_from_url(image_url)
    else:
        raise HTTPException(status_code=400, detail="No image provided.")
    if not step_image_id and image_url:
        step_image_id = lookup_step_image_id(supabase, image_url)
    mark_request_start()
    try:
        start = time.time()
        model_cache = app.state.model_cache
        try:
            predictions, raw_output = infer_backend(model_cache, supabase, run_id, image_array, options)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        duration_ms = int((time.time() - start) * 1000)
        should_save = SAVE_PREDICTIONS if save is None else bool(save)
        if should_save:
            try:
                store_predictions(supabase, run_id, predictions, step_image_id)
            except Exception as exc:
                log(f"Failed to store predictions: {exc}")
        response = {
            "timestamp": utc_now(),
            "run_id": run_id,
            "processing_time_ms": duration_ms,
            "predictions": predictions,
        }
        if raw_output is not None:
            response["raw_output"] = raw_output.tolist() if isinstance(raw_output, np.ndarray) else raw_output
        return JSONResponse(response)
    finally:
        mark_request_end()


@app.post("/models/{run_id}/reload")
def reload_model(run_id: str):
    model_cache: ModelCache = app.state.model_cache
    with model_cache._lock:
        model_cache._models.pop(run_id, None)
    return {"status": "ok", "run_id": run_id}


def start_background_tasks():
    model_cache = ModelCache()
    app.state.model_cache = model_cache
    supabase = build_supabase_client()
    set_worker_status("online")
    try:
        upsert_worker_heartbeat(supabase, status="online")
    except Exception as exc:
        log(f"Failed to send initial inference heartbeat: {exc}")
    start_heartbeat_thread(supabase)
    if POLL_DEPLOYMENTS:
        thread = threading.Thread(target=deployment_loop, args=(model_cache, supabase), daemon=True)
        thread.start()
    log(f"Inference worker started (worker_id={WORKER_ID}, backend={INFERENCE_BACKEND})")


@app.on_event("startup")
def on_startup():
    start_background_tasks()


def main():
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run("inference_service:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
