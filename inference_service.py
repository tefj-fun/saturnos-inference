import io
import os
import shlex
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from collections import deque
from datetime import datetime, timezone
import platform
import shutil
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

ENV_PATH = os.path.join(os.path.dirname(__file__), ".env.local")
preload_ld_library_path = os.environ.get("LD_LIBRARY_PATH")
load_dotenv(ENV_PATH, override=True)
if preload_ld_library_path and os.getenv("LD_LIBRARY_PATH"):
    current_ld_library_path = os.environ["LD_LIBRARY_PATH"]
    if preload_ld_library_path not in current_ld_library_path:
        os.environ["LD_LIBRARY_PATH"] = f"{current_ld_library_path}:{preload_ld_library_path}"
if os.getenv("LD_LIBRARY_PATH") and os.getenv("_INFERENCE_REEXEC") != "1":
    os.environ["_INFERENCE_REEXEC"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
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

ADMIN_ENABLED = os.getenv("ADMIN_ENABLED", "1").lower() in ("1", "true", "yes")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()
ADMIN_LOG_BUFFER_SIZE = int(os.getenv("ADMIN_LOG_BUFFER_SIZE", "2000"))
ADMIN_LOG_TAIL_DEFAULT = int(os.getenv("ADMIN_LOG_TAIL_DEFAULT", "200"))
ADMIN_REQUEST_HISTORY_SIZE = int(os.getenv("ADMIN_REQUEST_HISTORY_SIZE", "200"))

CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "")
if not CORS_ALLOW_ORIGINS:
    CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "")
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

AUTO_UPDATE_ENABLED = os.getenv("AUTO_UPDATE_ENABLED", "0").lower() in ("1", "true", "yes")
AUTO_UPDATE_INTERVAL = int(os.getenv("AUTO_UPDATE_INTERVAL", "300"))
AUTO_UPDATE_REMOTE = os.getenv("AUTO_UPDATE_REMOTE", "origin")
AUTO_UPDATE_BRANCH = os.getenv("AUTO_UPDATE_BRANCH", "").strip()
AUTO_UPDATE_ALLOW_DIRTY = os.getenv("AUTO_UPDATE_ALLOW_DIRTY", "0").lower() in ("1", "true", "yes")
AUTO_UPDATE_WAIT_FOR_IDLE = os.getenv("AUTO_UPDATE_WAIT_FOR_IDLE", "1").lower() in ("1", "true", "yes")
AUTO_UPDATE_MAX_WAIT = int(os.getenv("AUTO_UPDATE_MAX_WAIT", "60"))
AUTO_UPDATE_RESTART_COMMAND = os.getenv("AUTO_UPDATE_RESTART_COMMAND", "").strip()

WORKER_STATUS_LOCK = threading.Lock()
ACTIVE_REQUESTS_LOCK = threading.Lock()
WORKER_STATUS = "online"
ACTIVE_REQUESTS = 0

AUTO_UPDATE_LOCK = threading.Lock()
AUTO_UPDATE_IN_PROGRESS = False

PROCESS_STARTED_AT = datetime.now(timezone.utc).isoformat()
PROCESS_START_TIME = time.time()

LOG_LOCK = threading.Lock()
LOG_SEQ = 0
LOG_BUFFER: "deque[Dict[str, Any]]" = deque(maxlen=max(0, ADMIN_LOG_BUFFER_SIZE))

REQUEST_HISTORY_LOCK = threading.Lock()
REQUEST_SEQ = 0
REQUEST_HISTORY: "deque[Dict[str, Any]]" = deque(maxlen=max(0, ADMIN_REQUEST_HISTORY_SIZE))

LAST_HEARTBEAT_AT: Optional[str] = None
LAST_HEARTBEAT_ERROR: Optional[str] = None

LAST_DEPLOYMENT_POLL_AT: Optional[str] = None
LAST_DEPLOYMENT_POLL_OK_AT: Optional[str] = None
LAST_DEPLOYMENT_POLL_ERROR: Optional[str] = None

AUTO_UPDATE_LAST_CHECK_AT: Optional[str] = None
AUTO_UPDATE_LAST_RESULT: Optional[str] = None

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


@app.exception_handler(HTTPException)
def http_exception_handler(_request: Request, exc: HTTPException):
    detail = exc.detail
    message = detail if isinstance(detail, str) else str(detail)
    return JSONResponse(status_code=exc.status_code, content={"error": message})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    global LOG_SEQ
    timestamp = utc_now()
    with LOG_LOCK:
        LOG_SEQ += 1
        if LOG_BUFFER.maxlen and LOG_BUFFER.maxlen > 0:
            LOG_BUFFER.append({"seq": LOG_SEQ, "timestamp": timestamp, "message": message})
    print(f"[{timestamp}] {message}", flush=True)


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


def record_request(
    *,
    run_id: str,
    duration_ms: int,
    status_code: int,
    image_source: str,
    saved_predictions: bool,
) -> None:
    global REQUEST_SEQ
    timestamp = utc_now()
    with REQUEST_HISTORY_LOCK:
        REQUEST_SEQ += 1
        if REQUEST_HISTORY.maxlen and REQUEST_HISTORY.maxlen > 0:
            REQUEST_HISTORY.append(
                {
                    "seq": REQUEST_SEQ,
                    "timestamp": timestamp,
                    "run_id": run_id,
                    "duration_ms": duration_ms,
                    "status_code": status_code,
                    "image_source": image_source,
                    "saved_predictions": saved_predictions,
                }
            )


def run_git_command(args: List[str], cwd: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    return subprocess.run(
        ["git"] + args,
        cwd=cwd,
        text=True,
        capture_output=True,
        env=env,
    )


def is_git_repo(repo_root: str) -> bool:
    result = run_git_command(["rev-parse", "--is-inside-work-tree"], repo_root)
    return result.returncode == 0 and result.stdout.strip() == "true"


def repo_is_dirty(repo_root: str) -> bool:
    result = run_git_command(["status", "--porcelain"], repo_root)
    if result.returncode != 0:
        return True
    return bool(result.stdout.strip())


def resolve_update_target(repo_root: str):
    branch = AUTO_UPDATE_BRANCH
    if not branch:
        result = run_git_command(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
        if result.returncode != 0:
            log(f"Auto-update skipped: unable to detect branch ({result.stderr.strip()}).")
            return None, None, None, None
        branch = result.stdout.strip()
    if not branch or branch == "HEAD":
        log("Auto-update skipped: detached HEAD or unknown branch.")
        return None, None, None, None
    upstream_result = run_git_command(
        ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], repo_root
    )
    if upstream_result.returncode == 0:
        upstream = upstream_result.stdout.strip()
        remote = upstream.split("/", 1)[0] if "/" in upstream else AUTO_UPDATE_REMOTE
        return branch, upstream, remote, True
    if AUTO_UPDATE_REMOTE:
        upstream = f"{AUTO_UPDATE_REMOTE}/{branch}"
        return branch, upstream, AUTO_UPDATE_REMOTE, False
    log("Auto-update skipped: no upstream branch configured.")
    return None, None, None, None


def git_ref(repo_root: str, ref: str) -> Optional[str]:
    result = run_git_command(["rev-parse", ref], repo_root)
    if result.returncode != 0:
        log(f"Auto-update skipped: unable to resolve git ref {ref}.")
        return None
    return result.stdout.strip()


def check_for_git_update(repo_root: str):
    branch, upstream, remote, uses_tracking = resolve_update_target(repo_root)
    if not upstream:
        return None
    fetch_args = ["fetch"]
    if remote:
        fetch_args.append(remote)
    fetch_result = run_git_command(fetch_args, repo_root)
    if fetch_result.returncode != 0:
        log(f"Auto-update fetch failed: {fetch_result.stderr.strip() or fetch_result.stdout.strip()}")
        return None
    local = git_ref(repo_root, "HEAD")
    remote_ref = git_ref(repo_root, upstream)
    if not local or not remote_ref:
        return None
    if local == remote_ref:
        return None
    ff_result = run_git_command(["merge-base", "--is-ancestor", "HEAD", upstream], repo_root)
    if ff_result.returncode != 0:
        log("Auto-update skipped: remote has diverged from local.")
        return None
    return branch, uses_tracking


def wait_for_idle(timeout_seconds: int) -> bool:
    if not AUTO_UPDATE_WAIT_FOR_IDLE:
        return True
    start = time.time()
    while True:
        with ACTIVE_REQUESTS_LOCK:
            active = ACTIVE_REQUESTS
        if active == 0:
            return True
        if time.time() - start >= timeout_seconds:
            return False
        time.sleep(1)


def restart_after_update(repo_root: str) -> None:
    if AUTO_UPDATE_RESTART_COMMAND:
        command = shlex.split(AUTO_UPDATE_RESTART_COMMAND)
        if command:
            result = subprocess.run(command, cwd=repo_root, capture_output=True, text=True)
            if result.returncode == 0:
                log("Auto-update restart command succeeded; exiting.")
                os._exit(0)
            error = result.stderr.strip() or result.stdout.strip() or "unknown error"
            log(f"Auto-update restart command failed: {error}")
        else:
            log("Auto-update restart command is empty; falling back to self-restart.")
    log("Auto-update restarting process.")
    os.execv(sys.executable, [sys.executable] + sys.argv)


def perform_auto_update(repo_root: str, branch: str, uses_tracking: bool) -> None:
    set_worker_status("updating")
    if not AUTO_UPDATE_ALLOW_DIRTY and repo_is_dirty(repo_root):
        log("Auto-update skipped: repo has uncommitted changes.")
        set_worker_status("online")
        return
    idle_ok = wait_for_idle(AUTO_UPDATE_MAX_WAIT)
    if not idle_ok:
        log("Auto-update proceeding with active requests.")
    pull_args = ["pull", "--ff-only"]
    if not uses_tracking:
        pull_args.extend([AUTO_UPDATE_REMOTE, branch])
    pull_result = run_git_command(pull_args, repo_root)
    if pull_result.returncode != 0:
        error = pull_result.stderr.strip() or pull_result.stdout.strip() or "unknown error"
        log(f"Auto-update pull failed: {error}")
        set_worker_status("online")
        return
    log("Auto-update pulled latest changes.")
    restart_after_update(repo_root)


def start_auto_update_thread():
    if not AUTO_UPDATE_ENABLED or AUTO_UPDATE_INTERVAL <= 0:
        return None
    repo_root = os.path.dirname(os.path.abspath(__file__))

    def loop():
        global AUTO_UPDATE_IN_PROGRESS, AUTO_UPDATE_LAST_CHECK_AT, AUTO_UPDATE_LAST_RESULT
        try:
            if not is_git_repo(repo_root):
                log("Auto-update disabled: not a git repository.")
                return
        except Exception as exc:
            log(f"Auto-update disabled: git unavailable ({exc}).")
            return
        log(f"Auto-update enabled (interval={AUTO_UPDATE_INTERVAL}s).")
        while True:
            try:
                AUTO_UPDATE_LAST_CHECK_AT = utc_now()
                update = check_for_git_update(repo_root)
                if update:
                    AUTO_UPDATE_LAST_RESULT = "update_available"
                    branch, uses_tracking = update
                    with AUTO_UPDATE_LOCK:
                        if AUTO_UPDATE_IN_PROGRESS:
                            continue
                        AUTO_UPDATE_IN_PROGRESS = True
                    try:
                        perform_auto_update(repo_root, branch, uses_tracking)
                    finally:
                        AUTO_UPDATE_IN_PROGRESS = False
                        set_worker_status("online")
                else:
                    AUTO_UPDATE_LAST_RESULT = "no_update"
            except Exception as exc:
                AUTO_UPDATE_LAST_RESULT = f"error: {exc}"
                log(f"Auto-update check failed: {exc}")
            time.sleep(AUTO_UPDATE_INTERVAL)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return thread


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
    global LAST_HEARTBEAT_AT, LAST_HEARTBEAT_ERROR
    payload = build_worker_payload(status)
    supabase.table(WORKERS_TABLE).upsert(payload, on_conflict="worker_id").execute()
    LAST_HEARTBEAT_AT = payload.get("last_seen")
    LAST_HEARTBEAT_ERROR = None


def start_heartbeat_thread(supabase):
    stop_event = threading.Event()

    def loop():
        while not stop_event.is_set():
            try:
                upsert_worker_heartbeat(supabase)
            except Exception as exc:
                global LAST_HEARTBEAT_ERROR
                LAST_HEARTBEAT_ERROR = str(exc)
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
        self.loaded_at = time.time()
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

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            models = []
            for run_id, entry in self._models.items():
                models.append(
                    {
                        "run_id": run_id,
                        "source_url": entry.source_url,
                        "loaded_at": entry.loaded_at,
                        "last_used": entry.last_used,
                    }
                )
            loading = list(self._loading.keys())
        models.sort(key=lambda item: item.get("last_used") or 0, reverse=True)
        return {"models": models, "loading": loading, "max_models": MAX_MODELS}

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


def option_or_default(options: Dict[str, Any], key: str, default: Any) -> Any:
    value = options.get(key)
    return default if value is None else value


def run_ultralytics(model: YOLO, image: np.ndarray, options: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = model.predict(
        source=image,
        conf=option_or_default(options, "confidence", DEFAULT_CONFIDENCE),
        iou=option_or_default(options, "iou", DEFAULT_IOU),
        imgsz=option_or_default(options, "imgsz", DEFAULT_IMAGE_SIZE),
        max_det=option_or_default(options, "max_det", DEFAULT_MAX_DETECTIONS),
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
    global LAST_DEPLOYMENT_POLL_AT, LAST_DEPLOYMENT_POLL_OK_AT, LAST_DEPLOYMENT_POLL_ERROR
    while True:
        if not POLL_DEPLOYMENTS:
            time.sleep(DEPLOYMENT_POLL_INTERVAL)
            continue
        try:
            LAST_DEPLOYMENT_POLL_AT = utc_now()
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
            LAST_DEPLOYMENT_POLL_ERROR = None
            LAST_DEPLOYMENT_POLL_OK_AT = LAST_DEPLOYMENT_POLL_AT
        except Exception as exc:
            LAST_DEPLOYMENT_POLL_ERROR = str(exc)
            log(f"Deployment poll error: {exc}")
        time.sleep(DEPLOYMENT_POLL_INTERVAL)


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": utc_now(), "worker_id": WORKER_ID}


def iso_from_epoch(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def format_duration(seconds: int) -> str:
    seconds = max(0, int(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def require_admin(request: Request) -> None:
    if not ADMIN_ENABLED:
        raise HTTPException(status_code=404, detail="Not found")
    if not ADMIN_TOKEN:
        return
    header = request.headers.get("authorization", "")
    token = ""
    if header.lower().startswith("bearer "):
        token = header.split(" ", 1)[1].strip()
    if not token:
        token = request.query_params.get("token", "").strip()
    if not token or token != ADMIN_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )


ADMIN_DASHBOARD_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>SaturnOS Inference Admin</title>
    <style>
      :root { --bg:#0b1020; --panel:#111a33; --muted:#9aa7c7; --text:#e7ecff; --border:#26345f; --ok:#37d67a; --warn:#f6c177; --bad:#ff5c7a; }
      html, body { height:100%; }
      body { margin:0; background:linear-gradient(180deg,#070b18,#0b1020 40%,#070b18); color:var(--text); font:14px/1.45 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial; }
      .wrap { max-width:1200px; margin:0 auto; padding:18px; }
      .top { display:flex; align-items:center; justify-content:space-between; gap:12px; }
      .title { font-size:18px; font-weight:700; letter-spacing:0.2px; }
      .pill { display:inline-flex; align-items:center; gap:8px; padding:6px 10px; border:1px solid var(--border); border-radius:999px; background:rgba(17,26,51,0.7); }
      .dot { width:9px; height:9px; border-radius:999px; background:var(--muted); box-shadow:0 0 0 3px rgba(154,167,199,0.15); }
      .dot.ok { background:var(--ok); box-shadow:0 0 0 3px rgba(55,214,122,0.15); }
      .dot.warn { background:var(--warn); box-shadow:0 0 0 3px rgba(246,193,119,0.15); }
      .dot.bad { background:var(--bad); box-shadow:0 0 0 3px rgba(255,92,122,0.18); }
      .actions { display:flex; gap:8px; flex-wrap:wrap; }
      button { cursor:pointer; border:1px solid var(--border); border-radius:10px; background:rgba(17,26,51,0.7); color:var(--text); padding:7px 10px; }
      button:hover { border-color:#3b4f86; }
      .grid { display:grid; grid-template-columns: 1fr 1fr; gap:12px; margin-top:12px; }
      @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
      .card { border:1px solid var(--border); border-radius:14px; background:rgba(17,26,51,0.65); overflow:hidden; }
      .card h2 { margin:0; padding:10px 12px; font-size:13px; letter-spacing:0.3px; text-transform:uppercase; color:var(--muted); border-bottom:1px solid var(--border); }
      .card .body { padding:12px; }
      .kv { display:grid; grid-template-columns: 160px 1fr; gap:6px 10px; }
      .k { color:var(--muted); }
      code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace; font-size: 12px; }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace; font-size: 12px; }
      table { width:100%; border-collapse:collapse; }
      th, td { text-align:left; padding:8px 8px; border-bottom:1px solid rgba(38,52,95,0.8); vertical-align:top; }
      th { color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:0.25px; }
      .logs { height: 360px; overflow:auto; background:rgba(7,11,24,0.55); border:1px solid rgba(38,52,95,0.7); border-radius:10px; padding:10px; }
      .login { display:none; gap:8px; margin-top:10px; }
      input { border:1px solid var(--border); border-radius:10px; padding:8px 10px; background:rgba(7,11,24,0.55); color:var(--text); width: 320px; }
      .muted { color:var(--muted); }
      .row { display:flex; align-items:center; justify-content:space-between; gap:12px; }
      .small { font-size:12px; }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="top">
        <div>
          <div class="title">SaturnOS Inference Admin</div>
          <div class="muted small">Dashboard polls <code>/admin/api/status</code> and <code>/admin/api/logs</code>.</div>
        </div>
        <div class="actions">
          <button id="btnToken">Set token</button>
          <button id="btnPause">Pause</button>
          <button id="btnClearLogs">Clear view</button>
        </div>
      </div>

      <div class="login" id="login">
        <input id="tokenInput" placeholder="ADMIN_TOKEN (stored in localStorage)" />
        <button id="btnSaveToken">Save</button>
        <div class="muted small">If <code>ADMIN_TOKEN</code> is set on the server, requests must include <code>Authorization: Bearer &lt;token&gt;</code>.</div>
      </div>

      <div class="grid">
        <div class="card">
          <h2>Worker</h2>
          <div class="body">
            <div class="row">
              <div class="pill"><span id="statusDot" class="dot"></span><span id="statusText">—</span></div>
              <div class="pill"><span class="muted">uptime</span> <span id="uptimeText" class="mono">—</span></div>
            </div>
            <div style="height:10px"></div>
            <div class="kv">
              <div class="k">worker_id</div><div class="mono" id="workerId">—</div>
              <div class="k">active_requests</div><div class="mono" id="activeRequests">—</div>
              <div class="k">backend</div><div class="mono" id="backend">—</div>
              <div class="k">device</div><div class="mono" id="device">—</div>
              <div class="k">model_root</div><div class="mono" id="modelRoot">—</div>
              <div class="k">max_models</div><div class="mono" id="maxModels">—</div>
              <div class="k">heartbeat</div><div class="mono" id="heartbeat">—</div>
              <div class="k">deploy_poll</div><div class="mono" id="deployPoll">—</div>
              <div class="k">auto_update</div><div class="mono" id="autoUpdate">—</div>
            </div>
          </div>
        </div>

        <div class="card">
          <h2>Model Cache</h2>
          <div class="body">
            <div class="muted small">Loaded models currently cached in-process.</div>
            <div style="height:10px"></div>
            <table>
              <thead>
                <tr>
                  <th>run_id</th>
                  <th>loaded_at</th>
                  <th>last_used</th>
                  <th>source_url</th>
                  <th></th>
                </tr>
              </thead>
              <tbody id="modelsBody"></tbody>
            </table>
            <div style="height:8px"></div>
            <div class="muted small">Loading: <span id="loadingModels" class="mono">—</span></div>
          </div>
        </div>

        <div class="card" style="grid-column: 1 / -1;">
          <h2>Recent Requests</h2>
          <div class="body">
            <div class="muted small">Last few predictions handled by this process.</div>
            <div style="height:10px"></div>
            <table>
              <thead>
                <tr>
                  <th>timestamp</th>
                  <th>run_id</th>
                  <th>status</th>
                  <th>duration</th>
                  <th>source</th>
                  <th>saved</th>
                </tr>
              </thead>
              <tbody id="requestsBody"></tbody>
            </table>
          </div>
        </div>

        <div class="card" style="grid-column: 1 / -1;">
          <h2>Logs</h2>
          <div class="body">
            <div class="logs mono" id="logs"></div>
            <div style="height:8px"></div>
            <div class="muted small">Tip: use <code>ADMIN_LOG_BUFFER_SIZE</code> to adjust in-memory log retention.</div>
          </div>
        </div>
      </div>
    </div>

    <script>
      const $ = (id) => document.getElementById(id);
      let paused = false;
      let afterLogSeq = 0;
      const storageKey = "saturnos_inference_admin_token";

      function getToken() { return localStorage.getItem(storageKey) || ""; }
      function setToken(value) { localStorage.setItem(storageKey, value || ""); }

      function authHeaders() {
        const token = getToken();
        return token ? { "Authorization": `Bearer ${token}` } : {};
      }

      function showLogin(show) {
        $("login").style.display = show ? "flex" : "none";
        $("tokenInput").value = getToken();
      }

      async function fetchJson(url, opts = {}) {
        const res = await fetch(url, { ...opts, headers: { ...(opts.headers || {}), ...authHeaders() } });
        if (res.status === 401) {
          showLogin(true);
          throw new Error("unauthorized");
        }
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
      }

      function setStatusPill(status) {
        const dot = $("statusDot");
        dot.className = "dot";
        if (status === "online") dot.classList.add("ok");
        else if (status === "busy" || status === "updating") dot.classList.add("warn");
        else dot.classList.add("bad");
        $("statusText").textContent = status || "—";
      }

      function renderModels(models) {
        const body = $("modelsBody");
        body.innerHTML = "";
        for (const model of (models || [])) {
          const tr = document.createElement("tr");
          tr.innerHTML = `
            <td class="mono">${model.run_id || ""}</td>
            <td class="mono">${model.loaded_at || ""}</td>
            <td class="mono">${model.last_used || ""}</td>
            <td class="mono">${(model.source_url || "").slice(0, 80)}${(model.source_url || "").length > 80 ? "…" : ""}</td>
            <td><button data-run="${model.run_id}">Reload</button></td>
          `;
          tr.querySelector("button").addEventListener("click", async (e) => {
            const runId = e.target.getAttribute("data-run");
            try {
              await fetchJson(`/admin/api/models/${encodeURIComponent(runId)}/reload`, { method: "POST" });
            } catch (_) {}
          });
          body.appendChild(tr);
        }
      }

      function renderRequests(entries) {
        const body = $("requestsBody");
        body.innerHTML = "";
        for (const req of (entries || [])) {
          const tr = document.createElement("tr");
          tr.innerHTML = `
            <td class="mono">${req.timestamp || ""}</td>
            <td class="mono">${req.run_id || ""}</td>
            <td class="mono">${req.status_code ?? ""}</td>
            <td class="mono">${req.duration_ms ?? ""}ms</td>
            <td class="mono">${req.image_source || ""}</td>
            <td class="mono">${req.saved_predictions ? "yes" : "no"}</td>
          `;
          body.appendChild(tr);
        }
      }

      function appendLogs(entries) {
        const box = $("logs");
        const nearBottom = (box.scrollTop + box.clientHeight) >= (box.scrollHeight - 30);
        for (const e of (entries || [])) {
          const line = document.createElement("div");
          line.textContent = `[${e.timestamp}] ${e.message}`;
          box.appendChild(line);
        }
        if (nearBottom) box.scrollTop = box.scrollHeight;
      }

      async function poll() {
        if (paused) return;
        try {
          const status = await fetchJson("/admin/api/status");
          showLogin(false);
          setStatusPill(status.worker_status);
          $("uptimeText").textContent = status.uptime_human || "—";
          $("workerId").textContent = status.worker_id || "—";
          $("activeRequests").textContent = String(status.active_requests ?? "—");
          $("backend").textContent = status.inference_backend || "—";
          $("device").textContent = status.device || "—";
          $("modelRoot").textContent = status.model_root || "—";
          $("maxModels").textContent = String(status.max_models ?? "—");
          $("heartbeat").textContent = status.heartbeat?.last_ok_at ? `ok @ ${status.heartbeat.last_ok_at}` : (status.heartbeat?.last_error ? `error: ${status.heartbeat.last_error}` : "—");
          $("deployPoll").textContent = status.deployment_poll?.last_ok_at ? `ok @ ${status.deployment_poll.last_ok_at}` : (status.deployment_poll?.last_error ? `error: ${status.deployment_poll.last_error}` : "—");
          $("autoUpdate").textContent = status.auto_update?.enabled ? `enabled (${status.auto_update.last_result || "—"})` : "disabled";
          renderModels(status.model_cache?.models || []);
          $("loadingModels").textContent = (status.model_cache?.loading || []).join(", ") || "—";

          const requests = await fetchJson("/admin/api/requests?limit=20");
          renderRequests(requests.entries || []);

          const logs = await fetchJson(`/admin/api/logs?after=${afterLogSeq}`);
          afterLogSeq = logs.next_after || afterLogSeq;
          appendLogs(logs.entries || []);
        } catch (err) {
          if (String(err.message || "").includes("unauthorized")) return;
        }
      }

      $("btnToken").addEventListener("click", () => showLogin(true));
      $("btnSaveToken").addEventListener("click", () => { setToken($("tokenInput").value.trim()); showLogin(false); afterLogSeq = 0; });
      $("btnPause").addEventListener("click", () => { paused = !paused; $("btnPause").textContent = paused ? "Resume" : "Pause"; });
      $("btnClearLogs").addEventListener("click", () => { $("logs").innerHTML = ""; afterLogSeq = 0; });

      showLogin(false);
      poll();
      setInterval(poll, 2000);
    </script>
  </body>
</html>
""".strip()


@app.get("/admin")
def admin_dashboard():
    if not ADMIN_ENABLED:
        raise HTTPException(status_code=404, detail="Not found")
    return HTMLResponse(ADMIN_DASHBOARD_HTML)


@app.get("/admin/api/status")
def admin_status(request: Request):
    require_admin(request)
    with ACTIVE_REQUESTS_LOCK:
        active_requests = ACTIVE_REQUESTS
    model_cache: Optional[ModelCache] = getattr(app.state, "model_cache", None)
    model_snapshot = model_cache.snapshot() if model_cache else {"models": [], "loading": [], "max_models": MAX_MODELS}
    for item in model_snapshot.get("models", []):
        item["loaded_at"] = iso_from_epoch(item.get("loaded_at"))
        item["last_used"] = iso_from_epoch(item.get("last_used"))

    try:
        disk = shutil.disk_usage(MODEL_ROOT)
        disk_usage = {
            "total_bytes": int(disk.total),
            "used_bytes": int(disk.used),
            "free_bytes": int(disk.free),
        }
    except Exception:
        disk_usage = None

    with LOG_LOCK:
        current_log_seq = LOG_SEQ

    uptime_seconds = int(max(0, time.time() - PROCESS_START_TIME))
    return {
        "timestamp": utc_now(),
        "worker_id": WORKER_ID,
        "worker_status": get_worker_status(),
        "active_requests": active_requests,
        "process": {
            "pid": os.getpid(),
            "started_at": PROCESS_STARTED_AT,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "log_seq": current_log_seq,
        },
        "uptime_seconds": uptime_seconds,
        "uptime_human": format_duration(uptime_seconds),
        "inference_backend": INFERENCE_BACKEND,
        "device": DEVICE,
        "model_root": MODEL_ROOT,
        "max_models": MAX_MODELS,
        "disk_usage": disk_usage,
        "supabase": {
            "configured": bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY),
            "url": SUPABASE_URL,
            "artifacts_bucket": SUPABASE_ARTIFACTS_BUCKET,
        },
        "heartbeat": {"interval_seconds": HEARTBEAT_INTERVAL, "last_ok_at": LAST_HEARTBEAT_AT, "last_error": LAST_HEARTBEAT_ERROR},
        "deployment_poll": {
            "enabled": bool(POLL_DEPLOYMENTS),
            "interval_seconds": DEPLOYMENT_POLL_INTERVAL,
            "last_attempt_at": LAST_DEPLOYMENT_POLL_AT,
            "last_ok_at": LAST_DEPLOYMENT_POLL_OK_AT,
            "last_error": LAST_DEPLOYMENT_POLL_ERROR,
        },
        "auto_update": {
            "enabled": bool(AUTO_UPDATE_ENABLED),
            "interval_seconds": AUTO_UPDATE_INTERVAL,
            "in_progress": bool(AUTO_UPDATE_IN_PROGRESS),
            "last_check_at": AUTO_UPDATE_LAST_CHECK_AT,
            "last_result": AUTO_UPDATE_LAST_RESULT,
        },
        "model_cache": model_snapshot,
        "admin": {"enabled": bool(ADMIN_ENABLED), "auth_required": bool(ADMIN_TOKEN)},
    }


@app.get("/admin/api/logs")
def admin_logs(
    request: Request,
    limit: int = Query(default=ADMIN_LOG_TAIL_DEFAULT, ge=1, le=5000),
    after: int = Query(default=0, ge=0),
    format: str = Query(default="json"),
):
    require_admin(request)
    with LOG_LOCK:
        entries = [entry for entry in LOG_BUFFER if int(entry.get("seq", 0)) > after]
        if limit:
            entries = entries[-limit:]
        next_after = entries[-1]["seq"] if entries else after
    if format.lower() == "text":
        body = "\n".join([f"[{e['timestamp']}] {e['message']}" for e in entries])
        return PlainTextResponse(body)
    return {"entries": entries, "next_after": next_after}


@app.get("/admin/api/requests")
def admin_requests(
    request: Request,
    limit: int = Query(default=50, ge=1, le=5000),
):
    require_admin(request)
    with REQUEST_HISTORY_LOCK:
        entries = list(REQUEST_HISTORY)[-limit:]
    return {"entries": entries}


@app.post("/admin/api/models/{run_id}/reload")
def admin_reload_model(run_id: str, request: Request):
    require_admin(request)
    model_cache: Optional[ModelCache] = getattr(app.state, "model_cache", None)
    if not model_cache:
        raise HTTPException(status_code=503, detail="Model cache not initialized.")
    with model_cache._lock:
        model_cache._models.pop(run_id, None)
        model_cache._loading.pop(run_id, None)
    return {"status": "ok", "run_id": run_id}

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
    image_source = "unknown"
    if image or file:
        upload = image or file
        image_bytes = await upload.read()
        image_source = "upload"
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
        image_source = "url" if image_url else "unknown"
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
    infer_start = time.time()
    status_code = 500
    duration_ms = 0
    should_save = False
    try:
        model_cache = app.state.model_cache
        try:
            predictions, raw_output = infer_backend(model_cache, supabase, run_id, image_array, options)
        except Exception as exc:
            status_code = 500
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        duration_ms = int((time.time() - infer_start) * 1000)
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
        status_code = 200
        return JSONResponse(response)
    except HTTPException as exc:
        status_code = exc.status_code
        raise
    finally:
        if duration_ms == 0:
            duration_ms = int((time.time() - infer_start) * 1000)
        try:
            record_request(
                run_id=run_id,
                duration_ms=duration_ms,
                status_code=status_code,
                image_source=image_source,
                saved_predictions=bool(should_save and status_code == 200),
            )
        except Exception:
            pass
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
    start_auto_update_thread()
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
