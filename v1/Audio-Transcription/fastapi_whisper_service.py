import os
import uuid
import json
import time
import logging
import asyncio
import threading
from pathlib import Path
from typing import Optional, Dict, Any
import sys
import subprocess
import shutil
from collections import deque
try:
    import ai_summary  # type: ignore
except Exception:
    ai_summary = None  # type: ignore

from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from datetime import datetime, timezone
from pydantic import BaseModel, validator
import requests
from dotenv import load_dotenv
import re
from pydub import AudioSegment

# Optional parts
import boto3
import openai

# load .env
load_dotenv()

# Basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("transcribe_service")

# Try to import user's whisper+diarization class (put your script in whisper_diarizer.py)
try:
    from whisper_diarizer import AudioTranscriberWithDiarization
except Exception:
    # For the simplified flow, we don't require this import. The full flow can use it later.
    AudioTranscriberWithDiarization = None  # type: ignore

# Config
STORAGE = os.getenv("STORAGE", "local")  # 'local' or 's3'
AUDIO_TMP_DIR = Path(os.getenv("AUDIO_TMP_DIR", "./tmp_audio")).resolve()
PERSIST_AUDIO_DIR = Path(os.getenv("AUDIO_DIR", "./audio")).resolve()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output")).resolve()
AUDIO_TMP_DIR.mkdir(parents=True, exist_ok=True)
PERSIST_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

S3_BUCKET = os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION")
PUBLIC_APP_BASE_URL = "https://files.lead2424.com"  # Default domain - not reading from .env
IP_APP_BASE_URL = "http://16.52.197.95:8088"  # Default IP - not reading from .env
S3_CDN_BASE_URL = os.getenv("S3_CDN_BASE_URL", "https://files.lead2424.com")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# LeadConnector / HighLevel CRM config
LEADCONNECTOR_BASE_URL = os.getenv("LEADCONNECTOR_BASE_URL", "https://services.leadconnectorhq.com")
# Access token now comes from webhook payload as `token` (Make.com). We intentionally
# do not read a default from environment anymore.


# Whisper config defaults
MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")
DEVICE = os.getenv("DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")
NUM_THREADS = int(os.getenv("NUM_THREADS", "2"))

PRELOAD_MODELS = os.getenv("PRELOAD_MODELS", "false").lower() in ("1", "true", "yes")

# In-memory job store (replace with DB/Redis for production)
jobs: Dict[str, Dict[str, Any]] = {}

# Job queue management
job_queue = deque()
queue_lock = threading.Lock()
is_processing = False

# Initialize S3 client lazily
_s3_client = None

def get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=AWS_REGION,
        )
    return _s3_client

# Common field lists for logging
EXPECTED_INPUT_KEYS = [
    "audio",
    "location_id",
    "contact_id",
    "contact_first_name",
    "contact_last_name",
    "caller_name",
    "show_prompt",
    "token",
    "slug",
    "googlesheet",
    "company_name",
    "language",
    "make_url_out",
    "make_webhook_url",
    "client_webhook_url",
    "date_time",
]


def add_job_to_queue(job_id: str, payload: dict):
    """Add a job to the processing queue."""
    with queue_lock:
        job_queue.append((job_id, payload))
        jobs[job_id]["queue_position"] = len(job_queue)
        logger.info("Job %s added to queue at position %d", job_id, len(job_queue))


def process_queue():
    """Process jobs from the queue one at a time."""
    global is_processing
    
    while True:
        with queue_lock:
            if not job_queue:
                is_processing = False
                break
            job_id, payload = job_queue.popleft()
            is_processing = True
        
        # Update queue positions for remaining jobs
        with queue_lock:
            for i, (queued_job_id, _) in enumerate(job_queue):
                if queued_job_id in jobs:
                    jobs[queued_job_id]["queue_position"] = i + 1
        
        logger.info("Processing job %s from queue", job_id)
        process_job(job_id, payload)
        
        # Small delay before processing next job
        time.sleep(1)


def start_queue_processor():
    """Start the queue processor in a background thread."""
    def queue_worker():
        while True:
            try:
                process_queue()
                time.sleep(5)  # Check for new jobs every 5 seconds
            except Exception as e:
                logger.exception("Error in queue processor: %s", e)
                time.sleep(10)  # Wait longer on error
    
    thread = threading.Thread(target=queue_worker, daemon=True)
    thread.start()
    logger.info("Queue processor started")

# Instantiate a reusable transcriber object (kept for full flow; unused in simple mode)
_transcriber: Optional[AudioTranscriberWithDiarization] = None  # type: ignore

app = FastAPI(title="WhisperX + Diarization Transcription Service")

# Templates and static (mounted lazily if directories exist)
BASE_DIR = Path(__file__).parent.resolve()
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _compute_canonical_host() -> str:
    try:
        base = (PUBLIC_APP_BASE_URL or "").strip().rstrip("/")
        if base.startswith("http://"):
            return base[len("http://"):]
        if base.startswith("https://"):
            return base[len("https://"):]
        return base
    except Exception:
        return ""

_CANONICAL_HOST = _compute_canonical_host()

@app.middleware("http")
async def enforce_canonical_host(request: Request, call_next):
    """Redirect /view-session* requests to the canonical host if accessed via IP/other host.

    This ensures end-user links always use files.lead2424.com (or PUBLIC_APP_BASE_URL).
    """
    try:
        host = request.headers.get("host", "")
        path = request.url.path or ""
        if _CANONICAL_HOST and host and host != _CANONICAL_HOST and path.startswith("/view-session"):
            query = request.url.query
            target = f"{PUBLIC_APP_BASE_URL.rstrip('/')}{path}"
            if query:
                target = f"{target}?{query}"
            return RedirectResponse(url=target, status_code=308)
    except Exception:
        pass
    return await call_next(request)

class WebhookPayload(BaseModel):
    audio: str
    call_id: Optional[str] = None
    contact_id: Optional[str] = None
    # New fields from Make.com
    token: Optional[str] = None
    make_webhook_url: Optional[str] = None
    client_webhook_url: Optional[str] = None   # optional second webhook
    extra_webhook_urls: Optional[list[str]] = None  # optional list of additional webhooks
    # Google Sheets integration
    googlesheet: Optional[str] = None  # sheet id to echo back to Make.com
    # Tenant/customer routing - support both old and new field names
    customer_slug: Optional[str] = None  # e.g., "brix" to render /brix/view-session/{id}
    slug: Optional[str] = None  # alias for customer_slug (new client format)
    customer_id: Optional[str] = None    # used with CUSTOMER_SLUG_MAP if provided
    account_id: Optional[str] = None     # aliases supported for mapping
    location_id: Optional[str] = None
    caller_name: Optional[str] = None
    contact_first_name: Optional[str] = None
    contact_last_name: Optional[str] = None
    show_prompt: Optional[bool] = False
    language: Optional[str] = "en"
    company_name: Optional[str] = None
    date_time: Optional[str] = None
    no_diarization: Optional[bool] = False
    # allow additional fields; Pydantic will ignore unknowns unless configured otherwise
    make_url_out: Optional[str] = None  # optional custom outbound Make.com URL

    @validator('show_prompt', pre=True)
    def convert_show_prompt(cls, v):
        """Convert string 'true'/'false' to boolean for new client format."""
        if isinstance(v, str):
            return v.lower() in ('true', '1', 'yes', 'on')
        return v


@app.on_event("startup")
def startup_event():
    logger.info("Service starting up...")
    # Start the queue processor
    start_queue_processor()
    # Full flow init is skipped in simple mode


def detect_extension_from_content_type(content_type: str) -> str:
    if not content_type:
        return "mp3"
    content_type = content_type.lower()
    if "mpeg" in content_type or "mp3" in content_type:
        return "mp3"
    if "wav" in content_type:
        return "wav"
    if "mpeg" in content_type:
        return "mp3"
    if "ogg" in content_type or "opus" in content_type:
        return "ogg"
    return "mp3"


def probe_audio_duration_seconds(file_path: Path) -> Optional[float]:
    """Duration probe using pydub (requires ffmpeg/ffprobe installed)."""
    try:
        audio = AudioSegment.from_file(str(file_path))
        return len(audio) / 1000.0  # milliseconds -> seconds
    except Exception as e:
        logger.warning("Failed to probe audio duration with pydub: %s", e)
        return None


def format_seconds_mmss(seconds: Optional[float]) -> Optional[str]:
    """Format seconds as mm:ss with zero padding."""
    try:
        if seconds is None:
            return None
        total_seconds = int(seconds)
        minutes, secs = divmod(total_seconds, 60)
        return f"{minutes:02d}:{secs:02d}"
    except Exception:
        return None


def download_audio(url: str, dest_path: Path, auth: Optional[tuple] = None, headers: Optional[dict] = None, timeout: int = 60):
    # Ensure target directory exists
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as mk_ex:
        logger.exception("Failed to ensure tmp dir exists: %s", mk_ex)
    logger.info("Downloading audio from %s to %s", url, dest_path)
    with requests.get(url, stream=True, auth=auth, headers=headers, timeout=timeout) as r:
        r.raise_for_status()
        content_type = r.headers.get("content-type", "")
        # If dest_path doesn't have extension, try to set
        if dest_path.suffix == "":
            ext = detect_extension_from_content_type(content_type)
            dest_path = dest_path.with_suffix("." + ext)
            # Ensure directory remains ensured after suffix change
            dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    logger.info("Downloaded audio to %s", dest_path)
    return dest_path


def derive_readable_audio_name(url: str, fallback: str) -> str:
    """Derive a readable filename from the URL path parts, fallback to provided name."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        parts = [p for p in parsed.path.split("/") if p]
        if parts:
            # take last 2 parts if available (e.g., Recordings/RE...). Join with '_'.
            tail = parts[-2:]
            name = "_".join(tail)
            # ensure has extension
            if not os.path.splitext(name)[1]:
                name += ".wav"
            return name
    except Exception:
        pass
    return fallback


def upload_to_s3(local_path: Path, s3_key: str) -> str:
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET not set in env")
    s3 = get_s3_client()
    logger.info("Uploading %s to s3://%s/%s", local_path, S3_BUCKET, s3_key)
    
    # Determine content type based on file extension
    file_ext = local_path.suffix.lower()
    if file_ext in ['.wav', '.mp3', '.m4a', '.ogg']:
        content_type = 'audio/wav' if file_ext == '.wav' else f'audio/{file_ext[1:]}'
    elif file_ext == '.json':
        content_type = 'application/json'
    else:
        content_type = 'application/octet-stream'
    
    # Upload file with public read access
    s3.upload_file(
        str(local_path), 
        S3_BUCKET, 
        s3_key,
        ExtraArgs={
            'ACL': 'public-read',
            'ContentType': content_type
        }
    )
    
    # Generate direct public URL (not presigned)
    region = AWS_REGION or 'us-east-1'
    if region == 'us-east-1':
        url = f"https://{S3_BUCKET}.s3.amazonaws.com/{s3_key}"
    else:
        url = f"https://{S3_BUCKET}.s3.{region}.amazonaws.com/{s3_key}"
    
    logger.info("Uploaded to S3 with public access: %s", url)
    return url


def send_crm_note(contact_id: str, note_body: str, access_token: str, note_type: str = "general") -> bool:
    """Send a single note to CRM for a contact."""
    url = f"{LEADCONNECTOR_BASE_URL}/contacts/{contact_id}/notes"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Version": "2021-07-28",
        "Content-Type": "application/json",
    }
    payload_body = {"body": note_body}
    
    try:
        resp = requests.post(url, headers=headers, json=payload_body, timeout=30)
        if resp.ok:
            logger.info("Posted %s note to CRM for contact_id=%s", note_type, contact_id)
            return True
        else:
            logger.warning("CRM %s note post failed: %s %s", note_type, resp.status_code, resp.text)
            return False
    except Exception as crm_ex:
        logger.warning("Failed to POST CRM %s note: %s", note_type, crm_ex)
        return False


def _sanitize_slug(value: str) -> str:
    """Sanitize a slug to contain lowercase letters, digits, and dashes only."""
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9-]", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value

def _load_customer_slug_map() -> Dict[str, str]:
    """Load CUSTOMER_SLUG_MAP from env as JSON dict: {"account_id": "slug", ...}."""
    try:
        raw = os.getenv("CUSTOMER_SLUG_MAP", "").strip()
        if not raw:
            return {}
        data = json.loads(raw)
        if isinstance(data, dict):
            # sanitize values
            return {str(k): _sanitize_slug(str(v)) for k, v in data.items()}
    except Exception as e:
        logger.warning("Failed to parse CUSTOMER_SLUG_MAP: %s", e)
    return {}

def resolve_customer_slug_from_payload(payload: Dict[str, Any]) -> Optional[str]:
    """Resolve customer slug using payload fields or CUSTOMER_SLUG_MAP env.

    Priority:
      1) payload.customer_slug or payload.slug (used directly after sanitization)
      2) payload.customer_id/account_id/location_id looked up in CUSTOMER_SLUG_MAP
    """
    try:
        if not isinstance(payload, dict):
            return None
        # Check both customer_slug and slug (alias) - prioritize slug for new client format
        slug_value = payload.get("slug") or payload.get("customer_slug")
        if slug_value:
            return _sanitize_slug(str(slug_value))
        slug_map = _load_customer_slug_map()
        for key in ("customer_id", "account_id", "location_id"):
            identifier = payload.get(key)
            if identifier is None:
                continue
            identifier_str = str(identifier)
            if identifier_str in slug_map:
                return slug_map[identifier_str]
    except Exception:
        pass
    return None

def send_file_links_note(contact_id: str, job_data: dict, access_token: str) -> bool:
    """Send note with only GUI link."""
    # Get GUI link for the contact/session
    # Use entity_id for GUI link (with timestamp) to match S3 folder structure
    session_id = job_data.get("entity_id") or contact_id or job_data.get("call_id")
    gui_base = PUBLIC_APP_BASE_URL  # Use default domain from code
    
    # Debug logging
    logger.info(f"Creating GUI link - contact_id: '{contact_id}', session_id: '{session_id}', gui_base: '{gui_base}'")
    
    gui_link = None
    customer_slug = resolve_customer_slug_from_payload(job_data.get("payload", {}))
    if session_id and gui_base:
        ts = int(time.time())
        if customer_slug:
            gui_link = f"{gui_base.rstrip('/')}/view-session/{customer_slug}/{session_id}?t={ts}"
        else:
            gui_link = f"{gui_base.rstrip('/')}/view-session/{session_id}?t={ts}"
        logger.info(f"Generated GUI link: '{gui_link}'")

    if not gui_link:
        logger.warning("No GUI link available to send in file links note")
        return False

    note_body = f"🌐 **View Call Session**\n\n{gui_link}"
    return send_crm_note(contact_id, note_body, access_token, "file_links")

def build_gui_link(session_id: Optional[str], customer_slug: Optional[str] = None) -> Optional[str]:
    """Construct a GUI link for the given session/contact id using PUBLIC_APP_BASE_URL.
    session_id should be the entity_id (with timestamp) to match S3 folder structure."""
    try:
        gui_base = PUBLIC_APP_BASE_URL  # Use default domain from code
        logger.info("Building GUI link - session_id: %s, customer_slug: %s, gui_base: %s", session_id, customer_slug, gui_base)
        if session_id and gui_base:
            if customer_slug:
                link = f"{gui_base.rstrip('/')}/view-session/{customer_slug}/{session_id}"
            else:
                link = f"{gui_base.rstrip('/')}/view-session/{session_id}"
            logger.info("Generated GUI link: %s", link)
            return link
    except Exception as e:
        logger.warning("Error building GUI link: %s", e)
    return None

def build_gui_links(session_id: Optional[str], customer_slug: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Construct both domain and IP GUI links for the given session/contact id.
    Returns a dict with 'gui_link' (domain) and 'gui_link_ip' (IP address).
    session_id should be the entity_id (with timestamp) to match S3 folder structure."""
    result = {"gui_link": None, "gui_link_ip": None}
    try:
        # Domain link (using PUBLIC_APP_BASE_URL) - use default domain from code
        gui_base = PUBLIC_APP_BASE_URL
        if session_id and gui_base:
            if customer_slug:
                result["gui_link"] = f"{gui_base.rstrip('/')}/view-session/{customer_slug}/{session_id}"
            else:
                result["gui_link"] = f"{gui_base.rstrip('/')}/view-session/{session_id}"
        
        # IP link (using IP_APP_BASE_URL) - use default IP from code
        ip_base = IP_APP_BASE_URL
        if session_id and ip_base:
            if customer_slug:
                result["gui_link_ip"] = f"{ip_base.rstrip('/')}/view-session/{customer_slug}/{session_id}"
            else:
                result["gui_link_ip"] = f"{ip_base.rstrip('/')}/view-session/{session_id}"
        
        logger.info("Generated GUI links - domain: %s, IP: %s", result["gui_link"], result["gui_link_ip"])
        logger.info("GUI link base: %s, IP link base: %s", gui_base, ip_base)
    except Exception as e:
        logger.warning("Error building GUI links: %s", e)
    return result

def send_make_webhook(job_data: dict, contact_id: Optional[str], call_id: Optional[str], webhook_url: Optional[str] = None, original_payload: Optional[dict] = None, job_id: Optional[str] = None) -> bool:
    """Send the GUI link (and basic metadata) to a Make.com webhook for Sheets/Excel automations.

    The webhook URL is taken from the explicit `webhook_url` argument, or from the
    MAKE_WEBHOOK_URL environment variable if not provided.
    """
    try:
        url = webhook_url or os.getenv("MAKE_WEBHOOK_URL")
        if not url:
            logger.info("No Make.com webhook URL provided; skipping Make webhook post")
            return False

        # Use entity_id for GUI link (with timestamp) to match S3 folder structure
        session_id = job_data.get("entity_id") or contact_id or call_id
        logger.info("Webhook - job_data entity_id: %s", job_data.get("entity_id"))
        logger.info("Webhook - session_id for GUI: %s", session_id)
        customer_slug = resolve_customer_slug_from_payload(original_payload or {})
        
        # Build both domain and IP GUI links
        gui_links = build_gui_links(session_id, customer_slug)
        gui_link = gui_links.get("gui_link")
        gui_link_ip = gui_links.get("gui_link_ip")
        
        if not gui_link and not gui_link_ip:
            logger.warning("Could not build GUI links for Make.com webhook; missing session_id or base URLs")
            return False

        # Start with original payload (what Make.com sent us) to echo data back
        payload: Dict[str, Any] = dict(original_payload or {})

        # Helper to pick first non-empty value
        def _coalesce(*values):
            for v in values:
                if isinstance(v, str):
                    if v.strip() != "":
                        return v
                elif v is not None:
                    return v
            return values[0] if values else None

        # Add/override our fields
        payload.update({
            "session_id": session_id,
            "entity_id": job_data.get("entity_id"),  # unique per call
            "contact_id": contact_id,
            "call_id": call_id,
            "gui_link": gui_link,  # Domain link (files.lead2424.com)
            "gui_link_ip": gui_link_ip,  # IP address link (16.52.197.95:8088)
            "created_at": int(time.time()),
            "job_id": job_id,
            "audio_length": job_data.get("audio_length"),  # formatted mm:ss string
        })

        # Coalesce key fields to ensure non-empty values
        try:
            cd = (original_payload or {}).get("customData") if isinstance((original_payload or {}).get("customData"), dict) else {}
        except Exception:
            cd = {}
        # location_id from multiple places
        loc_from_nested = None
        try:
            loc_obj = (original_payload or {}).get("location")
            if isinstance(loc_obj, dict):
                loc_from_nested = loc_obj.get("id")
        except Exception:
            pass
        try:
            cd_loc = cd.get("location") if isinstance(cd, dict) else None
            if isinstance(cd_loc, dict) and not loc_from_nested:
                loc_from_nested = cd_loc.get("id")
        except Exception:
            pass

        payload["location_id"] = _coalesce(
            payload.get("location_id"),
            (original_payload or {}).get("location_id"),
            loc_from_nested,
        )
        payload["company_name"] = _coalesce(
            payload.get("company_name"),
            (original_payload or {}).get("company_name"),
            cd.get("company_name") if isinstance(cd, dict) else None,
        )
        payload["date_time"] = _coalesce(
            payload.get("date_time"),
            (original_payload or {}).get("date_time"),
            cd.get("date_time") if isinstance(cd, dict) else None,
        )

        # Add contact_url if we have both location_id and contact_id
        try:
            loc_id = payload.get("location_id") or (original_payload or {}).get("location_id")
            # Fallbacks for nested forms
            if not loc_id:
                loc_nested = (original_payload or {}).get("location")
                if isinstance(loc_nested, dict):
                    loc_id = loc_nested.get("id") or loc_id
                custom_nested = (original_payload or {}).get("customData")
                if isinstance(custom_nested, dict):
                    loc_obj = custom_nested.get("location")
                    if isinstance(loc_obj, dict):
                        loc_id = loc_obj.get("id") or loc_id
            if loc_id and contact_id:
                payload["contact_url"] = f"https://app.lead2424.com/v2/location/{loc_id}/detail/{contact_id}"
        except Exception:
            pass

        # Normalize/echo Google Sheet identifier to help downstream mapping
        try:
            sheet_id = (original_payload or {}).get("googlesheet")
            logger.info("Google Sheet ID from original payload: %s", sheet_id)
            if sheet_id:
                payload["googlesheet"] = sheet_id
                payload["googlesheet_id"] = sheet_id  # convenience alias
                logger.info("Set googlesheet and googlesheet_id to: %s", sheet_id)
            else:
                logger.warning("No googlesheet ID found in original payload")
        except Exception as e:
            logger.warning("Error processing googlesheet ID: %s", e)

        # Include file URLs if available to help downstream automations
        file_fields = {
            "audio_url": job_data.get("audio_s3_url"),
            "transcript_url": job_data.get("transcript_s3_url"),
            "summary_url": job_data.get("summary_s3_url"),
            "prompt_url": job_data.get("prompt_s3_url"),
        }
        payload["files"] = {k: v for k, v in file_fields.items() if v}

        # Ensure expected fields exist in outbound payload even if missing from input
        expected_keys = [
            "audio",
            "location_id",
            "contact_id",
            "contact_first_name",
            "contact_last_name",
            "caller_name",
            "show_prompt",
            "token",
            "slug",
            "googlesheet",
            "company_name",
            "language",
            "date_time",
        ]
        for k in expected_keys:
            payload.setdefault(k, (original_payload or {}).get(k))

        logger.info("Posting GUI link to outbound webhook: %s", url)
        logger.info("Final webhook selected fields: %s", {
            "googlesheet": payload.get("googlesheet"),
            "googlesheet_id": payload.get("googlesheet_id"),
            "company_name": payload.get("company_name"),
            "contact_url": payload.get("contact_url"),
            "location_id": payload.get("location_id"),
            "contact_id": payload.get("contact_id"),
            "date_time": payload.get("date_time"),
            "audio_length": payload.get("audio_length"),
        })
        resp = requests.post(url, json=payload, timeout=20)
        if not resp.ok:
            logger.warning("Outbound webhook post failed: %s %s", resp.status_code, resp.text)
            return False
        logger.info("Outbound webhook post succeeded")
        return True
    except Exception as e:
        logger.warning("Failed to send Make.com webhook: %s", e)
        return False

def _public_s3_base_url() -> Optional[str]:
    # Prefer CDN base if configured
    try:
        cdn = (S3_CDN_BASE_URL or "").strip()
        if cdn:
            return cdn.rstrip('/')
    except Exception:
        pass
    if not S3_BUCKET:
        return None
    region = AWS_REGION or 'us-east-1'
    return f"https://{S3_BUCKET}.s3.amazonaws.com" if region == 'us-east-1' else f"https://{S3_BUCKET}.s3.{region}.amazonaws.com"

def build_public_s3_url(session_id: str, filename: str, customer_slug: Optional[str] = None) -> Optional[str]:
    base = _public_s3_base_url()
    if not base:
        return None
    if customer_slug:
        return f"{base}/audio/{customer_slug}/{session_id}/{filename}"
    return f"{base}/audio/{session_id}/{filename}"

def build_public_s3_url_from_key(key: str) -> Optional[str]:
    base = _public_s3_base_url()
    if not base:
        return None
    return f"{base}/{key}"

def find_latest_session_files(session_id: str, customer_slug: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Return dict mapping type->S3 key for latest files with timestamped names.
    Keys: summary, transcript, call_to_action, prompt. Values are full S3 keys or None.
    """
    result: Dict[str, Optional[str]] = {"summary": None, "transcript": None, "call_to_action": None, "prompt": None}
    try:
        if not S3_BUCKET:
            return result
        s3 = get_s3_client()
        
        # Try multiple path patterns to handle different naming conventions
        prefixes_to_try = []
        if customer_slug:
            # Try exact session_id match first (now includes timestamp for entity_id)
            prefixes_to_try.append(f"audio/{customer_slug}/{session_id}/")
            # Try session_id with timestamp pattern (legacy entity_id format)
            prefixes_to_try.append(f"audio/{customer_slug}/{session_id}_")
        prefixes_to_try.append(f"audio/{session_id}/")
        prefixes_to_try.append(f"audio/{session_id}_")
        
        logger.info(f"Searching S3 prefixes: {prefixes_to_try}")
        
        contents = []
        for prefix in prefixes_to_try:
            continuation = None
            while True:
                kwargs = {"Bucket": S3_BUCKET, "Prefix": prefix}
                if continuation:
                    kwargs["ContinuationToken"] = continuation
                resp = s3.list_objects_v2(**kwargs)
                contents.extend(resp.get("Contents", []))
                if resp.get("IsTruncated"):
                    continuation = resp.get("NextContinuationToken")
                else:
                    break
            if contents:  # If we found files in this prefix, use them
                break
        
        logger.info(f"Found {len(contents)} S3 objects")

        # Pick latest by suffix
        def pick_latest(suffix: str) -> Optional[str]:
            candidates = [obj for obj in contents if obj.get("Key", "").endswith(suffix)]
            if not candidates:
                return None
            latest = max(candidates, key=lambda o: o.get("LastModified"))
            return latest.get("Key")

        # Audio: pick latest by common audio extensions
        audio_candidates = [obj for obj in contents if obj.get("Key", "").lower().endswith((".wav", ".mp3", ".m4a", ".ogg"))]
        if audio_candidates:
            latest_audio = max(audio_candidates, key=lambda o: o.get("LastModified"))
            result["audio"] = latest_audio.get("Key")
        else:
            result["audio"] = None

        result["transcript"] = pick_latest("_diarized.json")
        result["summary"] = pick_latest("_summary.json")
        result["call_to_action"] = pick_latest("_call_to_action.json")
        result["prompt"] = pick_latest("_prompt.json")
    except Exception as e:
        logger.warning("Failed to list latest session files for %s: %s", session_id, e)
    return result

def fetch_json(url: str) -> Optional[dict]:
    try:
        resp = requests.get(url, timeout=20)
        if resp.ok:
            return resp.json()
    except Exception as e:
        logger.warning("Failed to fetch JSON from %s: %s", url, e)
    return None

@app.get("/view-session/{session_id}", response_class=HTMLResponse)
def view_session(request: Request, session_id: str):
    """Render a simple Jinja2 dashboard for a session/contact id.
    It attempts to load summary, transcript (diarized), call-to-action, and prompt files.
    """
    # We try to locate latest files by prefix. If exact names not known, we attempt common suffixes.
    # Expected naming: {session_id}_{timestamp}_summary.json, ..._call_to_action.json, ..._diarized.json
    # For now, we probe by trying to get the latest by hitting a small list of plausible names.
    # If you maintain exact filenames, consider storing them or indexing. Here we allow 404s gracefully.

    # Try to discover latest by probing a small time window (optional). Simpler: if CRM only passes session link,
    # we can attempt to fetch a listing via S3 ListObjectsV2, but bucket is public; List requires creds typically.
    # We'll instead try a heuristic: request a diarized listing via a known example name from job data isn't available here.

    # Heuristic: try up to N recent timestamps embedded in typical patterns is complex; keep simple: attempt to fetch
    # a manifest if exists, else try the 'most recent' patterns are unknown. We will instead attempt with wildcards not possible over HTTP.
    # Therefore, document expected names and rely on that.

    # Prefer latest timestamped files discovered via S3 listing
    latest_keys = find_latest_session_files(session_id)
    data = {"summary": None, "transcript": None, "call_to_action": None, "prompt": None}
    urls: Dict[str, Optional[str]] = {"summary": None, "transcript": None, "prompt": None, "audio": None}

    # For each type, if a key was found use exact URL; otherwise fall back to non-timestamp name
    fallbacks = {
        "summary": f"{session_id}_summary.json",
        "transcript": f"{session_id}_diarized.json",
        "prompt": f"{session_id}_prompt.json",
    }

    for key in ["summary", "transcript", "prompt"]:
        if latest_keys.get(key):
            url = build_public_s3_url_from_key(latest_keys[key])  # preserves timestamp
        else:
            url = build_public_s3_url(session_id, fallbacks[key])
        urls[key] = url
        if url:
            data[key] = fetch_json(url)

    # Audio url if found
    if latest_keys.get("audio"):
        urls["audio"] = build_public_s3_url_from_key(latest_keys["audio"]) 

    # Extract call to action data from summary file
    if data.get("summary") and isinstance(data["summary"], dict):
        call_to_action_data = (
            data["summary"].get("call_to_action")
            or data["summary"].get("call_to_action_items")
        )
        if call_to_action_data:
            data["call_to_action"] = {"call_to_action": call_to_action_data}

    # Compute human-readable UTC for prompt created_at if available
    if data.get("prompt") and isinstance(data.get("prompt"), dict):
        try:
            created_raw = data["prompt"].get("created_at")
            if isinstance(created_raw, (int, float)):
                dt = datetime.fromtimestamp(float(created_raw), tz=timezone.utc)
                data["prompt"]["created_at_utc"] = dt.strftime("%Y-%m-%d %H:%M UTC")
        except Exception as _e:
            pass

    return templates.TemplateResponse(
        "session_view.html",
        {
            "request": request,
            "session_id": session_id,
            "summary": data["summary"],
            "transcript": data["transcript"],
            "call_to_action": data["call_to_action"],
            "prompt": data["prompt"],
            "urls": urls,
        },
    )


# Support tenant-prefixed GUI links like /view-session/{customer_slug}/{session_id}
@app.get("/view-session/{customer_slug}/{session_id}", response_class=HTMLResponse)
def view_session_with_slug(request: Request, customer_slug: str, session_id: str):
    """Render session view with customer slug context for S3 file discovery."""
    logger.info(f"Slug-aware route: customer_slug='{customer_slug}', session_id='{session_id}'")
    # Prefer latest timestamped files discovered via S3 listing with slug context
    latest_keys = find_latest_session_files(session_id, customer_slug)
    logger.info(f"Found latest keys: {latest_keys}")
    data = {"summary": None, "transcript": None, "call_to_action": None, "prompt": None}
    urls: Dict[str, Optional[str]] = {"summary": None, "transcript": None, "prompt": None, "audio": None}

    # For each type, if a key was found use exact URL; otherwise fall back to non-timestamp name
    fallbacks = {
        "summary": f"{session_id}_summary.json",
        "transcript": f"{session_id}_diarized.json",
        "prompt": f"{session_id}_prompt.json",
    }

    for key in ["summary", "transcript", "prompt"]:
        if latest_keys.get(key):
            url = build_public_s3_url_from_key(latest_keys[key])  # preserves timestamp
        else:
            # Build URL with slug context for fallback files
            if customer_slug:
                url = build_public_s3_url(session_id, fallbacks[key], customer_slug)
            else:
                url = build_public_s3_url(session_id, fallbacks[key])
        urls[key] = url
        if url:
            data[key] = fetch_json(url)

    # Audio url if found
    if latest_keys.get("audio"):
        urls["audio"] = build_public_s3_url_from_key(latest_keys["audio"]) 

    # Extract call to action data from summary file
    if data.get("summary") and isinstance(data["summary"], dict):
        call_to_action_data = (
            data["summary"].get("call_to_action")
            or data["summary"].get("call_to_action_items")
        )
        if call_to_action_data:
            data["call_to_action"] = {"call_to_action": call_to_action_data}

    # Compute human-readable UTC for prompt created_at if available
    if data.get("prompt") and isinstance(data.get("prompt"), dict):
        try:
            created_raw = data["prompt"].get("created_at")
            if isinstance(created_raw, (int, float)):
                dt = datetime.fromtimestamp(float(created_raw), tz=timezone.utc)
                data["prompt"]["created_at_utc"] = dt.strftime("%Y-%m-%d %H:%M UTC")
        except Exception as _e:
            pass

    return templates.TemplateResponse(
        "session_view.html",
        {
            "request": request,
            "session_id": session_id,
            "summary": data["summary"],
            "transcript": data["transcript"],
            "call_to_action": data["call_to_action"],
            "prompt": data["prompt"],
            "urls": urls,
        },
    )


def send_summary_note(contact_id: str, job_data: dict, access_token: str) -> bool:
    """Send second note with summary."""
    summary_s3_url = job_data.get("summary_s3_url")
    if not summary_s3_url:
        logger.warning("No summary file available to send")
        return False
    
    try:
        # Download and read summary from S3
        summary_resp = requests.get(summary_s3_url, timeout=30)
        if not summary_resp.ok:
            logger.warning("Failed to download summary from S3: %s", summary_resp.status_code)
            return False
        
        summary_data = summary_resp.json()
        summary_text = summary_data.get("summary", "No summary available")
        
        note_body = f"📋 **Call Summary**\n\n{summary_text}"
        return send_crm_note(contact_id, note_body, access_token, "summary")
        
    except Exception as e:
        logger.warning("Failed to process summary for CRM note: %s", e)
        return False


def send_call_to_action_note(contact_id: str, job_data: dict, access_token: str) -> bool:
    """Send third note with call to action items."""
    call_to_action_s3_url = job_data.get("call_to_action_s3_url")
    if not call_to_action_s3_url:
        logger.warning("No call to action file available to send")
        return False
    
    try:
        # Download and read call to action from S3
        cta_resp = requests.get(call_to_action_s3_url, timeout=30)
        if not cta_resp.ok:
            logger.warning("Failed to download call to action from S3: %s", cta_resp.status_code)
            return False
        
        cta_data = cta_resp.json()
        call_to_action_items = cta_data.get("call_to_action", [])
        
        if not call_to_action_items:
            logger.warning("No call to action items found in file")
            return False
        
        # Format call to action items
        cta_lines = ["✅ **Call to Action Items**\n"]
        for i, item in enumerate(call_to_action_items, 1):
            action = item.get("item", "No action specified")
            owner = item.get("owner", "Unassigned")
            due = item.get("due", "No due date")
            
            cta_lines.append(f"{i}. **{action}**")
            cta_lines.append(f"   👤 Owner: {owner}")
            if due and due != "":
                cta_lines.append(f"   📅 Due: {due}")
            cta_lines.append("")  # Empty line for spacing
        
        note_body = "\n".join(cta_lines)
        return send_crm_note(contact_id, note_body, access_token, "call_to_action")
        
    except Exception as e:
        logger.warning("Failed to process call to action for CRM note: %s", e)
        return False


def create_prompt_file(prompt_path: Path, caller_name: str = "", contact_first_name: str = "", contact_last_name: str = ""):
    """Create a prompt.json file with the current prompt template and caller information."""
    try:
        # Get the default prompt from ai_summary
        if ai_summary and hasattr(ai_summary, "get_default_prompt"):
            prompt_template = ai_summary.get_default_prompt()
        else:
            # Fallback prompt if ai_summary is not available
            prompt_template = """You are an expert meeting and call analysis assistant.
Your job is to read the transcript of a sales or support call and return ONLY valid JSON.

The JSON must follow this schema exactly:
{{
  "summary": "string – a clear, concise summary of the call (cover goals, decisions, commitments, numbers, and outcomes).",
  "call_to_action_items": [
    {{
      "item": "string – the specific task or next step",
      "owner": "string – person responsible (use speaker names if clear, else leave empty)",
      "due": "string – due date if mentioned, else empty"
    }}
  ],
  "call_quality_feedback": {{
    "strengths": ["list of things done well – rapport, clarity, active listening, etc."],
    "improvements": ["list of ways caller/agent can improve – tone, pacing, missing info, objection handling, etc."]
  }}
}}

Guidelines:
- Be **brief but insightful**: a short call = short summary; a long/complex call = more detail.
- Summaries should highlight outcomes, decisions, and commitments – not just a play-by-play.
- Call-to-actions must be **specific and actionable**. Avoid vague items like "follow up" unless no detail is provided.
- Owners: infer from speaker labels where possible. Example: if "Agent" says "I will send the proposal", set owner = "Agent".
- If dates are mentioned, capture them (e.g. "by Friday"); if not, leave due = "".
- Feedback should be constructive: balance what went well with what could improve.

TRANSCRIPT:
<<<
{{full_transcript}}
>>>"""

        # Create prompt data with caller and contact information
        prompt_data = {
            "prompt": prompt_template,
            "caller_information": {
                "full_name": caller_name.strip()
            },
            "contact_information": {
                "first_name": contact_first_name,
                "last_name": contact_last_name,
                "full_name": f"{contact_first_name} {contact_last_name}".strip()
            },
            "created_at": time.time(),
            "version": "1.0"
        }
        
        # Write to file
        with open(prompt_path, 'w', encoding='utf-8') as f:
            json.dump(prompt_data, f, indent=4, ensure_ascii=False)
        
        logger.info("Created prompt file: %s", prompt_path)
        return True
        
    except Exception as e:
        logger.exception("Failed to create prompt file: %s", e)
        return False


def send_crm_notes(contact_id: str, job_data: dict, access_token: str):
    """Send only the GUI link note to CRM."""
    logger.info("Sending GUI link note to CRM for contact_id=%s", contact_id)

    # Track success of file link note only
    notes_sent = {
        "file_links": False
    }

    # Send only file links note
    notes_sent["file_links"] = send_file_links_note(contact_id, job_data, access_token)

    # Log results
    if notes_sent["file_links"]:
        logger.info("Successfully sent notes to CRM: file_links")
    else:
        logger.warning("Failed to send notes to CRM: file_links")

    # Store results in job data
    job_data["crm_notes_sent"] = notes_sent


def summarize_transcript_with_openai(transcription: dict) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    # Build a readable transcript (speaker + text + time)
    lines = []
    for seg in transcription.get("segments", []):
        start = seg.get("start")
        speaker = seg.get("speaker", "Unknown")
        text = seg.get("text", "").strip()
        lines.append(f"[{start:.2f}] {speaker}: {text}")

    # join; if extremely long, we could chunk - for simplicity we send everything (you may want to chunk in production)
    joined = "\n".join(lines)

    system_prompt = (
        "You are an assistant that creates concise call summaries. Given the transcript below, output:\n"
        "1) A short TL;DR (1-2 lines)\n2) Key points / decisions (bullet list)\n3) Action items with owners (if mentioned)\n4) Important quotes or timestamps\n        "
    )

    user_prompt = "Transcript:\n\n" + joined

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    logger.info("Sending transcript to OpenAI for summarization (model=%s)", OPENAI_MODEL)
    resp = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=1500,
    )
    summary_text = resp["choices"][0]["message"]["content"].strip()
    return summary_text


def process_job(job_id: str, payload: dict):
    """Simplified pipeline: download audio, run CLI transcriber, upload to S3."""
    logger.info("Job %s started (simple mode)", job_id)
    jobs[job_id] = {"status": "started", "created_at": time.time(), "payload": payload}

    local_path = None
    run_output_dir = None
    transcript_path = None

    try:
        # Debug: Log the payload structure
        logger.info("Job %s payload keys: %s", job_id, list(payload.keys()) if isinstance(payload, dict) else "Not a dict")
        logger.info("Job %s payload type: %s", job_id, type(payload))
        logger.info("Job %s payload audio value: %s", job_id, payload.get("audio") if isinstance(payload, dict) else "N/A")
        
        audio_url = payload.get("audio")
        if not audio_url:
            raise ValueError("payload must include 'audio' url")
        # Prefer contact_id over call_id; maintain call_id for backward compatibility
        contact_id = payload.get("contact_id")
        call_id = payload.get("call_id")
        # Create unique session per call: contact_id + call_id + timestamp
        # This ensures each call gets its own files even for the same contact
        timestamp = int(time.time())
        if contact_id and call_id:
            entity_id = f"{contact_id}_{call_id}_{timestamp}"
        elif contact_id:
            entity_id = f"{contact_id}_{timestamp}"
        elif call_id:
            entity_id = f"{call_id}_{timestamp}"
        else:
            entity_id = f"session_{timestamp}_{uuid.uuid4().hex[:8]}"

        # Store entity_id early so it's available for webhooks
        jobs[job_id]["entity_id"] = entity_id
        jobs[job_id]["contact_id"] = contact_id
        jobs[job_id]["call_id"] = call_id

        jobs[job_id]["status"] = "downloading"
        # Resolve and store customer slug early for downstream uses
        try:
            jobs[job_id]["customer_slug"] = resolve_customer_slug_from_payload(payload)  # optional
        except Exception:
            pass
        # build a readable persisted audio name and a temp download target
        # Add timestamp to ensure unique file names
        readable_name = derive_readable_audio_name(audio_url, f"{entity_id}_{uuid.uuid4().hex[:8]}.wav")
        temp_download_name = f"{entity_id}_{uuid.uuid4().hex[:8]}"
        local_path = AUDIO_TMP_DIR / temp_download_name
        # support basic auth via audio_auth or headers
        audio_auth = None
        if isinstance(payload.get("audio_auth"), dict):
            aa = payload.get("audio_auth")
            audio_auth = (aa.get("username"), aa.get("password"))
        headers = payload.get("audio_headers")
        local_path = download_audio(audio_url, local_path, auth=audio_auth, headers=headers)
        # Probe audio duration
        try:
            audio_length = probe_audio_duration_seconds(local_path)
            if audio_length is not None:
                formatted_length = format_seconds_mmss(audio_length)
                jobs[job_id]["audio_length"] = formatted_length
                logger.info("Probed audio duration (mm:ss): %s", formatted_length)
        except Exception as dur_ex:
            logger.warning("Failed to probe audio duration: %s", dur_ex)

        # Run your CLI transcriber script
        jobs[job_id]["status"] = "transcribing"
        # Create temporary local output directory for processing
        run_dir_name = Path(readable_name).stem
        run_output_dir = OUTPUT_DIR / run_dir_name
        run_output_dir.mkdir(parents=True, exist_ok=True)

        transcript_filename = f"{entity_id}_diarized.json"
        transcript_path = run_output_dir / transcript_filename
        script_path = (Path(__file__).parent / "audio_transcribe_diarization.py").resolve()
        # Determine CLI model/device/threads from environment; normalize model names to CLI choices
        model_env = os.getenv("WHISPER_MODEL", MODEL_SIZE)
        cli_model = (model_env or "small").lower()
        if cli_model.startswith("large"):
            cli_model = "large"
        elif cli_model not in {"tiny", "base", "small", "medium", "large"}:
            cli_model = "small"

        cli_threads = str(NUM_THREADS)
        cli_device = DEVICE  # CLI currently supports only "cpu"

        cmd = [
            sys.executable,
            str(script_path),
            str(local_path),
            "-o", str(transcript_path),
            "-l", payload.get("language", "en"),
            "-m", cli_model,
            "-t", cli_threads,
            "-d", cli_device,
        ]
        if payload.get("no_diarization"):
            cmd.append("--no-diarization")
        logger.info("Running transcriber: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(Path(__file__).parent))

        # Summarize using ai_summary.process_transcript if available
        summary_path = None
        call_to_action_path = None
        prompt_path = None
        try:
            logger.info("Checking if summarization is available...")
            if ai_summary is not None and hasattr(ai_summary, "process_transcript"):
                logger.info("Starting summarization step...")
                jobs[job_id]["status"] = "summarizing"
                # Use a proper base output file so ai_summary writes:
                # <base>.json, <base>_summary.json, <base>_call_to_action.json
                base_output_filename = f"{entity_id}.json"
                base_output_path = run_output_dir / base_output_filename
                
                # Get caller and contact information from payload
                caller_name = payload.get("caller_name") or ""
                # For ai_summary.process_transcript we still split into first/last locally
                split_first = ""
                split_last = ""
                try:
                    parts = [p for p in caller_name.strip().split(" ") if p]
                    if len(parts) == 1:
                        split_first = parts[0]
                    elif len(parts) >= 2:
                        split_first = parts[0]
                        split_last = " ".join(parts[1:])
                except Exception:
                    pass
                contact_first_name = payload.get("contact_first_name") or ""
                contact_last_name = payload.get("contact_last_name") or ""
                
                # Process transcript with caller information
                logger.info("Calling ai_summary.process_transcript...")
                ai_summary.process_transcript(
                    str(transcript_path), 
                    str(base_output_path), 
                    OPENAI_API_KEY,
                    "prompt.json",  # prompt file path
                    split_first,
                    split_last
                )
                logger.info("ai_summary.process_transcript completed")
                
                # Expected generated files
                summary_path = run_output_dir / f"{entity_id}_summary.json"
                logger.info("Summary file expected at: %s", summary_path)
                
                # Call to action data is now included in summary file
                call_to_action_path = None
                    
                # Create prompt.json file if show_prompt is true
                if payload.get("show_prompt", False):
                    prompt_filename = f"{entity_id}_prompt.json"
                    prompt_path = run_output_dir / prompt_filename
                    create_prompt_file(prompt_path, caller_name, contact_first_name, contact_last_name)
                    
            else:
                logger.warning("ai_summary.process_transcript not available; skipping summarization")
            logger.info("Summarization step completed (or skipped)")
        except Exception as summarize_ex:
            logger.exception("Summarization step failed: %s", summarize_ex)

        # Inject caller/contact metadata into summary JSON (without changing summary text)
        try:
            if summary_path and summary_path.exists():
                with open(summary_path, "r", encoding="utf-8") as sf:
                    summary_json = json.load(sf)
                # Prepare names again (safe if already set above; handle path where ai_summary block skipped)
                caller_name = payload.get("caller_name") or ""
                contact_first_name = payload.get("contact_first_name") or ""
                contact_last_name = payload.get("contact_last_name") or ""

                summary_json.setdefault("metadata", {})
                summary_json["metadata"].update({
                    "caller": {
                        "full_name": caller_name.strip()
                    },
                    "contact": {
                        "first_name": contact_first_name,
                        "last_name": contact_last_name,
                        "full_name": f"{contact_first_name} {contact_last_name}".strip()
                    }
                })
                with open(summary_path, "w", encoding="utf-8") as sf:
                    json.dump(summary_json, sf, indent=2, ensure_ascii=False)
        except Exception as meta_ex:
            logger.warning("Failed to inject metadata into summary file: %s", meta_ex)

        # Upload files to S3
        logger.info("Starting S3 upload step...")
        jobs[job_id]["status"] = "uploading_to_s3"
        # Use customer slug in S3 path if available: audio/{slug}/{contact_id}/
        customer_slug = jobs[job_id].get("customer_slug")
        if customer_slug:
            s3_base_key = f"audio/{customer_slug}/{entity_id}/"
        else:
            s3_base_key = f"audio/{entity_id}/"
        
        # Upload audio file
        audio_s3_key = f"{s3_base_key}{readable_name}"
        audio_s3_url = upload_to_s3(local_path, audio_s3_key)
        jobs[job_id]["audio_s3_url"] = audio_s3_url
        jobs[job_id]["audio_s3_key"] = audio_s3_key

        # Upload transcription file
        transcript_s3_key = f"{s3_base_key}{transcript_filename}"
        transcript_s3_url = upload_to_s3(transcript_path, transcript_s3_key)
        jobs[job_id]["transcript_s3_url"] = transcript_s3_url
        jobs[job_id]["transcript_s3_key"] = transcript_s3_key

        # Upload summary file if it exists
        if summary_path and summary_path.exists():
            summary_s3_key = f"{s3_base_key}{summary_path.name}"
            summary_s3_url = upload_to_s3(summary_path, summary_s3_key)
            jobs[job_id]["summary_s3_url"] = summary_s3_url
            jobs[job_id]["summary_s3_key"] = summary_s3_key

        # Call to action data is now included in summary file, no separate upload needed

        # Upload prompt file if it exists (when show_prompt=true)
        if prompt_path and prompt_path.exists():
            prompt_s3_key = f"{s3_base_key}{prompt_path.name}"
            prompt_s3_url = upload_to_s3(prompt_path, prompt_s3_key)
            jobs[job_id]["prompt_s3_url"] = prompt_s3_url
            jobs[job_id]["prompt_s3_key"] = prompt_s3_key

        # Send GUI link to Make.com webhook if configured
        try:
            # Collect all outbound webhooks: env default, make.com, client-specific, and extras
            webhook_urls = []
            # Prefer per-request custom outbound URL over env default
            primary_out_url = payload.get("make_url_out")
            env_url = os.getenv("MAKE_WEBHOOK_URL")
            if primary_out_url:
                webhook_urls.append(primary_out_url)
            elif env_url:
                webhook_urls.append(env_url)
            if payload.get("make_webhook_url"):
                webhook_urls.append(payload.get("make_webhook_url"))
            if payload.get("client_webhook_url"):
                webhook_urls.append(payload.get("client_webhook_url"))
            if isinstance(payload.get("extra_webhook_urls"), list):
                webhook_urls.extend([u for u in payload.get("extra_webhook_urls") if isinstance(u, str) and u])

            # De-duplicate while preserving order
            seen = set()
            unique_urls = []
            for u in webhook_urls:
                if u not in seen:
                    seen.add(u)
                    unique_urls.append(u)

            for url in unique_urls:
                logger.info("Sending webhook to URL: %s", url)
                logger.info("Job data entity_id: %s", jobs[job_id].get("entity_id"))
                send_make_webhook(jobs[job_id], contact_id, call_id, url, original_payload=payload, job_id=job_id)
        except Exception as make_ex:
            logger.warning("Unexpected error while posting to Make.com webhook: %s", make_ex)

        # Send GUI link note to CRM using contact_id and token from payload (if provided)
        try:
            access_token = payload.get("token")
            if contact_id and access_token:
                jobs[job_id]["status"] = "posting_crm_notes"
                
                # Send only file links note
                send_crm_notes(contact_id, jobs[job_id], access_token)
                
            elif contact_id and not access_token:
                logger.warning("No access token provided in payload; skipping CRM note post")
            else:
                logger.info("No contact_id provided; skipping CRM note post")
        except Exception as post_ex:
            logger.warning("Unexpected error while posting CRM notes: %s", post_ex)

        jobs[job_id]["status"] = "done"
        jobs[job_id]["finished_at"] = time.time()
        jobs[job_id]["s3_base_key"] = s3_base_key

        logger.info("Job %s finished successfully (simple mode)", job_id)

    except Exception as e:
        logger.exception("Job %s failed: %s", job_id, e)
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
    finally:
        # Clean up local temporary files and working directory regardless of outcome
        try:
            if local_path and isinstance(local_path, Path) and local_path.exists():
                local_path.unlink()
            if run_output_dir and isinstance(run_output_dir, Path) and run_output_dir.exists():
                shutil.rmtree(run_output_dir)
            logger.info("Cleaned up local temporary files for job %s", job_id)
        except Exception as cleanup_ex:
            logger.warning("Failed to clean up local files for job %s: %s", job_id, cleanup_ex)


@app.post("/webhook")
def webhook_listener(payload: dict):
    """Receives webhook (from Make or direct CRM). Payload must contain `audio` url. Optionally call_id.

    Example payload:
    {
      "audio": "https://api.twilio.com/.../Recordings/RExxx",
      "contact_id": "abc123",
      "make_webhook_url": "https://hook.integromat.com/xxxxx",  # optional; else uses MAKE_WEBHOOK_URL env
      "customer_slug": "brix",                                   # optional; results in /brix/view-session/{id}
      "customer_id": "cust_001",                                 # optional; resolved via CUSTOMER_SLUG_MAP
      "caller_first_name": "John",
      "caller_last_name": "Doe",
      "show_prompt": true,
      "language": "en",
      "call_id": "12345"  # optional/backward compatibility
    }
    
    Returns:
    {
      "job_id": "uuid",
      "status_url": "/jobs/{job_id}",
      "contact_id": "abc123",
      "call_id": "12345",
      "status": "queued",
      "queue_position": 1
    }
    """
    logger.info("Inbound payload keys: %s", list(payload.keys()))
    # Log selected raw input fields for diagnostics
    try:
        raw_log = {k: payload.get(k) for k in EXPECTED_INPUT_KEYS}
        logger.info("Inbound selected fields (pre-normalization): %s", raw_log)
        # Nested indicators
        try:
            nested_loc = payload.get("location", {}).get("id") if isinstance(payload.get("location"), dict) else None
        except Exception:
            nested_loc = None
        try:
            cd = payload.get("customData", {}) if isinstance(payload.get("customData"), dict) else {}
        except Exception:
            cd = {}
        logger.info("Inbound nested indicators: location.id=%s, customData.keys=%s", nested_loc, list(cd.keys()) if isinstance(cd, dict) else None)
    except Exception:
        pass
    job_id = str(uuid.uuid4())

    # Normalize/augment payload with fields from nested customData if present
    try:
        custom = payload.get("customData") if isinstance(payload, dict) else None
        if isinstance(custom, dict):
            # Map common customData fields to top-level if missing
            payload.setdefault("audio", custom.get("audio"))
            payload.setdefault("token", custom.get("token"))
            payload.setdefault("slug", custom.get("slug"))
            payload.setdefault("googlesheet", custom.get("googlesheet"))
            payload.setdefault("company_name", custom.get("company_name"))
            payload.setdefault("date_time", custom.get("date_time"))
            payload.setdefault("make_url_out", custom.get("make_url_out"))
            # Location id can be nested under customData.location.id
            try:
                loc = custom.get("location") or {}
                loc_id = loc.get("id")
                if loc_id and not payload.get("location_id"):
                    payload["location_id"] = loc_id
            except Exception:
                pass
            # Caller name nested path: customData.phoneCall.user.name
            try:
                phone_call = custom.get("phoneCall") or {}
                user_obj = phone_call.get("user") or {}
                caller_name = user_obj.get("name")
                if caller_name and not payload.get("caller_name"):
                    payload["caller_name"] = caller_name
            except Exception:
                pass
        # Also support top-level nested location.id
        try:
            top_loc = payload.get("location") if isinstance(payload, dict) else None
            if isinstance(top_loc, dict) and top_loc.get("id") and not payload.get("location_id"):
                payload["location_id"] = top_loc.get("id")
        except Exception:
            pass
        # Default show_prompt to true if not provided
        if "show_prompt" not in payload:
            payload["show_prompt"] = True
        # Default language to en if missing
        if "language" not in payload:
            payload["language"] = "en"
    except Exception:
        # Non-fatal; continue with whatever was provided
        pass
    contact_id = payload.get("contact_id")
    call_id = payload.get("call_id")
    # Create unique session per call: contact_id + call_id + timestamp
    # This ensures each call gets its own files even for the same contact
    timestamp = int(time.time())
    if contact_id and call_id:
        entity_id = f"{contact_id}_{call_id}_{timestamp}"
    elif contact_id:
        entity_id = f"{contact_id}_{timestamp}"
    elif call_id:
        entity_id = f"{call_id}_{timestamp}"
    else:
        entity_id = f"session_{timestamp}_{uuid.uuid4().hex[:8]}"
    
    # Initialize job status
    jobs[job_id] = {
        "status": "queued", 
        "created_at": time.time(), 
        "entity_id": entity_id,  # unique per call
        "contact_id": contact_id,
        "call_id": call_id,
        "queue_position": 0
    }
    
    # Add to queue
    add_job_to_queue(job_id, payload)
    
    # Get current queue position
    current_position = jobs[job_id].get("queue_position", 0)
    
    # Log normalized input snapshot that will drive processing
    try:
        normalized_log = {k: payload.get(k) for k in EXPECTED_INPUT_KEYS}
        logger.info("Inbound selected fields (normalized): %s", normalized_log)
    except Exception:
        pass

    return {
        "job_id": job_id, 
        "status_url": f"/jobs/{job_id}",
        "entity_id": entity_id,  # unique per call
        "contact_id": contact_id,
        "call_id": call_id,
        "status": "queued",
        "queue_position": current_position
    }


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    """Get job status and results. When status is 'done', includes all S3 links.
    
    Response format when completed:
    {
      "job_id": "uuid",
      "call_id": "12345",
      "status": "done",
      "created_at": 1234567890,
      "finished_at": 1234567891,
      "s3_base_key": "audio/12345/",
      "files": {
        "audio": {
          "url": "https://s3.amazonaws.com/bucket/audio/12345/file.wav",
          "key": "audio/12345/file.wav",
          "filename": "12345_1234567890_abc12345.wav"
        },
        "transcription": {
          "url": "https://s3.amazonaws.com/bucket/audio/12345/transcript.json",
          "key": "audio/12345/12345_1234567890_diarized.json",
          "filename": "12345_1234567890_diarized.json"
        },
        "summary": {
          "url": "https://s3.amazonaws.com/bucket/audio/12345/summary.json",
          "key": "audio/12345/12345_1234567890_summary.json",
          "filename": "12345_1234567890_summary.json"
        },
        "call_to_action": {
          "url": "https://s3.amazonaws.com/bucket/audio/12345/call_to_action.json",
          "key": "audio/12345/12345_1234567890_call_to_action.json",
          "filename": "12345_1234567890_call_to_action.json"
        }
      }
    }
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")
    
    # If job is completed, structure the response with all S3 links
    if job.get("status") == "done":
        response = {
            "job_id": job_id,
            "contact_id": job.get("contact_id"),
            "call_id": job.get("call_id"),
            "status": job.get("status"),
            "created_at": job.get("created_at"),
            "finished_at": job.get("finished_at"),
            "s3_base_key": job.get("s3_base_key"),
            "files": {}
        }
        
        # Add audio file info
        if job.get("audio_s3_url"):
            response["files"]["audio"] = {
                "url": job.get("audio_s3_url"),
                "key": job.get("audio_s3_key"),
                "filename": job.get("audio_s3_key", "").split("/")[-1] if job.get("audio_s3_key") else None
            }
        
        # Add transcription file info
        if job.get("transcript_s3_url"):
            response["files"]["transcription"] = {
                "url": job.get("transcript_s3_url"),
                "key": job.get("transcript_s3_key"),
                "filename": job.get("transcript_s3_key", "").split("/")[-1] if job.get("transcript_s3_key") else None
            }
        
        # Add summary file info (if exists)
        if job.get("summary_s3_url"):
            response["files"]["summary"] = {
                "url": job.get("summary_s3_url"),
                "key": job.get("summary_s3_key"),
                "filename": job.get("summary_s3_key", "").split("/")[-1] if job.get("summary_s3_key") else None
            }
        
        # Add call-to-action file info (if exists)
        if job.get("call_to_action_s3_url"):
            response["files"]["call_to_action"] = {
                "url": job.get("call_to_action_s3_url"),
                "key": job.get("call_to_action_s3_key"),
                "filename": job.get("call_to_action_s3_key", "").split("/")[-1] if job.get("call_to_action_s3_key") else None
            }
        
        # Add prompt file info (if exists)
        if job.get("prompt_s3_url"):
            response["files"]["prompt"] = {
                "url": job.get("prompt_s3_url"),
                "key": job.get("prompt_s3_key"),
                "filename": job.get("prompt_s3_key", "").split("/")[-1] if job.get("prompt_s3_key") else None
            }
        
        return response
    
    # For non-completed jobs, return the basic job info with queue status
    response = dict(job)
    
    # Add queue information
    with queue_lock:
        response["is_processing"] = is_processing
        response["queue_length"] = len(job_queue)
        if "queue_position" in job:
            response["queue_position"] = job["queue_position"]
    
    return response


@app.get("/queue/status")
def get_queue_status():
    """Get current queue status and statistics."""
    with queue_lock:
        return {
            "is_processing": is_processing,
            "queue_length": len(job_queue),
            "total_jobs": len(jobs),
            "completed_jobs": len([j for j in jobs.values() if j.get("status") == "done"]),
            "failed_jobs": len([j for j in jobs.values() if j.get("status") == "error"]),
            "queued_jobs": len([j for j in jobs.values() if j.get("status") == "queued"])
        }


@app.get("/health")
def health():
    return {"status": "ok", "models_preloaded": PRELOAD_MODELS}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_whisper_service:app", host="0.0.0.0", port=int(os.getenv("PORT", 8088)), log_level="info")
