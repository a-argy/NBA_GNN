"""
GCS upload/download helpers for the SwishNet pipeline.

Uses google-cloud-storage SDK with gsutil subprocess fallback.
"""

import os
import subprocess
from pathlib import Path

DEFAULT_BUCKET = "swishnet-nba"

# Try to import the GCS SDK; fall back to gsutil subprocess if unavailable.
try:
    from google.cloud import storage as _gcs_storage
    _GCS_SDK_AVAILABLE = True
except ImportError:
    _GCS_SDK_AVAILABLE = False


def _get_client():
    if not _GCS_SDK_AVAILABLE:
        raise RuntimeError(
            "google-cloud-storage is not installed. "
            "Run: pip install google-cloud-storage"
        )
    return _gcs_storage.Client()


def upload_file(local_path: str, gcs_path: str, bucket: str = DEFAULT_BUCKET) -> None:
    """Upload a single local file to GCS.

    Args:
        local_path: Absolute or relative path to the local file.
        gcs_path:   Destination path within the bucket (no gs:// prefix).
        bucket:     GCS bucket name.
    """
    local_path = str(local_path)
    if _GCS_SDK_AVAILABLE:
        client = _get_client()
        blob = client.bucket(bucket).blob(gcs_path)
        blob.upload_from_filename(local_path)
    else:
        _gsutil(["cp", local_path, f"gs://{bucket}/{gcs_path}"])


def download_file(gcs_path: str, local_path: str, bucket: str = DEFAULT_BUCKET) -> None:
    """Download a single file from GCS to a local path.

    Args:
        gcs_path:   Source path within the bucket (no gs:// prefix).
        local_path: Destination local path (parent dirs are created if needed).
        bucket:     GCS bucket name.
    """
    local_path = str(local_path)
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    if _GCS_SDK_AVAILABLE:
        client = _get_client()
        blob = client.bucket(bucket).blob(gcs_path)
        blob.download_to_filename(local_path)
    else:
        _gsutil(["cp", f"gs://{bucket}/{gcs_path}", local_path])


def sync_dir(local_dir: str, gcs_prefix: str, bucket: str = DEFAULT_BUCKET) -> None:
    """Recursively upload a local directory to a GCS prefix.

    Equivalent to: gsutil -m rsync -r local_dir gs://bucket/gcs_prefix

    Args:
        local_dir:  Local directory to upload.
        gcs_prefix: Destination prefix within the bucket (no gs:// prefix).
        bucket:     GCS bucket name.
    """
    local_dir = str(local_dir).rstrip("/")
    gcs_prefix = gcs_prefix.rstrip("/")
    gcs_uri = f"gs://{bucket}/{gcs_prefix}"

    if _GCS_SDK_AVAILABLE:
        client = _get_client()
        bkt = client.bucket(bucket)
        for root, _dirs, files in os.walk(local_dir):
            for fname in files:
                local_file = os.path.join(root, fname)
                rel = os.path.relpath(local_file, local_dir)
                blob_name = f"{gcs_prefix}/{rel}".replace("\\", "/")
                bkt.blob(blob_name).upload_from_filename(local_file)
    else:
        _gsutil(["-m", "rsync", "-r", local_dir, gcs_uri])


def file_exists_on_gcs(gcs_path: str, bucket: str = DEFAULT_BUCKET) -> bool:
    """Return True if the given path exists in the GCS bucket.

    Used by the pipeline to skip already-processed games (resumable).

    Args:
        gcs_path: Path within the bucket (no gs:// prefix).
        bucket:   GCS bucket name.
    """
    if _GCS_SDK_AVAILABLE:
        client = _get_client()
        blob = client.bucket(bucket).blob(gcs_path)
        return blob.exists()
    else:
        result = subprocess.run(
            ["gsutil", "-q", "stat", f"gs://{bucket}/{gcs_path}"],
            capture_output=True,
        )
        return result.returncode == 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gsutil(args: list) -> None:
    """Run a gsutil command, raising RuntimeError on failure."""
    cmd = ["gsutil"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"gsutil command failed: {' '.join(cmd)}\n"
            f"stderr: {result.stderr}"
        )
