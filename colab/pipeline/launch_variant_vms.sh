#!/bin/bash
# Launch 3 parallel GCP CPU VMs to rebuild graphs for temporal variants A, B, C.
#
# Prerequisites:
#   - gcloud authenticated and project set (swishnet-489108)
#   - Shot data already in GCS (processed/shots_data/) from original pipeline
#   - Run from repo root: bash colab/pipeline/launch_variant_vms.sh
#
# Output in GCS:
#   gs://swishnet-nba/processed/variants/temporal_dense_short/   (A)
#   gs://swishnet-nba/processed/variants/temporal_sparse_long/   (B)
#   gs://swishnet-nba/processed/variants/temporal_dense_long/    (C)
#
# Monitor a VM:
#   gcloud compute ssh swishnet-variant-a --project=swishnet-489108 --zone=us-central1-a \
#     -- sudo tail -f /var/log/swishnet-startup.log
#
# Check VM status:
#   gcloud compute instances list --project=swishnet-489108 --filter="name~swishnet-variant"

set -euo pipefail

PROJECT=swishnet-489108
ZONE=us-central1-a
BUCKET=swishnet-nba
MACHINE_TYPE=n1-standard-4
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# ---------------------------------------------------------------------------
# Step 1 — Upload only the Python source files the VMs need (not data/results)
# ---------------------------------------------------------------------------
echo "Uploading pipeline source files to gs://$BUCKET/code/ ..."

NEEDED_FILES=(
  colab/load_with_context.py
  colab/build_graph.py
  colab/baseline_features.py
  colab/scrape_player_stats.py
  colab/pipeline/__init__.py
  colab/pipeline/run_full_pipeline.py
  colab/pipeline/gcp_utils.py
  colab/pipeline/stats_report.py
)

for f in "${NEEDED_FILES[@]}"; do
  gsutil -o "GSUtil:parallel_process_count=1" cp "$REPO_ROOT/$f" "gs://$BUCKET/code/$f"
  echo "  uploaded $f"
done

echo "Source files uploaded."
echo ""

# ---------------------------------------------------------------------------
# Step 2 — Create one VM per variant
# ---------------------------------------------------------------------------
for VARIANT in A B C; do
  VM_NAME="swishnet-variant-$(echo "$VARIANT" | tr '[:upper:]' '[:lower:]')"
  echo "Creating $VM_NAME (variant $VARIANT) ..."

  STARTUP_FILE="/tmp/swishnet_startup_${VARIANT}.sh"
  cat > "$STARTUP_FILE" << EOF
#!/bin/bash
set -e
exec > /var/log/swishnet-startup.log 2>&1

echo "=== SwishNet Variant ${VARIANT} — \$(date) ==="

# ── System deps ────────────────────────────────────────────────────────────
apt-get update -q
apt-get install -y python3-pip python3-dev

# ── Python deps ────────────────────────────────────────────────────────────
# CPU-only torch (graph building needs no GPU)
pip3 install --quiet torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
pip3 install --quiet torch_geometric numpy pandas google-cloud-storage

# ── Pull code from GCS ─────────────────────────────────────────────────────
mkdir -p /swishnet/colab/pipeline
gsutil -m cp -r gs://${BUCKET}/code/colab/ /swishnet/colab/
cd /swishnet

# ── Download pre-computed shot data (step 1 output) ────────────────────────
mkdir -p /data/shot_data
echo "Downloading shot files..."
gsutil -m cp "gs://${BUCKET}/processed/shots_data/*.json" /data/shot_data/
echo "Shot files: \$(ls /data/shot_data/ | wc -l) games"

# ── Download supplemental data ─────────────────────────────────────────────
mkdir -p /data/supplemental_data
gsutil cp "gs://${BUCKET}/raw/player_shooting_stats_2016.csv" /data/supplemental_data/

# ── Run pipeline (steps 2, 3, 4 — graphs, aggregate arrays, stats report) ──
echo "Starting pipeline variant ${VARIANT}..."
python3 colab/pipeline/run_full_pipeline.py \
  --data-dir /data \
  --step 2 --step 3 --step 4 \
  --variant ${VARIANT} \
  --upload

echo "=== Variant ${VARIANT} complete — \$(date) ==="

# Auto-shutdown to avoid idle billing
shutdown -h now
EOF

  gcloud compute instances create "$VM_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --image-family=debian-11 \
    --image-project=debian-cloud \
    --scopes=storage-rw,logging-write \
    --metadata-from-file startup-script="$STARTUP_FILE"

  rm -f "$STARTUP_FILE"
  echo "  ✓ $VM_NAME created"
  echo ""
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
cat << 'SUMMARY'
All 3 VMs launched. They will auto-shutdown when done.

Monitor logs:
  gcloud compute ssh swishnet-variant-a --project=swishnet-489108 --zone=us-central1-a -- sudo tail -f /var/log/swishnet-startup.log
  gcloud compute ssh swishnet-variant-b --project=swishnet-489108 --zone=us-central1-a -- sudo tail -f /var/log/swishnet-startup.log
  gcloud compute ssh swishnet-variant-c --project=swishnet-489108 --zone=us-central1-a -- sudo tail -f /var/log/swishnet-startup.log

Check VM status:
  gcloud compute instances list --project=swishnet-489108 --filter="name~swishnet-variant"

GCS output (once complete):
  gsutil ls gs://swishnet-nba/processed/variants/
SUMMARY
