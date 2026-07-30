#!/usr/bin/env bash
# =============================================================================
# setup_gcp.sh  —  One-time GCP infrastructure setup for SwishNet
#
# Run locally with gcloud authenticated to the GCP project:
#   bash setup_gcp.sh
#
# Prerequisites:
#   gcloud auth login
#   gcloud config set project swishnet-489108
# =============================================================================
set -euo pipefail

PROJECT="swishnet-489108"
BUCKET="swishnet-nba"
REGION="us-central1"
ZONE="us-central1-a"

# ── Bucket ────────────────────────────────────────────────────────────────────
echo "Creating GCS bucket gs://${BUCKET} …"
gcloud storage buckets create "gs://${BUCKET}" \
  --project="${PROJECT}" \
  --location="${REGION}" \
  --uniform-bucket-level-access

# ── Upload supplemental data (one-time) ───────────────────────────────────────
echo "Uploading supplemental data …"
gcloud storage cp colab/data/supplemental_data/pbp_cache.csv \
  "gs://${BUCKET}/raw/"
gcloud storage cp colab/data/supplemental_data/player_shooting_stats_2016.csv \
  "gs://${BUCKET}/raw/"

# ── CPU VM: data pipeline ──────────────────────────────────────────────────────
# n1-standard-4 = 4 vCPU, 15 GB RAM, ~$0.19/hr. No GPU needed for data work.
echo "Creating CPU data pipeline VM (swishnet-data) …"
gcloud compute instances create swishnet-data \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --machine-type=n1-standard-4 \
  --image-family=pytorch-latest-cpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=300GB \
  --scopes=storage-rw

# ── Budget alert at $25 (50% of $50 credit) ───────────────────────────────────
# Replace BILLING_ACCOUNT_ID below with your ID from GCP Console > Billing.
echo ""
echo "⚠  To create a budget alert, replace BILLING_ACCOUNT_ID below and run:"
echo ""
echo "  gcloud billing budgets create \\"
echo "    --billing-account=BILLING_ACCOUNT_ID \\"
echo "    --display-name=\"SwishNet budget alert\" \\"
echo "    --budget-amount=25 \\"
echo "    --threshold-rule=percent=0.5 \\"
echo "    --threshold-rule=percent=0.9"
echo ""

# ── Post-VM-creation setup instructions ───────────────────────────────────────
echo "============================================================"
echo "VM created. SSH in and run:"
echo ""
echo "  pip install torch-geometric py7zr scikit-learn tqdm google-cloud-storage"
echo ""
echo "Then clone the repo and run the pipeline:"
echo ""
echo "  git clone https://github.com/<your-org>/SwishNet.git"
echo "  cd SwishNet"
echo "  python colab/pipeline/run_full_pipeline.py --data-dir /data --all --upload"
echo "============================================================"
