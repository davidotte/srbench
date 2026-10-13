set -euo pipefail

pip install --no-cache-dir "setuptools<81"

# Install the CUDA build of torch BEFORE flash-ansr so its torch dependency is
# already satisfied and pip does not pull the CPU-only wheel. Mirrors rasr.
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126
pip install --no-cache-dir triton

pip install --no-cache-dir flash-ansr==0.4.5
pip install --no-cache-dir huggingface_hub
