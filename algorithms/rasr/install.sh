pip install "setuptools<81"
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install triton

# Install the RASR package copied in from the parent repo, together with its
# dependencies (pip resolves torchtitan from PyPI).
pip install /install/_local_submodules/rasr
