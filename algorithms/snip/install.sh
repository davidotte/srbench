pip install "setuptools<81"

# SNIP's vendored code (symbolicregression/envs/environment.py) still imports
# `distutils`, and its feature-selection helper imports `numpy.compat.py3k` --
# both removed upstream, hence the python<3.12 / numpy<2 pins in
# environment.yml.
#
# torch 2.0.0 is the version SNIP's environment.yml uses and the released
# snip-e2e-sr.pth checkpoint was produced with; newer versions break SNIP's
# code (torch>=2.6's weights_only=True default on torch.load, removed
# Byte-tensor mask support in masked_fill_). The cu118 build is used instead of
# PyPI's default cu117, which has no kernels for Ada GPUs such as the L40S
# (compute capability 8.9) and fails with "no kernel image is available for
# execution on the device".
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu118
pip install triton==2.0.0
pip install sympytorch
pip install numexpr seaborn matplotlib nevergrad tqdm requests

# Vendor SNIP's code (a loose collection of top-level modules, not an
# installable package) into site-packages so `import symbolicregression`,
# `from parsers import get_parser`, `from model import SNIPSymbolicRegressor`
# and `from LSO_fit import lso_fit, LSOFitNeverGrad` resolve at runtime.
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
cp -r /install/_local_submodules/snip/symbolicregression "$SITE_PACKAGES/"
cp /install/_local_submodules/snip/parsers.py "$SITE_PACKAGES/"
cp /install/_local_submodules/snip/model.py "$SITE_PACKAGES/"
cp /install/_local_submodules/snip/LSO_fit.py "$SITE_PACKAGES/"
cp /install/_local_submodules/snip/alg_update.py "$SITE_PACKAGES/"
cp /install/_local_submodules/snip/const_opt.py "$SITE_PACKAGES/"
