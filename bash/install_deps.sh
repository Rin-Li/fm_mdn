if ! [[ -n "${CONDA_PREFIX}" ]]; then
    echo "You are not inside a conda environment. Please activate your environment first."
    exit 1
fi

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install fvcore iopath
pip install --no-index --no-cache-dir pytorch3d==0.7.5 -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt212/download.html
# Or if on cpu: 
# pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
pip install -e .
rm -R *.egg-info

# Pypose
pip install --no-deps pypose
