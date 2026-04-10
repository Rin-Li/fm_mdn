if ! [[ -n "${CONDA_PREFIX}" ]]; then
    echo "You are not inside a conda environment. Please activate your environment first."
    exit 1
fi

if ! [[ -n "${COPPELIASIM_ROOT}" ]]; then
    echo "COPPELIASIM_ROOT is not defined."
    exit 1
fi

# Download Coppelia sim if not present
if ! [[-e $COPPELIASIM_ROOT]]; then
    wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
    mkdir -p $COPPELIASIM_ROOT && tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C $COPPELIASIM_ROOT --strip-components 1
    rm -rf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
fi

# Install PyRep and RLBench
pip install -r https://raw.githubusercontent.com/stepjam/PyRep/master/requirements.txt
pip install git+https://github.com/stepjam/PyRep.git
pip install git+https://github.com/stepjam/RLBench.git
