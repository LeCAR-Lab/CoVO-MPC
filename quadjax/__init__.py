import os

# iclude dynamics, envs folders in the package as submodules
from quadjax import dynamics
from quadjax import envs

def get_package_path():
    return os.path.dirname(os.path.abspath(__file__))