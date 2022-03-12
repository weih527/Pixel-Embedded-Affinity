import distutils.core
import Cython.Build
import numpy as np
distutils.core.setup(
    ext_modules = Cython.Build.cythonize("CVPPP_evaluate.pyx"),
    include_dirs = [np.get_include()])