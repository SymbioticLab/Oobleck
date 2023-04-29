from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "0.1.0"

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

ext_modules = [
    Pybind11Extension(
        "oobleck.cplanning",
        ["oobleck/csrc/planning/gpu_stage_mapping.cpp"],
        define_macros=[("VERSION_INFO", __version__)],
    ),
]

setup(
    name="oobleck",
    version=__version__,
    author="Insu Jang",
    author_email="insujang@umich.edu",
    description="Resilient Distributed Training Framework",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
)
