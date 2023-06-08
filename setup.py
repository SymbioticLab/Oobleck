import os
import subprocess
import sys
from pathlib import Path
import shutil

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

__version__ = "0.1.0"

# CMake build code borrowed from:
# https://github.com/pybind/cmake_example/blob/master/setup.py


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CmakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        # Using this requires trailing slash for auto-detection & inclusion of
        # auxiliary "native" libs
        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_CXX_FLAGS_RELEASE='-O1 -DNDEBUG'",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]
        build_args = []
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # In this example, we pass in the version to C++. You might not need to.
        cmake_args += [f"-DEXAMPLE_VERSION_INFO={self.distribution.get_version()}"]

        # Using Ninja-build since it a) is available as a wheel and b)
        # multithreads automatically.
        # Users can override the generator with CMAKE_GENERATOR in CMake
        # 3.15+.
        if not cmake_generator or cmake_generator == "Ninja":
            try:
                import ninja

                ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
                cmake_args += [
                    "-GNinja",
                    f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                ]
            except ImportError:
                pass

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        log = Path("/cmake.log")

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args], cwd=build_temp, check=True
        )

    def run(self):
        # First, run the original build_ext command
        build_ext.run(self)

        # Then, move the built library to the desired location
        source = Path(self.build_lib).glob("pipeline_template*.so")
        if not source:
            raise RuntimeError("Cannot find built library")

        source: Path = list(source)[0]
        target_dir = Path.cwd().joinpath("oobleck/csrc/planning")
        assert target_dir.is_dir()
        shutil.move(source, target_dir.joinpath(source.name))


setup(
    name="oobleck",
    version=__version__,
    author="Insu Jang",
    author_email="insujang@umich.edu",
    description="Resilient Distributed Training Framework",
    long_description="",
    ext_modules=[CMakeExtension("pipeline_template", Path("oobleck/csrc").absolute())],
    cmdclass={"build_ext": CmakeBuild},
    zip_safe=False,
    python_requires=">=3.8",
)
