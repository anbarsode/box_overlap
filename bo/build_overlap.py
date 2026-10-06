"""Build the C++ extension in place: python -m bo.build_overlap.

Requires the active environment's pybind11 and a C++17 compiler. No package
installation or downloads are performed. Rebuild after changing overlap_core.cpp
or changing Python environments. The resulting extension is specific to this Python.
"""
from pathlib import Path
import os
import shlex
import subprocess
import sysconfig
import pybind11


def build():
    folder = Path(__file__).resolve().parent
    output = folder / ("_overlap" + sysconfig.get_config_var("EXT_SUFFIX"))
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    command = shlex.split(os.environ.get("CXX", "c++")) + [
        "-O3", "-DNDEBUG", "-std=c++17", "-shared", "-fPIC", "-Wall", "-Wextra",
        "-I" + pybind11.get_include(), "-I" + sysconfig.get_path("include"),
        str(folder / "overlap_core.cpp"), "-o", str(temporary),
    ]
    # Deliberately omit fast-math: NaNs/infinities should remain visible.
    subprocess.run(command, check=True)
    temporary.replace(output)
    print(f"Built {output}")


if __name__ == "__main__":
    build()
