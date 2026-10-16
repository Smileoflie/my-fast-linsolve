"""Conditional Cython build.

If Cython and a C compiler are available, ``_cython_core`` is compiled from
``src/linsolve/_cython_core.pyx``. Otherwise the package installs as pure
Python and falls back to the Numba / pure-Python backends at runtime -- the
extension is optional by design.

Two details matter for reproducible releases:

* the extension is compiled with ``-O3`` on gcc-like toolchains (setuptools
  otherwise passes ``-O1`` for mingw32, which measurably loses to Numba);
* a *silent* degradation is easy to miss -- that is how earlier releases ended
  up shipping a ``py3-none-any`` wheel with no compiled kernel at all. The
  reason is now printed to stderr, and ``LINSOLVE_REQUIRE_CYTHON=1`` turns the
  skip into a hard error (CI and wheel builders set it so a broken build cannot
  ship unnoticed).
"""
import os
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

HERE = os.path.abspath(os.path.dirname(__file__))
# setuptools requires setup() arguments to be relative, slash-separated paths;
# the absolute twin is only used for existence checks (CWD need not be the root)
CYTHON_SRC = "src/linsolve/_cython_core.pyx"
CYTHON_SRC_ABS = os.path.join(HERE, *CYTHON_SRC.split("/"))
REQUIRE_CYTHON = os.environ.get("LINSOLVE_REQUIRE_CYTHON") == "1"

_OPT_FLAGS = ("-O3", "-funroll-loops")


class OptionalBuildExt(build_ext):
    """Compile the extension when possible; keep the package installable when not."""

    def build_extension(self, ext):
        toolchain = getattr(self.compiler, "compiler_type", "")
        if toolchain in ("mingw32", "unix", "cygwin"):
            args = list(ext.extra_compile_args or [])
            for flag in _OPT_FLAGS:
                if flag not in args:
                    args.append(flag)
            ext.extra_compile_args = args
        try:
            super().build_extension(ext)
        except Exception as exc:  # pragma: no cover - toolchain dependent
            if REQUIRE_CYTHON:
                raise
            print(
                f"\nlinsolve: WARNING - could not build the Cython extension "
                f"({type(exc).__name__}: {exc}).\n"
                f"linsolve: installing WITHOUT it; the runtime will fall back to "
                f"the Numba or pure-Python backend.\n"
                f"linsolve: set LINSOLVE_REQUIRE_CYTHON=1 to make this fatal.\n",
                file=sys.stderr,
            )


def _requested_compiler():
    """Pick up --compiler=... so the probe uses the toolchain that will build."""
    argv = sys.argv
    for i, arg in enumerate(argv):
        if arg.startswith("--compiler="):
            return arg.split("=", 1)[1]
        if arg == "--compiler" and i + 1 < len(argv):
            return argv[i + 1]
    return None


def _compiler_available() -> bool:
    """Probe for a working C compiler before declaring the extension.

    Declaring an extension that cannot be built would make bdist_wheel tag the
    artifact for this platform while it contains no compiled kernel -- worse
    than a clean pure-Python wheel, because pip would then happily hand that
    wheel to every other platform too. Probing up front keeps the tag honest.

    Set LINSOLVE_SKIP_COMPILER_PROBE=1 to skip the probe (and let the build
    succeed or fail for itself).
    """
    if os.environ.get("LINSOLVE_SKIP_COMPILER_PROBE") == "1":
        return True
    try:
        import tempfile

        from setuptools._distutils.ccompiler import new_compiler

        cc = new_compiler(compiler=_requested_compiler())
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "probe.c")
            with open(src, "w", encoding="ascii") as handle:
                handle.write("int main(void) { return 0; }\n")
            objects = cc.compile([src], output_dir=tmp)
            cc.link_shared_object(objects, os.path.join(tmp, "probe.pyd"))
        return True
    except Exception as exc:
        print(f"linsolve: no usable C compiler ({type(exc).__name__}: {exc})",
              file=sys.stderr)
        return False


def _ext_modules():
    if os.environ.get("LINSOLVE_NO_CYTHON") == "1":
        # explicit opt-out: build the universal py3-none-any fallback wheel
        print("linsolve: LINSOLVE_NO_CYTHON=1 - building a pure-Python wheel.",
              file=sys.stderr)
        return []
    if not os.path.exists(CYTHON_SRC_ABS):
        return []
    try:
        from Cython.Build import cythonize
        import numpy
    except ImportError as exc:
        message = (
            f"Cython extension skipped ({exc}). Install 'Cython' and 'numpy' "
            "(plus a C compiler) to build it, otherwise the package uses the "
            "Numba / pure-Python backends."
        )
        if REQUIRE_CYTHON:
            raise RuntimeError(message) from exc
        print(f"\nlinsolve: WARNING - {message}\n", file=sys.stderr)
        return []
    if not _compiler_available():
        message = (
            "no C compiler found - skipping the Cython extension. The installed "
            "package falls back to the Numba / pure-Python backends."
        )
        if REQUIRE_CYTHON:
            raise RuntimeError("linsolve: " + message)
        print(f"\nlinsolve: WARNING - {message}\n", file=sys.stderr)
        return []
    return cythonize(
        [
            Extension(
                "linsolve._cython_core",
                [CYTHON_SRC],
                include_dirs=[numpy.get_include()],
            )
        ],
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    )


setup(ext_modules=_ext_modules(), cmdclass={"build_ext": OptionalBuildExt})
