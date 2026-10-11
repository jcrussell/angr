# pylint: disable=missing-class-docstring
from __future__ import annotations

import glob
import importlib
import importlib.metadata
import importlib.resources
import importlib.util
import os
import shutil
import subprocess
import sys
from distutils.command.build import build as st_build

from setuptools import Command, setup
from setuptools.command.develop import develop as st_develop
from setuptools.errors import LibError

# Import setuptools_rust to ensure an error is raised if not installed
try:
    _ = importlib.import_module("setuptools_rust")
except ImportError as err:
    raise Exception("angr requires setuptools-rust to build") from err


def _configure_z3() -> None:
    """Point z3-sys at the libz3 that ships in the ``z3-solver`` wheel.

    z3-sys 0.13 ships pregenerated bindings, so no C header is needed. It
    still has to know (a) where libz3 lives and (b) which Z3 version it is,
    because it probes pkg-config first (which would pick a system Z3 of some
    other version -- the bindings are tied to one release) and, failing that,
    assumes its minimum-supported version and hides newer API. Python and
    Rust must load one libz3 (docs/advanced-topics/rust_z3_sharing.rst), so
    the wheel's copy is the only acceptable answer.
    """
    spec = importlib.util.find_spec("z3")
    if spec is None or spec.origin is None:
        sys.stderr.write(
            "warning: z3-solver is not installed; z3-sys will fall back to pkg-config.\n"
            "Install z3-solver (see pyproject.toml for the pinned version) before building.\n"
        )
        return
    lib_dir = os.path.join(os.path.dirname(spec.origin), "lib")
    os.environ.setdefault("Z3_NO_PKG_CONFIG", "1")
    if os.path.isdir(lib_dir):
        os.environ.setdefault("Z3_LIBRARY_PATH_OVERRIDE", lib_dir)
    if not os.environ.get("Z3_SYS_Z3_VERSION"):
        version = importlib.metadata.version("z3-solver")
        # The wheel version carries a trailing build component ("5.1.0.0").
        os.environ["Z3_SYS_Z3_VERSION"] = ".".join(version.split(".")[:3])


_configure_z3()


def _resolve_pyvex_libdir() -> None:
    # The `libvex-ffi` Rust feature -- default-ON, see _rust_features below --
    # links the venv's pyvex/lib/libpyvex.so (see
    # docs/advanced-topics/rust_libvex_ffi.rst).
    # build.rs resolves this itself via `python3 -c 'import pyvex'`, but export
    # PYVEX_FFI_LIB_DIR here as the authoritative override so a setuptools-rust
    # build (which may run cargo in a different cwd/interpreter) links the same
    # pyvex the Python side loads. Harmless when the feature is off — build.rs
    # only reads it under CARGO_FEATURE_LIBVEX_FFI.
    if os.environ.get("PYVEX_FFI_LIB_DIR"):
        return
    try:
        import pyvex
    except ImportError:
        return
    lib_dir = os.path.join(os.path.dirname(pyvex.__file__), "lib")
    for so in ("libpyvex.so", "libpyvex.dylib"):
        if os.path.isfile(os.path.join(lib_dir, so)):
            os.environ["PYVEX_FFI_LIB_DIR"] = lib_dir
            return


_resolve_pyvex_libdir()


def _rust_features() -> list[str]:
    # `libvex-ffi` (native cold-block lifting through libpyvex.so) is ON by
    # default as of the human GO 2026-07-15 (bd angr-3trr7): a stock
    # `pip install -e .` builds WITH the native libVEX lifter. Set
    # ANGR_LIBVEX_FFI=0 (or false/off/no) to opt OUT -- the escape hatch.
    #
    # The resulting .so carries an rpath into the venv's pyvex/lib; wheel builds
    # MUST exclude libpyvex.so from the repair step (see .github/workflows/
    # wheels.yml -- mirrors the libz3 exclusion) so the wheel does not vendor a
    # second copy of libVEX with its own vex_control/arena globals. See the
    # "Shipping status" section of docs/advanced-topics/rust_libvex_ffi.rst.
    if os.environ.get("ANGR_LIBVEX_FFI", "").strip().lower() in ("0", "false", "off", "no"):
        return []
    if not os.environ.get("PYVEX_FFI_LIB_DIR"):
        # No libpyvex.so next to the installed pyvex (e.g. Windows, where pyvex
        # ships no shared object) -- degrade gracefully to the pyvex-callback
        # lift path rather than failing the build.
        sys.stderr.write(
            "note: no libpyvex.so was found next to the installed pyvex; building "
            "without the libvex-ffi feature (set ANGR_LIBVEX_FFI=0 to silence).\n"
        )
        return []
    return ["libvex-ffi"]


if sys.platform == "darwin":
    library_file = "unicornlib.dylib"
elif sys.platform in ("win32", "cygwin"):
    library_file = "unicornlib.dll"
else:
    library_file = "unicornlib.so"


def build_unicornlib():
    try:
        importlib.import_module("pyvex")
    except ImportError as e:
        raise LibError("You must install pyvex before building angr") from e

    env = os.environ.copy()
    env_data = (
        ("PYVEX_INCLUDE_PATH", "pyvex", "include"),
        ("PYVEX_LIB_PATH", "pyvex", "lib"),
        ("PYVEX_LIB_FILE", "pyvex", "lib\\pyvex.lib"),
    )
    for var, pkg, fnm in env_data:
        base = importlib.resources.files(pkg)
        for child in fnm.split("\\"):
            base = base.joinpath(child)
        env[var] = str(base)

    if sys.platform == "win32":
        cmd = ["nmake", "/f", "Makefile-win"]
    elif shutil.which("gmake") is not None:
        cmd = ["gmake"]
    else:
        cmd = ["make"]
    try:
        subprocess.run(cmd, cwd="native/unicornlib", env=env, check=True)
    except FileNotFoundError as err:
        raise LibError("Couldn't find " + cmd[0] + " in PATH") from err
    except subprocess.CalledProcessError as err:
        raise LibError("Error while building unicornlib: " + str(err)) from err

    shutil.rmtree("angr/lib", ignore_errors=True)
    os.mkdir("angr/lib")
    shutil.copy(os.path.join("native/unicornlib", library_file), "angr")


def clean_unicornlib():
    oglob = glob.glob("native/*.o")
    oglob += glob.glob("native/*.obj")
    oglob += glob.glob("native/*.so")
    oglob += glob.glob("native/*.dll")
    oglob += glob.glob("native/*.dylib")
    for fname in oglob:
        os.unlink(fname)


class build(st_build):
    def run(self, *args):
        self.execute(build_unicornlib, (), msg="Building unicornlib")
        super().run(*args)


class clean(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        self.execute(clean, (), msg="Cleaning unicornlib")


class develop(st_develop):
    def run(self):
        self.run_command("build")
        super().run()


cmdclass = {
    "build": build,
    "clean_unicornlib": clean,
    "develop": develop,
}


try:
    from setuptools_rust.build import build_rust as st_build_rust

    class build_rust(st_build_rust):
        # The extension itself is declared by the [[tool.setuptools-rust.ext-modules]]
        # table in pyproject.toml, which has no way to express a conditional cargo
        # feature. Append the opt-in ones here instead, leaving the default build
        # byte-identical to the table.
        def run(self):
            features = _rust_features()
            if features:
                for ext in getattr(self.distribution, "rust_extensions", None) or []:
                    ext.features = [*ext.features, *features]
            super().run()

    cmdclass["build_rust"] = build_rust
except ModuleNotFoundError:
    pass


try:
    from setuptools.command.editable_wheel import editable_wheel as st_editable_wheel

    class editable_wheel(st_editable_wheel):
        def run(self):
            self.run_command("build")
            super().run()

    cmdclass["editable_wheel"] = editable_wheel
except ModuleNotFoundError:
    pass


setup(cmdclass=cmdclass)
