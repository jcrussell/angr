# pylint: disable=missing-class-docstring
from __future__ import annotations

import glob
import importlib
import importlib.resources
import importlib.util
import os
import shutil
import subprocess
import sys

import setuptools_rust
from setuptools import Command, setup
from setuptools.command.build import build as st_build
from setuptools.command.develop import develop as st_develop
from setuptools.errors import LibError

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
        # importlib.resources.files() returns a MultiplexedPath for editable
        # installs, whose str() is not a filesystem path; resolve via the
        # imported package's location instead.
        pkg_file = importlib.import_module(pkg).__file__
        if pkg_file is None:
            raise LibError(f"Cannot locate {pkg} on disk; is it installed as a namespace package?")
        env[var] = os.path.join(os.path.dirname(pkg_file), *fnm.split("\\"))

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


def z3_loader():
    """angr/_z3.py, loaded out of the source tree: importing angr needs the extension we are building."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "angr", "_z3.py")
    spec = importlib.util.spec_from_file_location("angr_z3_loader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_z3():
    """Point the Rust build at the libz3 that ships in the z3-solver wheel.

    z3-sys probes pkg-config before honoring the override, and its search path would win, so a
    machine with a system-wide Z3 of another version would link that instead -- the bindings are
    tied to one Z3 release. Windows takes its import library from Z3's own release (see
    native/angr/Cargo.toml) and ignores both of these.
    """
    try:
        library_dir = z3_loader().library_dir()
    except ImportError as err:
        raise LibError("You must install z3-solver before building angr") from err

    os.environ.setdefault("Z3_NO_PKG_CONFIG", "1")
    os.environ.setdefault("Z3_LIBRARY_PATH_OVERRIDE", str(library_dir))


def build_protos():
    proto_files = sorted(glob.glob("angr/protos/*.proto"))
    cmd = [sys.executable, "-m", "grpc_tools.protoc", "-I.", "--python_out=.", "--pyi_out=.", *proto_files]
    try:
        subprocess.run(cmd, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as err:
        raise LibError("Error while generating protobuf modules: " + str(err)) from err


def clean_unicornlib():
    oglob = glob.glob("native/*.o")
    oglob += glob.glob("native/*.obj")
    oglob += glob.glob("native/*.so")
    oglob += glob.glob("native/*.dll")
    oglob += glob.glob("native/*.dylib")
    for fname in oglob:
        os.unlink(fname)


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


class build_rust(setuptools_rust.build_rust):
    def run(self):
        configure_z3()
        # The extension is declared by the [[tool.setuptools-rust.ext-modules]] table in
        # pyproject.toml, which cannot express a conditional cargo feature; append it here.
        features = _rust_features()
        if features:
            for ext in getattr(self.distribution, "rust_extensions", None) or []:
                ext.features = [*ext.features, *features]
        super().run()


class build(st_build):
    def run(self, *args):
        self.execute(build_protos, (), msg="Generating protobuf modules")
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
    "build_rust": build_rust,
    "clean_unicornlib": clean,
    "develop": develop,
}


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
