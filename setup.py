# pylint: disable=missing-class-docstring
from __future__ import annotations

import glob
import importlib
import importlib.resources
import os
import shutil
import subprocess
import sys
import sysconfig
from distutils.command.build import build as st_build

from setuptools import Command, setup
from setuptools.command.develop import develop as st_develop
from setuptools.errors import LibError

# Import setuptools_rust to ensure an error is raised if not installed
try:
    _ = importlib.import_module("setuptools_rust")
except ImportError as err:
    raise Exception("angr requires setuptools-rust to build") from err


def _resolve_z3_header() -> None:
    # z3-sys's build script reads Z3_SYS_Z3_HEADER first; if unset it falls back
    # to pkg-config + its bundled wrapper.h. Fresh installs without pkg-config
    # registration for z3 fail with "Unable to generate bindings: NotExist z3.h".
    # Probe common locations preemptively so the typical setups (libz3-dev,
    # z3-devel, brew z3, or headers copied into the venv) build without manual
    # env-var setup, and emit an actionable hint when nothing is found.
    if os.environ.get("Z3_SYS_Z3_HEADER"):
        return

    candidates: list[str] = []

    # 1. venv site-packages/z3/include — rare (PyPI z3-solver omits headers),
    #    but covers the case where they were copied in to match libz3.so.
    purelib = sysconfig.get_paths().get("purelib")
    if purelib:
        candidates.append(os.path.join(purelib, "z3", "include", "z3.h"))

    # 2. pkg-config — z3-sys would do this too, but we probe so the path that
    #    succeeds here is reused in the error message if no header exists.
    if shutil.which("pkg-config"):
        try:
            result = subprocess.run(
                ["pkg-config", "--variable=includedir", "z3"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            inc = result.stdout.strip()
            if inc:
                candidates.append(os.path.join(inc, "z3.h"))
        except (subprocess.SubprocessError, OSError):
            pass

    # 3. Standard system paths.
    for inc in ("/usr/include", "/usr/local/include", "/opt/homebrew/include", "/opt/local/include"):
        candidates.append(os.path.join(inc, "z3.h"))

    for path in candidates:
        if os.path.isfile(path):
            os.environ["Z3_SYS_Z3_HEADER"] = path
            return

    sys.stderr.write(
        "warning: could not locate z3.h for the Rust extension. Install Z3 dev headers:\n"
        "  Debian/Ubuntu: apt install libz3-dev pkg-config\n"
        "  Fedora/RHEL:   dnf install z3-devel pkgconf-pkg-config\n"
        "  macOS:         brew install z3 pkg-config\n"
        "Or set Z3_SYS_Z3_HEADER=/path/to/z3.h before building.\n"
    )


_resolve_z3_header()


def _resolve_pyvex_libdir() -> None:
    # The non-default `libvex-ffi` Rust feature links the venv's
    # pyvex/lib/libpyvex.so (see docs/advanced-topics/rust_libvex_ffi.rst).
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
