"""Replay a bench's found stdin against the real binary (angr-vmi2l.120).

A bench's own ``test()`` checks the answer angr printed, not that the answer
works: a native proc that wrote the wrong bytes into ``posix.dumps(0)`` (scanf
storing raw bytes, fgets losing the seed) can still hand ``test()`` a string
that matches. ``run_single.py --test`` therefore records every value the
bench's entry function (``ReplaySpec.entry``) returns and, for a bench listed
in ``REPLAY_ORACLES``, runs the binary with each one on stdin and asserts its
stdout.

Only a bench whose entry function returns the found state's ``posix.dumps(0)``
(possibly stripped) belongs here, and only an x86-64 Linux ELF that runs on
the host. ``ekopartyctf2016_rev250`` already replays inside its own
``test()``; the synthetic ``cli_ctype_fprintf`` does too.
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class ReplaySpec:
    """How to replay one bench: ``binary`` is relative to the example dir."""

    binary: str
    expect_stdout: bytes
    # Appended to the recorded stdin, for an entry that strips the newline.
    stdin_suffix: bytes = b""
    # The solve.py function that test() calls and that returns the found stdin.
    entry: str = "main"


REPLAY_ORACLES = {
    "fauxware": ReplaySpec(
        binary="fauxware",
        expect_stdout=b"Welcome to the admin console, trusted user!",
        entry="basic_symbolic_execution",
    ),
    "defcamp_r100": ReplaySpec(binary="r100", expect_stdout=b"Nice!", stdin_suffix=b"\n"),
}

REPLAY_TIMEOUT_SECS = 10


def host_can_replay() -> bool:
    return platform.system() == "Linux" and platform.machine() == "x86_64"


def replay(spec: ReplaySpec, example_dir: str, stdin: bytes) -> str | None:
    """Run the binary on ``stdin``; return a failure message, or None on success."""
    if not isinstance(stdin, bytes):
        return f"{spec.entry}() returned {type(stdin).__name__}, not the found stdin bytes"
    binary = os.path.join(example_dir, spec.binary)
    proc = subprocess.run(
        [binary],
        input=stdin + spec.stdin_suffix,
        capture_output=True,
        timeout=REPLAY_TIMEOUT_SECS,
        check=False,
    )
    if spec.expect_stdout in proc.stdout:
        return None
    return (
        f"replaying stdin {stdin!r} on {spec.binary} did not print {spec.expect_stdout!r} "
        f"(exit {proc.returncode}, stdout {proc.stdout[-200:]!r})"
    )
