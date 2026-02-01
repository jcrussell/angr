"""
Shellcode generators for differential testing.

Provides functions to generate test cases for different operation categories.
"""
from __future__ import annotations

from .arithmetic import generate_add_tests, generate_sub_tests, generate_mul_tests, generate_div_tests
from .bitwise import generate_and_tests, generate_or_tests, generate_xor_tests, generate_shift_tests
from .memory import generate_mov_tests, generate_push_pop_tests, generate_lea_tests

__all__ = [
    # Arithmetic
    "generate_add_tests",
    "generate_sub_tests",
    "generate_mul_tests",
    "generate_div_tests",
    # Bitwise
    "generate_and_tests",
    "generate_or_tests",
    "generate_xor_tests",
    "generate_shift_tests",
    # Memory
    "generate_mov_tests",
    "generate_push_pop_tests",
    "generate_lea_tests",
]
