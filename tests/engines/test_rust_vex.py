"""
Tests for the Rust VEX execution engine integration.
"""
import unittest
import logging

# Suppress logging during tests
logging.getLogger("angr").setLevel(logging.ERROR)


class TestRustVEXEngineImport(unittest.TestCase):
    """Test that the Rust VEX engine can be imported."""

    def test_import_rust_vex_module(self):
        """Test importing the rust_vex module."""
        try:
            from angr.engines.rust_vex import (
                RustVEXMixin,
                RustVEXEngineWrapper,
                RUST_ENGINE_AVAILABLE,
            )
            self.assertIsNotNone(RustVEXMixin)
            self.assertIsNotNone(RustVEXEngineWrapper)
            # RUST_ENGINE_AVAILABLE can be True or False depending on build
        except ImportError as e:
            self.skipTest(f"rust_vex module not available: {e}")

    def test_import_uber_engine_rust(self):
        """Test importing UberEngineRust from engines."""
        try:
            from angr.engines import UberEngineRust, RUST_ENGINE_AVAILABLE
            if RUST_ENGINE_AVAILABLE:
                self.assertIsNotNone(UberEngineRust)
            else:
                # UberEngineRust won't be defined if rust not available
                pass
        except ImportError:
            pass  # Expected if rustylib not built with vex-engine


class TestRustVEXEngineWrapper(unittest.TestCase):
    """Test the RustVEXEngineWrapper standalone interface."""

    def setUp(self):
        try:
            from angr.engines.rust_vex import (
                RustVEXEngineWrapper,
                RUST_ENGINE_AVAILABLE,
            )
            if not RUST_ENGINE_AVAILABLE:
                self.skipTest("Rust VEX engine not available")
            self.wrapper_class = RustVEXEngineWrapper
        except ImportError:
            self.skipTest("rust_vex module not available")

    def test_create_amd64_engine(self):
        """Test creating an AMD64 engine."""
        engine = self.wrapper_class("amd64")
        self.assertEqual(engine.pc, 0)

    def test_create_x86_engine(self):
        """Test creating an x86 engine."""
        engine = self.wrapper_class("x86")
        self.assertEqual(engine.pc, 0)

    def test_create_arm_engine(self):
        """Test creating an ARM engine."""
        engine = self.wrapper_class("arm")
        self.assertEqual(engine.pc, 0)

    def test_create_arm64_engine(self):
        """Test creating an ARM64 engine."""
        engine = self.wrapper_class("arm64")
        self.assertEqual(engine.pc, 0)

    def test_register_access_amd64(self):
        """Test register access on AMD64."""
        engine = self.wrapper_class("amd64")

        # Set and get register
        engine.set_register("rax", 0x12345678DEADBEEF)
        self.assertEqual(engine.get_register("rax"), 0x12345678DEADBEEF)

        # Set PC
        engine.pc = 0x401000
        self.assertEqual(engine.pc, 0x401000)

    def test_memory_access(self):
        """Test memory mapping and access."""
        engine = self.wrapper_class("amd64")

        # Map memory
        engine.map_memory(0x1000, 0x1000)

        # Write and read
        engine.write_memory(0x1000, b"\x01\x02\x03\x04")
        data = engine.read_memory(0x1000, 4)
        self.assertEqual(data, b"\x01\x02\x03\x04")

    def test_memory_with_data(self):
        """Test mapping memory with initial data."""
        engine = self.wrapper_class("amd64")

        # Map with data
        engine.map_memory_data(0x2000, b"\xDE\xAD\xBE\xEF")
        data = engine.read_memory(0x2000, 4)
        self.assertEqual(data, b"\xDE\xAD\xBE\xEF")

    def test_fork(self):
        """Test forking the engine."""
        engine = self.wrapper_class("amd64")
        engine.set_register("rax", 100)
        engine.pc = 0x401000

        # Fork
        forked = engine.fork()

        # Verify forked has same state
        self.assertEqual(forked.get_register("rax"), 100)
        self.assertEqual(forked.pc, 0x401000)

        # Modify forked - should not affect original
        forked.set_register("rax", 200)
        forked.pc = 0x402000

        self.assertEqual(engine.get_register("rax"), 100)
        self.assertEqual(engine.pc, 0x401000)
        self.assertEqual(forked.get_register("rax"), 200)
        self.assertEqual(forked.pc, 0x402000)

    def test_hooks(self):
        """Test hook management."""
        engine = self.wrapper_class("amd64")

        # Add hook
        engine.add_hook(0x401000)

        # Can't directly check if hooked via wrapper, but should not error
        engine.remove_hook(0x401000)

    def test_stats(self):
        """Test getting engine stats."""
        engine = self.wrapper_class("amd64")

        stats = engine.stats()
        self.assertIn("cached_blocks", stats)
        self.assertIn("hooks", stats)
        self.assertIn("memory_regions", stats)

    def test_get_all_registers(self):
        """Test getting all registers at once."""
        engine = self.wrapper_class("amd64")
        engine.set_register("rax", 1)
        engine.set_register("rbx", 2)
        engine.set_register("rcx", 3)

        regs = engine.get_registers()
        self.assertEqual(regs.get("rax"), 1)
        self.assertEqual(regs.get("rbx"), 2)
        self.assertEqual(regs.get("rcx"), 3)


class TestRustVEXMixin(unittest.TestCase):
    """Test the RustVEXMixin integration with angr."""

    def test_mixin_initialization(self):
        """Test that the mixin can be initialized with a project."""
        try:
            import angr
            from angr.engines.rust_vex import RustVEXMixin, RUST_ENGINE_AVAILABLE

            if not RUST_ENGINE_AVAILABLE:
                self.skipTest("Rust engine not available")

            # Create a simple project
            proj = angr.Project(
                "/bin/true",  # Use a simple binary
                auto_load_libs=False,
            )

            # Create mixin instance
            mixin = RustVEXMixin(proj)
            self.assertTrue(mixin.rust_engine_available)

        except ImportError as e:
            self.skipTest(f"angr or dependencies not available: {e}")
        except FileNotFoundError:
            self.skipTest("/bin/true not found")


if __name__ == "__main__":
    unittest.main()
