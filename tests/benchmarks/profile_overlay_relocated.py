#!/usr/bin/env python3
"""Wall-clock profile of _overlay_relocated_sections.

Spawns a child subprocess (4GB RLIMIT_AS) so this can run safely on the
8GB no-swap host. Measures:
  - Time per call to _overlay_relocated_sections on fauxware
  - Section-count breakdown by size
  - Top-N sections by time
"""
import multiprocessing
import os
import resource
import sys
import time


EXAMPLES_DIR = os.path.expanduser("~/repos/angr-examples/examples")


def _child(out_q, example_name, n_iters):
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024 * 1024 * 1024,
                                            resource.getrlimit(resource.RLIMIT_AS)[1]))
    _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    import angr
    from angr.exploration import RustExplorationManager

    binary = os.path.join(EXAMPLES_DIR, example_name, example_name)
    if not os.path.exists(binary):
        # fauxware binary lives at top-level
        binary = os.path.join(EXAMPLES_DIR, example_name, "fauxware")
    proj = angr.Project(binary, auto_load_libs=False)
    state = proj.factory.entry_state()

    # Construct manager so internal caches/setup are warm.
    mgr = RustExplorationManager(proj, [state.copy()])

    # Import the inner symbols we want to measure.
    from angr.rustylib.vex_engine import RustSimState as _RustSimState
    is_le = proj.arch.memory_endness == 'Iend_LE'

    # We will time three flavors:
    #   1) the full _overlay_relocated_sections call
    #   2) the inner state.memory.load loop, by section size buckets
    #   3) just the loop body without map_memory_data (to isolate FFI)
    arch = proj.arch
    objs = proj.loader.all_objects

    # First: full call timing.
    full_times = []
    for _ in range(n_iters):
        rust_state = _RustSimState(arch.name, little_endian=is_le)
        t0 = time.perf_counter_ns()
        mgr._overlay_relocated_sections(state, rust_state)
        t1 = time.perf_counter_ns()
        full_times.append(t1 - t0)

    # Section-by-section: load only (no FFI), then load+map (with FFI).
    from angr.exploration._constants import MAX_OVERLAY_SECTION_SIZE

    section_load_times = []
    section_concrete_count = 0
    section_symbolic_count = 0
    section_skip_count = 0
    total_load_ns = 0
    total_eval_ns = 0
    total_ffi_ns = 0
    sizes_hist = {"<256": 0, "<4K": 0, "<16K": 0, "<64K": 0, "skipped_oversize": 0}
    for obj in objs:
        if obj.binary is None or not hasattr(obj, 'sections'):
            continue
        for section in obj.sections:
            if section.memsize <= 0:
                continue
            if section.memsize >= MAX_OVERLAY_SECTION_SIZE:
                sizes_hist["skipped_oversize"] += 1
                continue
            if section.memsize < 256:
                sizes_hist["<256"] += 1
            elif section.memsize < 4096:
                sizes_hist["<4K"] += 1
            elif section.memsize < 16384:
                sizes_hist["<16K"] += 1
            else:
                sizes_hist["<64K"] += 1

            t0 = time.perf_counter_ns()
            try:
                val = state.memory.load(section.min_addr, section.memsize,
                                        endness='Iend_BE', inspect=False,
                                        disable_actions=True)
            except Exception:
                section_skip_count += 1
                continue
            t1 = time.perf_counter_ns()
            total_load_ns += (t1 - t0)

            t0 = time.perf_counter_ns()
            sym = val.symbolic
            if sym:
                section_symbolic_count += 1
                continue
            section_concrete_count += 1
            t1 = time.perf_counter_ns()
            try:
                data = state.solver.eval(val).to_bytes(section.memsize, 'big')
            except Exception:
                section_skip_count += 1
                continue
            t2 = time.perf_counter_ns()
            total_eval_ns += (t2 - t1)

            rust_state2 = _RustSimState(arch.name, little_endian=is_le)
            t3 = time.perf_counter_ns()
            rust_state2.map_memory_data(section.min_addr, data, 7)
            t4 = time.perf_counter_ns()
            total_ffi_ns += (t4 - t3)

            section_load_times.append((section.min_addr, section.memsize, t1 - t0))

    section_load_times.sort(key=lambda x: -x[2])
    n_sections_total = sum(sizes_hist.values()) - sizes_hist["skipped_oversize"]

    out_q.put({
        "full_times_ms": [t / 1e6 for t in full_times],
        "n_iters": n_iters,
        "n_sections": n_sections_total,
        "n_sections_oversized": sizes_hist["skipped_oversize"],
        "n_concrete": section_concrete_count,
        "n_symbolic": section_symbolic_count,
        "n_skipped": section_skip_count,
        "size_hist": sizes_hist,
        "total_load_ms": total_load_ns / 1e6,
        "total_eval_ms": total_eval_ns / 1e6,
        "total_ffi_ms": total_ffi_ns / 1e6,
        "top10_sections": section_load_times[:10],
    })


def main():
    multiprocessing.set_start_method('spawn', force=True)
    out_q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_child, args=(out_q, "fauxware", 20))
    p.start()
    result = out_q.get(timeout=120)
    p.join(timeout=10)

    times = result["full_times_ms"]
    print(f"=== _overlay_relocated_sections wall-clock profile (fauxware) ===")
    print(f"Iterations: {result['n_iters']}")
    print(f"Per-call (ms): min={min(times):.3f} max={max(times):.3f} "
          f"mean={sum(times)/len(times):.3f} median={sorted(times)[len(times)//2]:.3f}")
    print()
    print(f"Section breakdown (one full sweep):")
    print(f"  Sections in range:    {result['n_sections']}")
    print(f"  Sections oversized:   {result['n_sections_oversized']}")
    print(f"  Concrete (overlaid):  {result['n_concrete']}")
    print(f"  Symbolic (skipped):   {result['n_symbolic']}")
    print(f"  Errored (skipped):    {result['n_skipped']}")
    print()
    print(f"Size histogram: {result['size_hist']}")
    print()
    print(f"Total time breakdown (one full sweep):")
    print(f"  state.memory.load:    {result['total_load_ms']:.3f} ms")
    print(f"  solver.eval+to_bytes: {result['total_eval_ms']:.3f} ms")
    print(f"  rust map FFI:         {result['total_ffi_ms']:.3f} ms")
    print()
    print(f"Top 10 slowest sections by load time:")
    for addr, size, ns in result['top10_sections']:
        print(f"  addr=0x{addr:x} size={size} load_ms={ns/1e6:.3f}")


if __name__ == "__main__":
    main()
