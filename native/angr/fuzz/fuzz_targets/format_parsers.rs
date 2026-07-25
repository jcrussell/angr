//! Fuzz the pure printf/scanf format-string sub-parsers — width digits and
//! length modifiers — over arbitrary bytes. These are the width/truncation
//! surfaces behind angr-vfhyx and angr-n0irt.4. Both are total functions that
//! must never panic and must never read past `fmt.len()`; the harness asserts
//! the reported `consumed` count stays in bounds.
#![no_main]

use libfuzzer_sys::fuzz_target;

use rustylib::fuzz_api::{parse_length_modifier, parse_width_digits};

fuzz_target!(|data: &[u8]| {
    // Sweep every start offset (including `data.len()`, the empty-tail case)
    // so a bug that only trips near the end is reachable.
    for start in 0..=data.len() {
        let (_width, consumed) = parse_width_digits(data, start);
        assert!(
            start + consumed <= data.len(),
            "width parse over-consumed: start={start} consumed={consumed} len={}",
            data.len()
        );

        let (_lm, consumed) = parse_length_modifier(data, start);
        assert!(
            start + consumed <= data.len(),
            "length-modifier parse over-consumed: start={start} consumed={consumed} len={}",
            data.len()
        );
    }
});
