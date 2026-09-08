//! The misroute `inspect_dispatch!` exists to prevent: a `reg_write` body
//! reaching for the identically-typed `inspect_reg_read` slot. The placeholder
//! is the only sanctioned route to a slot, so naming one directly is rejected
//! even though the body forwards correctly and would compile.
struct Holder {
    inspect_reg_read: Option<u8>,
    inspect_reg_write: Option<u8>,
}

impl Holder {
    fn with_inspect_cb<R>(
        &self,
        slot: Option<&u8>,
        absent: R,
        f: impl FnOnce(&u8) -> Result<R, ()>,
    ) -> Result<R, ()> {
        match slot {
            Some(cb) => f(cb),
            None => Ok(absent),
        }
    }
}

angr_macros::inspect_dispatch! {
    Holder =>

    fn reg_write(&self) -> Result<(), ()> {
        let _peek = self.inspect_reg_read.is_some();
        self.with_slot((), |_cb| Ok(()))
    }
}

fn main() {}
