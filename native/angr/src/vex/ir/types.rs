/// IR constant values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IRConst {
    U1(bool),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    U128(u128),
    F32(f32),
    F64(f64),
    V128(u128),
    V256([u64; 4]),
}

impl IRConst {
    /// Get the type of this constant.
    pub fn get_type(&self) -> IRType {
        match self {
            IRConst::U1(_) => IRType::I1,
            IRConst::U8(_) => IRType::I8,
            IRConst::U16(_) => IRType::I16,
            IRConst::U32(_) => IRType::I32,
            IRConst::U64(_) => IRType::I64,
            IRConst::U128(_) => IRType::I128,
            IRConst::F32(_) => IRType::F32,
            IRConst::F64(_) => IRType::F64,
            IRConst::V128(_) => IRType::V128,
            IRConst::V256(_) => IRType::V256,
        }
    }

    /// Get the value as u128.
    pub fn as_u128(&self) -> u128 {
        match self {
            IRConst::U1(v) => *v as u128,
            IRConst::U8(v) => *v as u128,
            IRConst::U16(v) => *v as u128,
            IRConst::U32(v) => *v as u128,
            IRConst::U64(v) => *v as u128,
            IRConst::U128(v) => *v,
            IRConst::F32(v) => v.to_bits() as u128,
            IRConst::F64(v) => v.to_bits() as u128,
            IRConst::V128(v) => *v,
            IRConst::V256(v) => {
                // Only return lower 128 bits
                v[0] as u128 | ((v[1] as u128) << 64)
            }
        }
    }
}

/// IR types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IRType {
    /// 1-bit integer (boolean).
    I1,
    /// 8-bit integer.
    I8,
    /// 16-bit integer.
    I16,
    /// 32-bit integer.
    I32,
    /// 64-bit integer.
    I64,
    /// 128-bit integer.
    I128,
    /// 32-bit float (IEEE 754).
    F32,
    /// 64-bit float (IEEE 754).
    F64,
    /// 80-bit float (x87).
    F80,
    /// 16-bit float (IEEE 754).
    F16,
    /// 128-bit vector.
    V128,
    /// 256-bit vector.
    V256,
}

impl IRType {
    /// Get the size in bits.
    pub fn bits(&self) -> u32 {
        match self {
            IRType::I1 => 1,
            IRType::I8 => 8,
            IRType::I16 => 16,
            IRType::I32 => 32,
            IRType::I64 => 64,
            IRType::I128 => 128,
            IRType::F16 => 16,
            IRType::F32 => 32,
            IRType::F64 => 64,
            IRType::F80 => 80,
            IRType::V128 => 128,
            IRType::V256 => 256,
        }
    }

    /// Get the size in bytes (rounded up).
    pub fn bytes(&self) -> u32 {
        self.bits().div_ceil(8)
    }
}
