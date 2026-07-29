//! Lazy deterministic GPU-side random value generation.

use cubecl::prelude::*;

use crate::core::iter::Zip;
use crate::{Error, MIndex, lazy};

type RandomInput<T> = Zip<
    Zip<
        Zip<crate::core::read::Counting, crate::core::read::Constant<T>>,
        crate::core::read::Constant<T>,
    >,
    crate::core::read::Constant<u32>,
>;

fn pcg_hash32_host(input: u32) -> u32 {
    let state = input.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

fn seed_key(seed: u64) -> u32 {
    let low = seed as u32;
    let high = (seed >> 32) as u32;
    pcg_hash32_host(low ^ pcg_hash32_host(high))
}

fn validate_inclusive_range<T: PartialOrd>(min: T, max: T) -> Result<(), Error> {
    if min <= max {
        Ok(())
    } else {
        Err(Error::Launch {
            message: "random distribution range must satisfy min <= max".to_string(),
        })
    }
}

macro_rules! uniform_stream {
    ($stream:ident, $item:ty, $op:ident, $constructor:ident, $doc:literal, $example:literal) => {
        #[doc = $doc]
        #[derive(Clone, Copy, Debug)]
        pub struct $stream {
            min: $item,
            max: $item,
            key: u32,
        }

        #[doc = $doc]
        #[doc = ""]
        #[doc = "The same seed and logical index always produce the same value."]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = $example]
        pub fn $constructor(min: $item, max: $item, seed: u64) -> Result<$stream, Error> {
            validate_inclusive_range(min, max)?;
            Ok($stream {
                min,
                max,
                key: seed_key(seed),
            })
        }

        impl $stream {
            /// Limits this generator to `len` logical items.
            pub fn take(self, len: MIndex) -> lazy::Taken<Self> {
                lazy::Taken::new(self, len as usize)
            }
        }

        impl crate::core::read::TakenSource for $stream {
            type Read = crate::core::read::Transform<RandomInput<$item>, $op>;

            fn lower(&self, offset: usize, len: usize) -> Self::Read {
                crate::core::read::Transform::new(
                    Zip::new(
                        Zip::new(
                            Zip::new(
                                crate::core::read::Counting::new(
                                    u32::try_from(offset)
                                        .expect("random stream offset exceeds device u32 range"),
                                    len,
                                ),
                                crate::core::read::Constant::new(self.min, len),
                            ),
                            crate::core::read::Constant::new(self.max, len),
                        ),
                        crate::core::read::Constant::new(self.key, len),
                    ),
                    $op,
                )
            }
        }
    };
}

uniform_stream!(
    UniformU32,
    u32,
    UniformU32Op,
    uniform_u32,
    "A deterministic uniform `u32` stream over an inclusive range.",
    r#"```
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op, util::random, vector::map};

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let values = random::uniform_u32(10, 20, 123).unwrap().take(8);
let output = map(&exec, values, op::Identity).unwrap();

let values = exec.to_host(&output).unwrap();
assert!(values.iter().all(|value| (10..=20).contains(value)));
```"#
);
uniform_stream!(
    UniformU64,
    u64,
    UniformU64Op,
    uniform_u64,
    "A deterministic uniform `u64` stream over an inclusive range.",
    r#"```
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op, util::random, vector::map};

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let values = random::uniform_u64(100, 200, 123).unwrap().take(8);
let output = map(&exec, values, op::Identity).unwrap();

let values = exec.to_host(&output).unwrap();
assert!(values.iter().all(|value| (100..=200).contains(value)));
```"#
);
uniform_stream!(
    UniformF32,
    f32,
    UniformF32Op,
    uniform_f32,
    "A deterministic uniform `f32` stream over a bounded range.",
    r#"```
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op, util::random, vector::map};

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let values = random::uniform_f32(-1.0, 1.0, 123).unwrap().take(8);
let output = map(&exec, values, op::Identity).unwrap();

let values = exec.to_host(&output).unwrap();
assert!(values.iter().all(|value| (-1.0..=1.0).contains(value)));
```"#
);
uniform_stream!(
    UniformF64,
    f64,
    UniformF64Op,
    uniform_f64,
    "A deterministic uniform `f64` stream over a bounded range.",
    r#"```
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op, util::random, vector::map};

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let values = random::uniform_f64(-1.0, 1.0, 123).unwrap().take(8);
let output = map(&exec, values, op::Identity).unwrap();

let values = exec.to_host(&output).unwrap();
assert!(values.iter().all(|value| (-1.0..=1.0).contains(value)));
```"#
);

macro_rules! normal_stream {
    ($stream:ident, $item:ty, $op:ident, $constructor:ident, $doc:literal, $example:literal) => {
        #[doc = $doc]
        #[derive(Clone, Copy, Debug)]
        pub struct $stream {
            mean: $item,
            stddev: $item,
            key: u32,
        }

        #[doc = $doc]
        #[doc = ""]
        #[doc = "The same seed and logical index always produce the same value."]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = $example]
        pub fn $constructor(mean: $item, stddev: $item, seed: u64) -> $stream {
            $stream {
                mean,
                stddev,
                key: seed_key(seed),
            }
        }

        impl $stream {
            /// Limits this generator to `len` logical items.
            pub fn take(self, len: MIndex) -> lazy::Taken<Self> {
                lazy::Taken::new(self, len as usize)
            }
        }

        impl crate::core::read::TakenSource for $stream {
            type Read = crate::core::read::Transform<RandomInput<$item>, $op>;

            fn lower(&self, offset: usize, len: usize) -> Self::Read {
                crate::core::read::Transform::new(
                    Zip::new(
                        Zip::new(
                            Zip::new(
                                crate::core::read::Counting::new(
                                    u32::try_from(offset)
                                        .expect("random stream offset exceeds device u32 range"),
                                    len,
                                ),
                                crate::core::read::Constant::new(self.mean, len),
                            ),
                            crate::core::read::Constant::new(self.stddev, len),
                        ),
                        crate::core::read::Constant::new(self.key, len),
                    ),
                    $op,
                )
            }
        }
    };
}

normal_stream!(
    NormalF32,
    f32,
    NormalF32Op,
    normal_f32,
    "A deterministic normally distributed `f32` stream.",
    r#"```
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op, util::random, vector::map};

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let values = random::normal_f32(0.0, 1.0, 123).take(8);
let output = map(&exec, values, op::Identity).unwrap();

assert!(exec.to_host(&output).unwrap().iter().all(|value| value.is_finite()));
```"#
);
normal_stream!(
    NormalF64,
    f64,
    NormalF64Op,
    normal_f64,
    "A deterministic normally distributed `f64` stream.",
    r#"```
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op, util::random, vector::map};

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let values = random::normal_f64(0.0, 1.0, 123).take(8);
let output = map(&exec, values, op::Identity).unwrap();

assert!(exec.to_host(&output).unwrap().iter().all(|value| value.is_finite()));
```"#
);

#[doc(hidden)]
pub struct UniformU32Op;
#[doc(hidden)]
pub struct UniformU64Op;
#[doc(hidden)]
pub struct UniformF32Op;
#[doc(hidden)]
pub struct UniformF64Op;
#[doc(hidden)]
pub struct NormalF32Op;
#[doc(hidden)]
pub struct NormalF64Op;

type RandomU32Args = (u32, u32, u32, u32);
type RandomU64Args = (u32, u64, u64, u32);
type RandomF32Args = (u32, f32, f32, u32);
type RandomF64Args = (u32, f64, f64, u32);

#[cubecl::cube]
impl crate::op::UnaryOp<RandomU32Args> for UniformU32Op {
    type Output = u32;

    fn apply(input: RandomU32Args) -> u32 {
        uniform_u32_value(input.1, input.2, input.3, input.0)
    }
}

#[cubecl::cube]
impl crate::op::UnaryOp<RandomU64Args> for UniformU64Op {
    type Output = u64;

    fn apply(input: RandomU64Args) -> u64 {
        uniform_u64_value(input.1, input.2, input.3, input.0)
    }
}

#[cubecl::cube]
impl crate::op::UnaryOp<RandomF32Args> for UniformF32Op {
    type Output = f32;

    fn apply(input: RandomF32Args) -> f32 {
        uniform_f32_value(input.1, input.2, input.3, input.0)
    }
}

#[cubecl::cube]
impl crate::op::UnaryOp<RandomF64Args> for UniformF64Op {
    type Output = f64;

    fn apply(input: RandomF64Args) -> f64 {
        uniform_f64_value(input.1, input.2, input.3, input.0)
    }
}

#[cubecl::cube]
impl crate::op::UnaryOp<RandomF32Args> for NormalF32Op {
    type Output = f32;

    fn apply(input: RandomF32Args) -> f32 {
        normal_f32_value(input.1, input.2, input.3, input.0)
    }
}

#[cubecl::cube]
impl crate::op::UnaryOp<RandomF64Args> for NormalF64Op {
    type Output = f64;

    fn apply(input: RandomF64Args) -> f64 {
        normal_f64_value(input.1, input.2, input.3, input.0)
    }
}

#[cubecl::cube]
fn pcg_hash32(input: u32) -> u32 {
    let state = input * 747_796_405u32 + 2_891_336_453u32;
    let word = ((state >> ((state >> 28u32) + 4u32)) ^ state) * 277_803_737u32;
    (word >> 22u32) ^ word
}

#[cubecl::cube]
fn random_u32_at(key: u32, index: u32, stream: u32) -> u32 {
    pcg_hash32(index ^ key ^ stream)
}

#[cubecl::cube]
fn random_u64_at(key: u32, index: u32, stream: u32) -> u64 {
    let low = random_u32_at(key, index, stream);
    let high = random_u32_at(key, index, stream ^ 0x9e37_79b9u32);
    (low as u64) | ((high as u64) << 32u64)
}

#[cubecl::cube]
fn unit_f32(key: u32, index: u32, stream: u32) -> f32 {
    ((random_u32_at(key, index, stream) >> 8u32) as f32) * 0.000_000_059_604_65_f32
}

#[cubecl::cube]
fn unit_f64(key: u32, index: u32, stream: u32) -> f64 {
    ((random_u64_at(key, index, stream) >> 11u64) as f64) * 0.00000000000000011102230246251567f64
}

#[cubecl::cube]
fn open_unit_f32(key: u32, index: u32, stream: u32) -> f32 {
    (((random_u32_at(key, index, stream) >> 8u32) as f32) + 0.5f32) * 0.000_000_059_604_65_f32
}

#[cubecl::cube]
fn uniform_f32_value(min: f32, max: f32, key: u32, index: u32) -> f32 {
    min + unit_f32(key, index, 0u32) * (max - min)
}

#[cubecl::cube]
fn uniform_f64_value(min: f64, max: f64, key: u32, index: u32) -> f64 {
    min + unit_f64(key, index, 0u32) * (max - min)
}

#[cubecl::cube]
fn uniform_u32_value(min: u32, max: u32, key: u32, index: u32) -> u32 {
    let span = max - min;
    let value = random_u32_at(key, index, 0u32);
    if span == 0xffff_ffffu32 {
        value
    } else {
        min + value % (span + 1u32)
    }
}

#[cubecl::cube]
fn uniform_u64_value(min: u64, max: u64, key: u32, index: u32) -> u64 {
    let span = max - min;
    let value = random_u64_at(key, index, 0u32);
    if span == 0xffff_ffff_ffff_ffffu64 {
        value
    } else {
        min + value % (span + 1u64)
    }
}

#[cubecl::cube]
fn normal_f32_value(mean: f32, stddev: f32, key: u32, index: u32) -> f32 {
    let u1 = open_unit_f32(key, index, 0u32);
    let u2 = open_unit_f32(key, index, 0x9e37_79b9u32);
    let radius = (-2.0f32 * u1.ln()).sqrt();
    let angle = core::f32::consts::TAU * u2;
    mean + stddev * radius * angle.cos()
}

#[cubecl::cube]
fn normal_f64_value(mean: f64, stddev: f64, key: u32, index: u32) -> f64 {
    let u1 = open_unit_f32(key, index, 0u32);
    let u2 = open_unit_f32(key, index, 0x9e37_79b9u32);
    let radius = (-2.0f32 * u1.ln()).sqrt();
    let angle = core::f32::consts::TAU * u2;
    mean + stddev * ((radius * angle.cos()) as f64)
}
