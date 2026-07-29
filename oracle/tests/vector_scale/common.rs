use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{
    Executor, MFlag, MIndex, MStorage, op::BinaryPredicateOp, op::PredicateOp, op::ReductionOp,
    op::UnaryOp, vector,
};
use oracle::op;

const DEFAULT_SCALE_LEN: usize = 10_000_000;

pub struct MaxU32;
pub struct NonZero;
pub struct IdentityU32;
pub struct EqualU32;
pub struct LessU32;

#[cubecl::cube]
impl ReductionOp<u32> for MaxU32 {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs.max(rhs)
    }
}

impl op::ReductionOp<u32> for MaxU32 {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs.max(rhs)
    }
}

#[cubecl::cube]
impl PredicateOp<u32> for NonZero {
    fn apply(input: u32) -> MFlag {
        massively::flag::from_bool(input != 0u32)
    }
}

impl op::PredicateOp<u32> for NonZero {
    fn apply(input: u32) -> bool {
        input != 0
    }
}

#[cubecl::cube]
impl UnaryOp<u32> for IdentityU32 {
    type Output = u32;

    fn apply(input: u32) -> u32 {
        input
    }
}

impl op::UnaryOp<u32> for IdentityU32 {
    type Output = u32;

    fn apply(input: u32) -> u32 {
        input
    }
}

#[cubecl::cube]
impl BinaryPredicateOp<u32> for EqualU32 {
    fn apply(lhs: u32, rhs: u32) -> MFlag {
        massively::flag::from_bool(lhs == rhs)
    }
}

impl op::BinaryPredicateOp<u32> for EqualU32 {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs == rhs
    }
}

#[cubecl::cube]
impl BinaryPredicateOp<u32> for LessU32 {
    fn apply(lhs: u32, rhs: u32) -> MFlag {
        massively::flag::from_bool(lhs < rhs)
    }
}

impl op::BinaryPredicateOp<u32> for LessU32 {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs < rhs
    }
}

pub fn as_stencil<Input>(input: Input) -> Input {
    input
}

pub fn as_indices<Input>(input: Input) -> Input {
    input
}

pub fn expected_flag(value: bool) -> MFlag {
    massively::flag::from_bool(value)
}

pub fn exec() -> Executor<WgpuRuntime> {
    Executor::new(WgpuDevice::DefaultDevice)
}

pub fn logical_len<Storage>(exec: &Executor<WgpuRuntime>, storage: &Storage) -> usize
where
    Storage: MStorage<WgpuRuntime>,
{
    vector::count(exec, storage.slice(..)).unwrap() as usize
}

pub fn read_value<T>(_exec: &Executor<WgpuRuntime>, value: T) -> T {
    value
}

pub fn read_optional_index(_exec: &Executor<WgpuRuntime>, value: Option<MIndex>) -> Option<usize> {
    value.map(|index| index as usize)
}

pub fn read_optional_index_pair(
    _exec: &Executor<WgpuRuntime>,
    value: Option<(MIndex, MIndex)>,
) -> Option<(usize, usize)> {
    value.map(|(first, second)| (first as usize, second as usize))
}

pub fn lazify<Input>(input: Input) -> Input
where
    Input: massively::MIter<WgpuRuntime>,
{
    input
}

pub fn scale_len() -> usize {
    std::env::var("MASSIVELY_SCALE_LEN")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_SCALE_LEN)
}

pub fn scale_input() -> Vec<u32> {
    (0..scale_len())
        .map(|index| ((index as u64 * 1_103_515_245 + 12_345) % 1_000_003) as u32)
        .collect()
}

pub fn scale_other() -> Vec<u32> {
    (0..scale_len())
        .map(|index| ((index as u64 * 1_664_525 + 1_013_904_223) % 1_000_003) as u32)
        .collect()
}

pub fn flags_for(input: &[u32]) -> Vec<u32> {
    input
        .iter()
        .map(|value| u32::from(value & 1 != 0))
        .collect()
}

pub fn indices_for(len: usize) -> Vec<u32> {
    (0..len).rev().map(|index| index as u32).collect()
}
