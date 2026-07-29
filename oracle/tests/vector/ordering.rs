use cubecl::prelude::*;
use massively::{
    Executor, MStorage, op::BinaryPredicateOp, vector::sort, vector::sort_by_key, zip2, zip7,
};

use super::common::exec;

struct LessU32;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for LessU32 {
    fn apply(lhs: u32, rhs: u32) -> massively::MFlag {
        massively::flag::from_bool(lhs < rhs)
    }
}

struct LessFirst;

#[cubecl::cube]
impl BinaryPredicateOp<(u32, u32)> for LessFirst {
    fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> massively::MFlag {
        massively::flag::from_bool(lhs.0 < rhs.0)
    }
}

type Seven = (u32, u32, u32, u32, u32, u32, u32);

struct LessSevenFirst;

#[cubecl::cube]
impl BinaryPredicateOp<Seven> for LessSevenFirst {
    fn apply(lhs: Seven, rhs: Seven) -> massively::MFlag {
        massively::flag::from_bool(lhs.0 < rhs.0)
    }
}

const BOUNDARY_LENGTHS: &[usize] = &[
    0, 1, 2, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1_023, 1_024, 1_025,
];

fn keys(len: usize) -> Vec<u32> {
    (0..len)
        .map(|index| ((index * 1_103_515_245 + 12_345) % 37) as u32)
        .collect()
}

#[test]
fn sort_crosses_block_and_merge_tile_boundaries() {
    let exec = exec();
    for &len in BOUNDARY_LENGTHS {
        let input = keys(len);
        let device = exec.to_device(&input);
        let output = sort(&exec, device.slice(..), LessU32).unwrap();

        let mut expected = input;
        expected.sort();
        assert_eq!(exec.to_host(&output).unwrap(), expected, "len={len}");
    }
}

#[test]
fn sort_is_naturally_stable_across_merge_tiles() {
    let exec = exec();
    let len = 2_049;
    let first = keys(len);
    let ordinal: Vec<u32> = (0..len as u32).collect();
    let first_device = exec.to_device(&first);
    let ordinal_device = exec.to_device(&ordinal);
    let output = sort(
        &exec,
        zip2(first_device.slice(..), ordinal_device.slice(..)),
        LessFirst,
    )
    .unwrap();

    let mut expected: Vec<_> = first.into_iter().zip(ordinal).collect();
    expected.sort_by_key(|item| item.0);
    let (first, ordinal) = MStorage::into_columns(output);
    let actual_first = exec.to_host(&first).unwrap();
    let actual_ordinal = exec.to_host(&ordinal).unwrap();
    assert_eq!(
        actual_first
            .into_iter()
            .zip(actual_ordinal)
            .collect::<Vec<_>>(),
        expected,
    );
}

#[test]
fn sort_by_key_preserves_equal_key_value_order() {
    let exec: Executor<_> = exec();
    for &len in BOUNDARY_LENGTHS {
        let input_keys = keys(len);
        let values: Vec<u32> = (0..len as u32).collect();
        let key_device = exec.to_device(&input_keys);
        let value_device = exec.to_device(&values);
        let out_values =
            sort_by_key(&exec, key_device.slice(..), value_device.slice(..), LessU32).unwrap();

        let mut expected: Vec<_> = input_keys.into_iter().zip(values).collect();
        expected.sort_by_key(|item| item.0);
        let actual_values = exec.to_host(&out_values).unwrap();
        assert_eq!(
            actual_values,
            expected
                .into_iter()
                .map(|(_, value)| value)
                .collect::<Vec<_>>(),
            "len={len}",
        );
    }
}

#[test]
fn seven_column_sort_runs_the_global_merge_resource_plan() {
    let exec = exec();
    let len = 513;
    let first = keys(len);
    let inputs: Vec<_> = (0_u32..7)
        .map(|column| {
            if column == 0 {
                exec.to_device(&first)
            } else {
                exec.to_device(
                    &(0..len)
                        .map(|index| index as u32 + column * 10_000)
                        .collect::<Vec<_>>(),
                )
            }
        })
        .collect();
    let outputs = sort(
        &exec,
        zip7(
            inputs[0].slice(..),
            inputs[1].slice(..),
            inputs[2].slice(..),
            inputs[3].slice(..),
            inputs[4].slice(..),
            inputs[5].slice(..),
            inputs[6].slice(..),
        ),
        LessSevenFirst,
    )
    .unwrap();

    let mut expected: Vec<_> = first.into_iter().zip(0_u32..len as u32).collect();
    expected.sort_by_key(|item| item.0);
    let (first, ordinal, _, _, _, _, _) = MStorage::into_columns(outputs);
    let actual_first = exec.to_host(&first).unwrap();
    let actual_ordinal = exec.to_host(&ordinal).unwrap();
    assert_eq!(
        actual_first
            .into_iter()
            .zip(actual_ordinal.into_iter().map(|value| value - 10_000))
            .collect::<Vec<_>>(),
        expected,
    );
}
