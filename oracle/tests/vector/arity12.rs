use cubecl::prelude::*;
use massively::{op::*, *};
use oracle::{op, vector as reference};
use proptest::prelude::*;

use super::common::*;

type Twelve = (u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32);

struct ExpandTwelve;
struct MaxTwelve;

#[cubecl::cube]
impl UnaryOp<u32> for ExpandTwelve {
    type Output = Twelve;

    fn apply(value: u32) -> Twelve {
        (
            value,
            value + 1u32,
            value + 2u32,
            value + 3u32,
            value + 4u32,
            value + 5u32,
            value + 6u32,
            value + 7u32,
            value + 8u32,
            value + 9u32,
            value + 10u32,
            value + 11u32,
        )
    }
}

#[cubecl::cube]
impl ReductionOp<Twelve> for MaxTwelve {
    fn apply(lhs: Twelve, rhs: Twelve) -> Twelve {
        let (l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11) = lhs;
        let (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11) = rhs;
        (
            l0.max(r0),
            l1.max(r1),
            l2.max(r2),
            l3.max(r3),
            l4.max(r4),
            l5.max(r5),
            l6.max(r6),
            l7.max(r7),
            l8.max(r8),
            l9.max(r9),
            l10.max(r10),
            l11.max(r11),
        )
    }
}

impl op::ReductionOp<Twelve> for MaxTwelve {
    fn apply(lhs: Twelve, rhs: Twelve) -> Twelve {
        let (l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11) = lhs;
        let (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11) = rhs;
        (
            l0.max(r0),
            l1.max(r1),
            l2.max(r2),
            l3.max(r3),
            l4.max(r4),
            l5.max(r5),
            l6.max(r6),
            l7.max(r7),
            l8.max(r8),
            l9.max(r9),
            l10.max(r10),
            l11.max(r11),
        )
    }
}

macro_rules! zip12_columns {
    ($columns:expr) => {
        zip12(
            $columns[0].slice(..),
            $columns[1].slice(..),
            $columns[2].slice(..),
            $columns[3].slice(..),
            $columns[4].slice(..),
            $columns[5].slice(..),
            $columns[6].slice(..),
            $columns[7].slice(..),
            $columns[8].slice(..),
            $columns[9].slice(..),
            $columns[10].slice(..),
            $columns[11].slice(..),
        )
    };
}

fn columns(seed: &[u32]) -> [Vec<u32>; 12] {
    core::array::from_fn(|column| {
        seed.iter()
            .map(|value| value.wrapping_add(column as u32))
            .collect()
    })
}

fn rows(columns: &[Vec<u32>; 12]) -> Vec<Twelve> {
    (0..columns[0].len())
        .map(|index| {
            (
                columns[0][index],
                columns[1][index],
                columns[2][index],
                columns[3][index],
                columns[4][index],
                columns[5][index],
                columns[6][index],
                columns[7][index],
                columns[8][index],
                columns[9][index],
                columns[10][index],
                columns[11][index],
            )
        })
        .collect()
}

fn row_columns(rows: &[Twelve]) -> [Vec<u32>; 12] {
    core::array::from_fn(|column| {
        rows.iter()
            .copied()
            .map(|row| match column {
                0 => row.0,
                1 => row.1,
                2 => row.2,
                3 => row.3,
                4 => row.4,
                5 => row.5,
                6 => row.6,
                7 => row.7,
                8 => row.8,
                9 => row.9,
                10 => row.10,
                11 => row.11,
                _ => unreachable!(),
            })
            .collect()
    })
}

macro_rules! zip_prefix_columns {
    (8, $columns:expr) => {
        zip8(
            $columns[0].slice(..),
            $columns[1].slice(..),
            $columns[2].slice(..),
            $columns[3].slice(..),
            $columns[4].slice(..),
            $columns[5].slice(..),
            $columns[6].slice(..),
            $columns[7].slice(..),
        )
    };
    (9, $columns:expr) => {
        zip9(
            $columns[0].slice(..),
            $columns[1].slice(..),
            $columns[2].slice(..),
            $columns[3].slice(..),
            $columns[4].slice(..),
            $columns[5].slice(..),
            $columns[6].slice(..),
            $columns[7].slice(..),
            $columns[8].slice(..),
        )
    };
    (10, $columns:expr) => {
        zip10(
            $columns[0].slice(..),
            $columns[1].slice(..),
            $columns[2].slice(..),
            $columns[3].slice(..),
            $columns[4].slice(..),
            $columns[5].slice(..),
            $columns[6].slice(..),
            $columns[7].slice(..),
            $columns[8].slice(..),
            $columns[9].slice(..),
        )
    };
    (11, $columns:expr) => {
        zip11(
            $columns[0].slice(..),
            $columns[1].slice(..),
            $columns[2].slice(..),
            $columns[3].slice(..),
            $columns[4].slice(..),
            $columns[5].slice(..),
            $columns[6].slice(..),
            $columns[7].slice(..),
            $columns[8].slice(..),
            $columns[9].slice(..),
            $columns[10].slice(..),
        )
    };
}

macro_rules! assert_prefix {
    (@check $exec:expr, $expected:expr, [$($output:ident),+]) => {{
        let outputs = [$($output),+];
        for (actual, expected) in outputs.iter().zip($expected.iter()) {
            assert_eq!($exec.to_host(actual).unwrap(), expected.clone());
        }
    }};
    (8, $exec:expr, $output:expr, $expected:expr) => {{
        let (o0, o1, o2, o3, o4, o5, o6, o7) = MStorage::into_columns($output);
        assert_prefix!(@check $exec, $expected, [o0, o1, o2, o3, o4, o5, o6, o7]);
    }};
    (9, $exec:expr, $output:expr, $expected:expr) => {{
        let (o0, o1, o2, o3, o4, o5, o6, o7, o8) = MStorage::into_columns($output);
        assert_prefix!(@check $exec, $expected, [o0, o1, o2, o3, o4, o5, o6, o7, o8]);
    }};
    (10, $exec:expr, $output:expr, $expected:expr) => {{
        let (o0, o1, o2, o3, o4, o5, o6, o7, o8, o9) = MStorage::into_columns($output);
        assert_prefix!(@check $exec, $expected, [o0, o1, o2, o3, o4, o5, o6, o7, o8, o9]);
    }};
    (11, $exec:expr, $output:expr, $expected:expr) => {{
        let (o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10) = MStorage::into_columns($output);
        assert_prefix!(@check $exec, $expected, [o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10]);
    }};
}

macro_rules! assert_twelve {
    ($exec:expr, $output:expr, $expected:expr) => {{
        let (o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11) = MStorage::into_columns($output);
        let outputs = [o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11];
        for (actual, expected) in outputs.iter().zip($expected.iter()) {
            assert_eq!($exec.to_host(actual).unwrap(), expected.clone());
        }
    }};
}

#[test]
fn zip8_through_zip11_match_their_visible_columns() {
    let exec = exec();
    let expected = columns(&[3, 7, 11, 19]);
    let device: Vec<_> = expected
        .iter()
        .map(|column| exec.to_device(column))
        .collect();

    let output =
        massively::vector::map(&exec, lazify(zip_prefix_columns!(8, device)), Identity).unwrap();
    assert_prefix!(8, exec, output, expected);

    let output =
        massively::vector::map(&exec, lazify(zip_prefix_columns!(9, device)), Identity).unwrap();
    assert_prefix!(9, exec, output, expected);

    let output =
        massively::vector::map(&exec, lazify(zip_prefix_columns!(10, device)), Identity).unwrap();
    assert_prefix!(10, exec, output, expected);

    let output =
        massively::vector::map(&exec, lazify(zip_prefix_columns!(11, device)), Identity).unwrap();
    assert_prefix!(11, exec, output, expected);
}

#[test]
fn zip12_segmented_scans_cross_block_boundaries() {
    let exec = exec();
    let seed: Vec<u32> = (0..769).map(|index| (index * 37 % 101) as u32).collect();
    let expected_columns = columns(&seed);
    let input_rows = rows(&expected_columns);
    let device: Vec<_> = expected_columns
        .iter()
        .map(|column| exec.to_device(column))
        .collect();
    let keys: Vec<u32> = (0..seed.len()).map(|index| (index / 300) as u32).collect();
    let device_keys = exec.to_device(&keys);
    let zero = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

    let inclusive = massively::vector::inclusive_scan_by_key(
        &exec,
        device_keys.slice(..),
        lazify(zip12_columns!(device)),
        Equal,
        MaxTwelve,
    )
    .unwrap();
    let expected = reference::inclusive_scan_by_key(&keys, &input_rows, Equal, MaxTwelve);
    assert_twelve!(exec, inclusive, row_columns(&expected));

    let exclusive = massively::vector::exclusive_scan_by_key(
        &exec,
        device_keys.slice(..),
        lazify(zip12_columns!(device)),
        Equal,
        zero,
        MaxTwelve,
    )
    .unwrap();
    let expected = reference::exclusive_scan_by_key(&keys, &input_rows, Equal, zero, MaxTwelve);
    assert_twelve!(exec, exclusive, row_columns(&expected));
}

proptest! {
    #![proptest_config(ProptestConfig { cases: CASES, .. ProptestConfig::default() })]

    #[test]
    fn fixed_twelve_output_matches_oracle(seed in oracle_vec(0_u32..100)) {
        let exec = exec();
        let input = exec.to_device(&seed);
        let output = massively::vector::map(&exec, lazify(input.slice(..)), ExpandTwelve).unwrap();
        let expected = columns(&seed);
        assert_twelve!(exec, output, expected);
    }

    #[test]
    fn zip12_and_internal_arity13_match_oracle(seed in oracle_vec(0_u32..100)) {
        let exec = exec();
        let expected_columns = columns(&seed);
        let device: Vec<_> = expected_columns
            .iter()
            .map(|column| exec.to_device(column))
            .collect();
        let input_rows = rows(&expected_columns);

        let copied = massively::vector::map(
            &exec,
            lazify(zip12_columns!(device)),
            Identity,
        ).unwrap();
        assert_twelve!(exec, copied, expected_columns);

        // Twelve value leaves plus the permutation index consume the internal
        // maximum of thirteen physical read slots.
        let permuted = lazy::identity(lazy::permute(
            zip12_columns!(device),
            lazy::counting(0).take(seed.len() as massively::MIndex),
        ));
        let copied = massively::vector::map(&exec, permuted, Identity).unwrap();
        assert_twelve!(exec, copied, expected_columns);

        let zero = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        prop_assert_eq!(
            read_value(
                &exec,
                massively::vector::reduce(
                    &exec,
                    lazify(zip12_columns!(device)),
                    zero,
                    MaxTwelve,
                )
                .unwrap(),
            ),
            reference::reduce(&input_rows, zero, MaxTwelve),
        );

        let scanned = massively::vector::inclusive_scan(
            &exec,
            lazify(zip12_columns!(device)),
            MaxTwelve,
        ).unwrap();
        let expected_rows = reference::inclusive_scan(&input_rows, MaxTwelve);
        assert_twelve!(exec, scanned, row_columns(&expected_rows));

        let scanned = massively::vector::exclusive_scan(
            &exec,
            lazify(zip12_columns!(device)),
            zero,
            MaxTwelve,
        ).unwrap();
        let expected_rows = reference::exclusive_scan(&input_rows, zero, MaxTwelve);
        assert_twelve!(exec, scanned, row_columns(&expected_rows));
    }
}
