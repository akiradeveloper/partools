use cubecl::prelude::*;
use massively::op::Identity;
use massively::{op::*, *};
use oracle::{op, vector as reference};
use proptest::prelude::*;

use super::common::*;

type Two = (u32, u32);
type Three = (u32, u32, u32);
type Four = (u32, u32, u32, u32);
type Five = (u32, u32, u32, u32, u32);
type Six = (u32, u32, u32, u32, u32, u32);
type Seven = (u32, u32, u32, u32, u32, u32, u32);

struct MaxItem;
struct EvenItem;
struct EqualItem;
struct LessItem;

macro_rules! splat {
    (1, $value:expr) => {
        $value
    };
    (2, $value:expr) => {
        ($value, $value)
    };
    (3, $value:expr) => {
        ($value, $value, $value)
    };
    (4, $value:expr) => {
        ($value, $value, $value, $value)
    };
    (5, $value:expr) => {
        ($value, $value, $value, $value, $value)
    };
    (6, $value:expr) => {
        ($value, $value, $value, $value, $value, $value)
    };
    (7, $value:expr) => {
        ($value, $value, $value, $value, $value, $value, $value)
    };
}

macro_rules! impl_item_ops {
    (
        $item:ty;
        reduce |$reduce_lhs:ident, $reduce_rhs:ident| $reduce:expr;
        first |$input:ident| $first:expr;
        pair |$pair_lhs:ident, $pair_rhs:ident| $first_lhs:expr, $first_rhs:expr
    ) => {
        #[cubecl::cube]
        impl ReductionOp<$item> for MaxItem {
            fn apply($reduce_lhs: $item, $reduce_rhs: $item) -> $item {
                $reduce
            }
        }

        impl op::ReductionOp<$item> for MaxItem {
            fn apply($reduce_lhs: $item, $reduce_rhs: $item) -> $item {
                $reduce
            }
        }

        #[cubecl::cube]
        impl PredicateOp<$item> for EvenItem {
            fn apply($input: $item) -> MFlag {
                massively::flag::from_bool($first % 2u32 == 0u32)
            }
        }

        impl op::PredicateOp<$item> for EvenItem {
            fn apply($input: $item) -> bool {
                $first % 2 == 0
            }
        }

        #[cubecl::cube]
        impl BinaryPredicateOp<$item> for EqualItem {
            fn apply($pair_lhs: $item, $pair_rhs: $item) -> MFlag {
                massively::flag::from_bool($first_lhs == $first_rhs)
            }
        }

        impl op::BinaryPredicateOp<$item> for EqualItem {
            fn apply($pair_lhs: $item, $pair_rhs: $item) -> bool {
                $first_lhs == $first_rhs
            }
        }

        #[cubecl::cube]
        impl BinaryPredicateOp<$item> for LessItem {
            fn apply($pair_lhs: $item, $pair_rhs: $item) -> MFlag {
                massively::flag::from_bool($first_lhs < $first_rhs)
            }
        }

        impl op::BinaryPredicateOp<$item> for LessItem {
            fn apply($pair_lhs: $item, $pair_rhs: $item) -> bool {
                $first_lhs < $first_rhs
            }
        }
    };
}

impl_item_ops!(u32;
    reduce |lhs, rhs| lhs.max(rhs);
    first |input| input;
    pair |lhs, rhs| lhs, rhs
);
impl_item_ops!(Two;
    reduce |lhs, rhs| (lhs.0.max(rhs.0), lhs.1.max(rhs.1));
    first |input| input.0;
    pair |lhs, rhs| lhs.0, rhs.0
);
impl_item_ops!(Three;
    reduce |lhs, rhs| (lhs.0.max(rhs.0), lhs.1.max(rhs.1), lhs.2.max(rhs.2));
    first |input| input.0;
    pair |lhs, rhs| lhs.0, rhs.0
);
impl_item_ops!(Four;
    reduce |lhs, rhs| (lhs.0.max(rhs.0), lhs.1.max(rhs.1), lhs.2.max(rhs.2), lhs.3.max(rhs.3));
    first |input| input.0;
    pair |lhs, rhs| lhs.0, rhs.0
);
impl_item_ops!(Five;
    reduce |lhs, rhs| (lhs.0.max(rhs.0), lhs.1.max(rhs.1), lhs.2.max(rhs.2), lhs.3.max(rhs.3), lhs.4.max(rhs.4));
    first |input| input.0;
    pair |lhs, rhs| lhs.0, rhs.0
);
impl_item_ops!(Six;
    reduce |lhs, rhs| (lhs.0.max(rhs.0), lhs.1.max(rhs.1), lhs.2.max(rhs.2), lhs.3.max(rhs.3), lhs.4.max(rhs.4), lhs.5.max(rhs.5));
    first |input| input.0;
    pair |lhs, rhs| lhs.0, rhs.0
);
impl_item_ops!(Seven;
    reduce |lhs, rhs| (lhs.0.max(rhs.0), lhs.1.max(rhs.1), lhs.2.max(rhs.2), lhs.3.max(rhs.3), lhs.4.max(rhs.4), lhs.5.max(rhs.5), lhs.6.max(rhs.6));
    first |input| input.0;
    pair |lhs, rhs| lhs.0, rhs.0
);

fn seven_columns(seed: &[u32]) -> [Vec<u32>; 7] {
    core::array::from_fn(|column| {
        seed.iter()
            .map(|value| value.wrapping_add(column as u32 * 17))
            .collect()
    })
}

macro_rules! rows {
    (1, $columns:expr) => {
        $columns[0].clone()
    };
    (2, $columns:expr) => {
        (0..$columns[0].len())
            .map(|i| ($columns[0][i], $columns[1][i]))
            .collect::<Vec<Two>>()
    };
    (3, $columns:expr) => {
        (0..$columns[0].len())
            .map(|i| ($columns[0][i], $columns[1][i], $columns[2][i]))
            .collect::<Vec<Three>>()
    };
    (4, $columns:expr) => {
        (0..$columns[0].len())
            .map(|i| {
                (
                    $columns[0][i],
                    $columns[1][i],
                    $columns[2][i],
                    $columns[3][i],
                )
            })
            .collect::<Vec<Four>>()
    };
    (5, $columns:expr) => {
        (0..$columns[0].len())
            .map(|i| {
                (
                    $columns[0][i],
                    $columns[1][i],
                    $columns[2][i],
                    $columns[3][i],
                    $columns[4][i],
                )
            })
            .collect::<Vec<Five>>()
    };
    (6, $columns:expr) => {
        (0..$columns[0].len())
            .map(|i| {
                (
                    $columns[0][i],
                    $columns[1][i],
                    $columns[2][i],
                    $columns[3][i],
                    $columns[4][i],
                    $columns[5][i],
                )
            })
            .collect::<Vec<Six>>()
    };
    (7, $columns:expr) => {
        (0..$columns[0].len())
            .map(|i| {
                (
                    $columns[0][i],
                    $columns[1][i],
                    $columns[2][i],
                    $columns[3][i],
                    $columns[4][i],
                    $columns[5][i],
                    $columns[6][i],
                )
            })
            .collect::<Vec<Seven>>()
    };
}

macro_rules! raw_input_expr {
    (1, $device:expr) => {
        $device[0].slice(..)
    };
    (2, $device:expr) => {
        zip2($device[0].slice(..), $device[1].slice(..))
    };
    (3, $device:expr) => {
        zip3(
            $device[0].slice(..),
            $device[1].slice(..),
            $device[2].slice(..),
        )
    };
    (4, $device:expr) => {
        zip4(
            $device[0].slice(..),
            $device[1].slice(..),
            $device[2].slice(..),
            $device[3].slice(..),
        )
    };
    (5, $device:expr) => {
        zip5(
            $device[0].slice(..),
            $device[1].slice(..),
            $device[2].slice(..),
            $device[3].slice(..),
            $device[4].slice(..),
        )
    };
    (6, $device:expr) => {
        zip6(
            $device[0].slice(..),
            $device[1].slice(..),
            $device[2].slice(..),
            $device[3].slice(..),
            $device[4].slice(..),
            $device[5].slice(..),
        )
    };
    (7, $device:expr) => {
        zip7(
            $device[0].slice(..),
            $device[1].slice(..),
            $device[2].slice(..),
            $device[3].slice(..),
            $device[4].slice(..),
            $device[5].slice(..),
            $device[6].slice(..),
        )
    };
}

macro_rules! input_expr {
    ($arity:tt, $device:ident) => {
        lazify(raw_input_expr!($arity, $device))
    };
}

macro_rules! output_expr {
    (1, $output:expr) => {
        $output[0].slice_mut(..)
    };
    (2, $output:expr) => {
        zip2($output[0].slice_mut(..), $output[1].slice_mut(..))
    };
    (3, $output:expr) => {
        zip3(
            $output[0].slice_mut(..),
            $output[1].slice_mut(..),
            $output[2].slice_mut(..),
        )
    };
    (4, $output:expr) => {
        zip4(
            $output[0].slice_mut(..),
            $output[1].slice_mut(..),
            $output[2].slice_mut(..),
            $output[3].slice_mut(..),
        )
    };
    (5, $output:expr) => {
        zip5(
            $output[0].slice_mut(..),
            $output[1].slice_mut(..),
            $output[2].slice_mut(..),
            $output[3].slice_mut(..),
            $output[4].slice_mut(..),
        )
    };
    (6, $output:expr) => {
        zip6(
            $output[0].slice_mut(..),
            $output[1].slice_mut(..),
            $output[2].slice_mut(..),
            $output[3].slice_mut(..),
            $output[4].slice_mut(..),
            $output[5].slice_mut(..),
        )
    };
    (7, $output:expr) => {
        zip7(
            $output[0].slice_mut(..),
            $output[1].slice_mut(..),
            $output[2].slice_mut(..),
            $output[3].slice_mut(..),
            $output[4].slice_mut(..),
            $output[5].slice_mut(..),
            $output[6].slice_mut(..),
        )
    };
}

macro_rules! row_column {
    (1, $rows:expr, 0) => {
        $rows.iter().copied().collect::<Vec<u32>>()
    };
    (2, $rows:expr, 0) => {
        $rows.iter().map(|row| row.0).collect::<Vec<u32>>()
    };
    (2, $rows:expr, 1) => {
        $rows.iter().map(|row| row.1).collect::<Vec<u32>>()
    };
    (3, $rows:expr, 0) => {
        $rows.iter().map(|row| row.0).collect::<Vec<u32>>()
    };
    (3, $rows:expr, 1) => {
        $rows.iter().map(|row| row.1).collect::<Vec<u32>>()
    };
    (3, $rows:expr, 2) => {
        $rows.iter().map(|row| row.2).collect::<Vec<u32>>()
    };
    (4, $rows:expr, 0) => {
        $rows.iter().map(|row| row.0).collect::<Vec<u32>>()
    };
    (4, $rows:expr, 1) => {
        $rows.iter().map(|row| row.1).collect::<Vec<u32>>()
    };
    (4, $rows:expr, 2) => {
        $rows.iter().map(|row| row.2).collect::<Vec<u32>>()
    };
    (4, $rows:expr, 3) => {
        $rows.iter().map(|row| row.3).collect::<Vec<u32>>()
    };
    (5, $rows:expr, 0) => {
        $rows.iter().map(|row| row.0).collect::<Vec<u32>>()
    };
    (5, $rows:expr, 1) => {
        $rows.iter().map(|row| row.1).collect::<Vec<u32>>()
    };
    (5, $rows:expr, 2) => {
        $rows.iter().map(|row| row.2).collect::<Vec<u32>>()
    };
    (5, $rows:expr, 3) => {
        $rows.iter().map(|row| row.3).collect::<Vec<u32>>()
    };
    (5, $rows:expr, 4) => {
        $rows.iter().map(|row| row.4).collect::<Vec<u32>>()
    };
    (6, $rows:expr, 0) => {
        $rows.iter().map(|row| row.0).collect::<Vec<u32>>()
    };
    (6, $rows:expr, 1) => {
        $rows.iter().map(|row| row.1).collect::<Vec<u32>>()
    };
    (6, $rows:expr, 2) => {
        $rows.iter().map(|row| row.2).collect::<Vec<u32>>()
    };
    (6, $rows:expr, 3) => {
        $rows.iter().map(|row| row.3).collect::<Vec<u32>>()
    };
    (6, $rows:expr, 4) => {
        $rows.iter().map(|row| row.4).collect::<Vec<u32>>()
    };
    (6, $rows:expr, 5) => {
        $rows.iter().map(|row| row.5).collect::<Vec<u32>>()
    };
    (7, $rows:expr, 0) => {
        $rows.iter().map(|row| row.0).collect::<Vec<u32>>()
    };
    (7, $rows:expr, 1) => {
        $rows.iter().map(|row| row.1).collect::<Vec<u32>>()
    };
    (7, $rows:expr, 2) => {
        $rows.iter().map(|row| row.2).collect::<Vec<u32>>()
    };
    (7, $rows:expr, 3) => {
        $rows.iter().map(|row| row.3).collect::<Vec<u32>>()
    };
    (7, $rows:expr, 4) => {
        $rows.iter().map(|row| row.4).collect::<Vec<u32>>()
    };
    (7, $rows:expr, 5) => {
        $rows.iter().map(|row| row.5).collect::<Vec<u32>>()
    };
    (7, $rows:expr, 6) => {
        $rows.iter().map(|row| row.6).collect::<Vec<u32>>()
    };
}

macro_rules! device_rows {
    (1, $exec:expr, $rows:expr) => {
        vec![$exec.to_device(&row_column!(1, $rows, 0))]
    };
    (2, $exec:expr, $rows:expr) => {
        vec![
            $exec.to_device(&row_column!(2, $rows, 0)),
            $exec.to_device(&row_column!(2, $rows, 1)),
        ]
    };
    (3, $exec:expr, $rows:expr) => {
        vec![
            $exec.to_device(&row_column!(3, $rows, 0)),
            $exec.to_device(&row_column!(3, $rows, 1)),
            $exec.to_device(&row_column!(3, $rows, 2)),
        ]
    };
    (4, $exec:expr, $rows:expr) => {
        vec![
            $exec.to_device(&row_column!(4, $rows, 0)),
            $exec.to_device(&row_column!(4, $rows, 1)),
            $exec.to_device(&row_column!(4, $rows, 2)),
            $exec.to_device(&row_column!(4, $rows, 3)),
        ]
    };
    (5, $exec:expr, $rows:expr) => {
        vec![
            $exec.to_device(&row_column!(5, $rows, 0)),
            $exec.to_device(&row_column!(5, $rows, 1)),
            $exec.to_device(&row_column!(5, $rows, 2)),
            $exec.to_device(&row_column!(5, $rows, 3)),
            $exec.to_device(&row_column!(5, $rows, 4)),
        ]
    };
    (6, $exec:expr, $rows:expr) => {
        vec![
            $exec.to_device(&row_column!(6, $rows, 0)),
            $exec.to_device(&row_column!(6, $rows, 1)),
            $exec.to_device(&row_column!(6, $rows, 2)),
            $exec.to_device(&row_column!(6, $rows, 3)),
            $exec.to_device(&row_column!(6, $rows, 4)),
            $exec.to_device(&row_column!(6, $rows, 5)),
        ]
    };
    (7, $exec:expr, $rows:expr) => {
        vec![
            $exec.to_device(&row_column!(7, $rows, 0)),
            $exec.to_device(&row_column!(7, $rows, 1)),
            $exec.to_device(&row_column!(7, $rows, 2)),
            $exec.to_device(&row_column!(7, $rows, 3)),
            $exec.to_device(&row_column!(7, $rows, 4)),
            $exec.to_device(&row_column!(7, $rows, 5)),
            $exec.to_device(&row_column!(7, $rows, 6)),
        ]
    };
}

macro_rules! assert_output {
    ($arity:tt, $exec:expr, $output:expr, $expected:expr, $len:expr) => {{
        let expected = &$expected;
        assert_output_columns!($arity, $exec, $output, expected, $len);
    }};
}

macro_rules! assert_mvec {
    ($arity:tt, $exec:expr, $output:expr, $expected:expr, $len:expr) => {{
        let expected = &$expected;
        assert_flat_mvec_columns!($arity, $exec, $output, expected, $len);
    }};
}

macro_rules! assert_flat_mvec_columns {
    (1, $exec:expr, $output:expr, $expected:expr, $len:expr) => {{
        let actual = $exec.to_host(&$output).unwrap();
        let expected = row_column!(1, $expected, 0);
        prop_assert_eq!(&actual[..$len], &expected[..$len]);
    }};
    (2, $exec:expr, $output:expr, $expected:expr, $len:expr) => {{
        let output = MStorage::into_columns($output);
        let actual = [&output.0, &output.1];
        let expected = [row_column!(2, $expected, 0), row_column!(2, $expected, 1)];
        assert_flat_mvec_columns!(@all $exec, actual, expected, $len);
    }};
    (3, $exec:expr, $output:expr, $expected:expr, $len:expr) => {{
        let output = MStorage::into_columns($output);
        let actual = [&output.0, &output.1, &output.2];
        let expected = [row_column!(3, $expected, 0), row_column!(3, $expected, 1), row_column!(3, $expected, 2)];
        assert_flat_mvec_columns!(@all $exec, actual, expected, $len);
    }};
    (4, $exec:expr, $output:expr, $expected:expr, $len:expr) => {{
        let output = MStorage::into_columns($output);
        let actual = [&output.0, &output.1, &output.2, &output.3];
        let expected = [row_column!(4, $expected, 0), row_column!(4, $expected, 1), row_column!(4, $expected, 2), row_column!(4, $expected, 3)];
        assert_flat_mvec_columns!(@all $exec, actual, expected, $len);
    }};
    (5, $exec:expr, $output:expr, $expected:expr, $len:expr) => {{
        let output = MStorage::into_columns($output);
        let actual = [&output.0, &output.1, &output.2, &output.3, &output.4];
        let expected = [row_column!(5, $expected, 0), row_column!(5, $expected, 1), row_column!(5, $expected, 2), row_column!(5, $expected, 3), row_column!(5, $expected, 4)];
        assert_flat_mvec_columns!(@all $exec, actual, expected, $len);
    }};
    (6, $exec:expr, $output:expr, $expected:expr, $len:expr) => {{
        let output = MStorage::into_columns($output);
        let actual = [&output.0, &output.1, &output.2, &output.3, &output.4, &output.5];
        let expected = [row_column!(6, $expected, 0), row_column!(6, $expected, 1), row_column!(6, $expected, 2), row_column!(6, $expected, 3), row_column!(6, $expected, 4), row_column!(6, $expected, 5)];
        assert_flat_mvec_columns!(@all $exec, actual, expected, $len);
    }};
    (7, $exec:expr, $output:expr, $expected:expr, $len:expr) => {{
        let output = MStorage::into_columns($output);
        let actual = [&output.0, &output.1, &output.2, &output.3, &output.4, &output.5, &output.6];
        let expected = [row_column!(7, $expected, 0), row_column!(7, $expected, 1), row_column!(7, $expected, 2), row_column!(7, $expected, 3), row_column!(7, $expected, 4), row_column!(7, $expected, 5), row_column!(7, $expected, 6)];
        assert_flat_mvec_columns!(@all $exec, actual, expected, $len);
    }};
    (@all $exec:expr, $actual:expr, $expected:expr, $len:expr) => {{
        for (actual, expected) in $actual.into_iter().zip($expected) {
            let actual = $exec.to_host(actual).unwrap();
            prop_assert_eq!(&actual[..$len], &expected[..$len]);
        }
    }};
}

macro_rules! assert_one_column {
    ($arity:tt, $column:tt, $exec:expr, $output:expr, $expected:expr, $len:expr) => {{
        let actual = $exec.to_host(&$output[$column]).unwrap();
        let expected_column = row_column!($arity, $expected, $column);
        prop_assert_eq!(&actual[..$len], &expected_column[..$len]);
    }};
}

macro_rules! assert_output_columns {
    (1, $exec:expr, $output:expr, $expected:expr, $len:expr) => {
        assert_one_column!(1, 0, $exec, $output, $expected, $len)
    };
    (2, $exec:expr, $output:expr, $expected:expr, $len:expr) => {{
        assert_one_column!(2, 0, $exec, $output, $expected, $len);
        assert_one_column!(2, 1, $exec, $output, $expected, $len);
    }};
    (3, $exec:expr, $output:expr, $expected:expr, $len:expr) => {{
        assert_one_column!(3, 0, $exec, $output, $expected, $len);
        assert_one_column!(3, 1, $exec, $output, $expected, $len);
        assert_one_column!(3, 2, $exec, $output, $expected, $len);
    }};
    (4, $exec:expr, $output:expr, $expected:expr, $len:expr) => {{
        assert_one_column!(4, 0, $exec, $output, $expected, $len);
        assert_one_column!(4, 1, $exec, $output, $expected, $len);
        assert_one_column!(4, 2, $exec, $output, $expected, $len);
        assert_one_column!(4, 3, $exec, $output, $expected, $len);
    }};
    (5, $exec:expr, $output:expr, $expected:expr, $len:expr) => {{
        assert_one_column!(5, 0, $exec, $output, $expected, $len);
        assert_one_column!(5, 1, $exec, $output, $expected, $len);
        assert_one_column!(5, 2, $exec, $output, $expected, $len);
        assert_one_column!(5, 3, $exec, $output, $expected, $len);
        assert_one_column!(5, 4, $exec, $output, $expected, $len);
    }};
    (6, $exec:expr, $output:expr, $expected:expr, $len:expr) => {{
        assert_one_column!(6, 0, $exec, $output, $expected, $len);
        assert_one_column!(6, 1, $exec, $output, $expected, $len);
        assert_one_column!(6, 2, $exec, $output, $expected, $len);
        assert_one_column!(6, 3, $exec, $output, $expected, $len);
        assert_one_column!(6, 4, $exec, $output, $expected, $len);
        assert_one_column!(6, 5, $exec, $output, $expected, $len);
    }};
    (7, $exec:expr, $output:expr, $expected:expr, $len:expr) => {{
        assert_one_column!(7, 0, $exec, $output, $expected, $len);
        assert_one_column!(7, 1, $exec, $output, $expected, $len);
        assert_one_column!(7, 2, $exec, $output, $expected, $len);
        assert_one_column!(7, 3, $exec, $output, $expected, $len);
        assert_one_column!(7, 4, $exec, $output, $expected, $len);
        assert_one_column!(7, 5, $exec, $output, $expected, $len);
        assert_one_column!(7, 6, $exec, $output, $expected, $len);
    }};
}

macro_rules! setup {
    ($arity:tt, $seed:expr; $exec:ident, $input:ident, $device:ident, $columns:ident) => {
        let $exec = exec();
        let $columns = seven_columns(&$seed);
        let $input = rows!($arity, $columns);
        #[allow(unused_variables)]
        let $device: Vec<_> = $columns
            .iter()
            .map(|column| $exec.to_device(column))
            .collect();
    };
}

macro_rules! empty_output {
    ($exec:expr, $arity:tt, $len:expr) => {
        (0..$arity)
            .map(|_| $exec.to_device(&vec![0_u32; $len]))
            .collect::<Vec<_>>()
    };
}

macro_rules! value_case {
    (reduce, $arity:tt, $seed:expr) => {{
        setup!($arity, $seed; exec, input, device, columns);
        let zero = splat!($arity, 0u32);
        prop_assert_eq!(read_value(&exec, massively::vector::reduce(&exec, input_expr!($arity, device), zero, MaxItem).unwrap()), reference::reduce(&input, zero, MaxItem));
    }};
    (inclusive_scan, $arity:tt, $seed:expr) => {{
        setup!($arity, $seed; exec, input, device, columns);
        let output = massively::vector::inclusive_scan(&exec, input_expr!($arity, device), MaxItem).unwrap();
        assert_mvec!($arity, exec, output, reference::inclusive_scan(&input, MaxItem), input.len());
    }};
    (exclusive_scan, $arity:tt, $seed:expr) => {{
        setup!($arity, $seed; exec, input, device, columns); let zero = splat!($arity, 0u32);
        let device_zero = zero;
        let output = massively::vector::exclusive_scan(&exec, input_expr!($arity, device), device_zero, MaxItem).unwrap();
        assert_mvec!($arity, exec, output, reference::exclusive_scan(&input, zero, MaxItem), input.len());
    }};
    (adjacent_difference, $arity:tt, $seed:expr) => {{
        setup!($arity, $seed; exec, input, device, columns);
        let output = massively::vector::adjacent_difference(&exec, input_expr!($arity, device), MaxItem).unwrap();
        assert_mvec!($arity, exec, output, reference::adjacent_difference(&input, MaxItem), input.len());
    }};
    ($case:ident, $arity:tt, $seed:expr) => {{ value_case_other!($case, $arity, $seed) }};
}

macro_rules! value_case_other {
    (copy_where, $arity:tt, $seed:expr) => {{
        setup!($arity, $seed; exec, input, device, columns); let flags = flags_for(&$seed); let flags_gpu = exec.to_device(&flags);
        let output = massively::vector::copy_where(&exec, input_expr!($arity, device), as_stencil(lazify(flags_gpu.slice(..)))).unwrap();
        let expected = reference::copy_where(&input, &flags); let len = expected.len(); assert_mvec!($arity, exec, output, expected, len);
    }};
    (remove_where, $arity:tt, $seed:expr) => {{
        setup!($arity, $seed; exec, input, device, columns); let flags = flags_for(&$seed); let flags_gpu = exec.to_device(&flags);
        let output = massively::vector::remove_where(&exec, input_expr!($arity, device), as_stencil(lazify(flags_gpu.slice(..)))).unwrap();
        let expected = reference::remove_where(&input, &flags); let len = expected.len(); assert_mvec!($arity, exec, output, expected, len);
    }};
    (reverse, $arity:tt, $seed:expr) => {{
        setup!($arity, $seed; exec, input, device, columns); let output = massively::vector::reverse(&exec, input_expr!($arity, device)).unwrap(); assert_mvec!($arity, exec, output, reference::reverse(&input), input.len());
    }};
    (count_if, $arity:tt, $seed:expr) => {{ setup!($arity, $seed; exec, input, device, columns); prop_assert_eq!(read_value(&exec, massively::vector::count_if(&exec, input_expr!($arity, device), EvenItem).unwrap()) as usize, reference::count_if(&input, EvenItem)); }};
    (all_of, $arity:tt, $seed:expr) => {{ setup!($arity, $seed; exec, input, device, columns); prop_assert_eq!(read_value(&exec, massively::vector::all_of(&exec, input_expr!($arity, device), EvenItem).unwrap()), expected_flag(reference::all_of(&input, EvenItem))); }};
    (any_of, $arity:tt, $seed:expr) => {{ setup!($arity, $seed; exec, input, device, columns); prop_assert_eq!(read_value(&exec, massively::vector::any_of(&exec, input_expr!($arity, device), EvenItem).unwrap()), expected_flag(reference::any_of(&input, EvenItem))); }};
    (none_of, $arity:tt, $seed:expr) => {{ setup!($arity, $seed; exec, input, device, columns); prop_assert_eq!(read_value(&exec, massively::vector::none_of(&exec, input_expr!($arity, device), EvenItem).unwrap()), expected_flag(reference::none_of(&input, EvenItem))); }};
    (find_if, $arity:tt, $seed:expr) => {{ setup!($arity, $seed; exec, input, device, columns); prop_assert_eq!(read_optional_index(&exec, massively::vector::find_if(&exec, input_expr!($arity, device), EvenItem).unwrap()), reference::find_if(&input, EvenItem)); }};
    (is_partitioned, $arity:tt, $seed:expr) => {{ setup!($arity, $seed; exec, input, device, columns); prop_assert_eq!(read_value(&exec, massively::vector::is_partitioned(&exec, input_expr!($arity, device), EvenItem).unwrap()), expected_flag(reference::is_partitioned(&input, EvenItem))); }};
    (partition, $arity:tt, $seed:expr) => {{
        setup!($arity, $seed; exec, input, device, columns); let (output, boundary) = massively::vector::partition(&exec, input_expr!($arity, device), EvenItem).unwrap(); let boundary = read_value(&exec, boundary) as usize;
        let (mut selected, rejected) = reference::partition(&input, EvenItem); prop_assert_eq!(boundary, selected.len()); selected.extend(rejected); assert_mvec!($arity, exec, output, selected, input.len());
    }};
    (permute, $arity:tt, $seed:expr) => {{
        setup!($arity, $seed; exec, input, device, columns); let indices = indices_for(input.len()); let indices_gpu = exec.to_device(&indices); let mut expected = vec![splat!($arity, 0u32); input.len()]; reference::gather(&input, &indices, &mut expected);
        let permuted = lazy::identity(lazy::permute(
            raw_input_expr!($arity, device),
            as_indices(indices_gpu.slice(..)),
        ));
        let output = massively::vector::map(&exec, permuted, Identity).unwrap(); assert_mvec!($arity, exec, output, expected, input.len());
    }};
    (gather, $arity:tt, $seed:expr) => {{
        setup!($arity, $seed; exec, input, device, columns); let indices = indices_for(input.len()); let indices_gpu = exec.to_device(&indices); let mut expected = vec![splat!($arity, 0u32); input.len()]; reference::gather(&input, &indices, &mut expected);
        let output = massively::vector::gather(&exec, input_expr!($arity, device), as_indices(lazify(indices_gpu.slice(..)))).unwrap(); assert_mvec!($arity, exec, output, expected, input.len());
    }};
    (gather_where, $arity:tt, $seed:expr) => {{
        setup!($arity, $seed; exec, input, device, columns); let indices = indices_for(input.len()); let indices_gpu = exec.to_device(&indices); let flags = flags_for(&$seed); let flags_gpu = exec.to_device(&flags); let output = device_rows!($arity, exec, input); let mut expected = input.clone(); reference::gather_where(&input, &indices, &flags, &mut expected);
        massively::vector::gather_where(&exec, input_expr!($arity, device), as_indices(lazify(indices_gpu.slice(..))), as_stencil(lazify(flags_gpu.slice(..))), output_expr!($arity, output)).unwrap(); assert_output!($arity, exec, output, expected, input.len());
    }};
    (scatter, $arity:tt, $seed:expr) => {{
        setup!($arity, $seed; exec, input, device, columns); let indices = indices_for(input.len()); let indices_gpu = exec.to_device(&indices); let output = empty_output!(exec, $arity, input.len()); let mut expected = vec![splat!($arity, 0u32); input.len()]; reference::scatter(&input, &indices, &mut expected);
        massively::vector::scatter(&exec, input_expr!($arity, device), as_indices(lazify(indices_gpu.slice(..))), output_expr!($arity, output)).unwrap(); assert_output!($arity, exec, output, expected, input.len());
    }};
    (scatter_where, $arity:tt, $seed:expr) => {{
        setup!($arity, $seed; exec, input, device, columns); let indices = indices_for(input.len()); let indices_gpu = exec.to_device(&indices); let flags = flags_for(&$seed); let flags_gpu = exec.to_device(&flags); let output = device_rows!($arity, exec, input); let mut expected = input.clone(); reference::scatter_where(&input, &indices, &flags, &mut expected);
        massively::vector::scatter_where(&exec, input_expr!($arity, device), as_indices(lazify(indices_gpu.slice(..))), as_stencil(lazify(flags_gpu.slice(..))), output_expr!($arity, output)).unwrap(); assert_output!($arity, exec, output, expected, input.len());
    }};
    (equal, $arity:tt, $seed:expr) => {{ setup!($arity, $seed; exec, input, device, columns); prop_assert_eq!(read_value(&exec, massively::vector::equal(&exec, input_expr!($arity, device), input_expr!($arity, device), EqualItem).unwrap()), expected_flag(reference::equal(&input, &input, EqualItem))); }};
    (mismatch, $arity:tt, $seed:expr) => {{ setup!($arity, $seed; exec, input, device, columns); let mut other = input.clone(); if let Some(last) = other.last_mut() { *last = splat!($arity, 999u32); } let other_device = device_rows!($arity, exec, other); prop_assert_eq!(read_optional_index(&exec, massively::vector::mismatch(&exec, input_expr!($arity, device), input_expr!($arity, other_device), EqualItem).unwrap()), reference::mismatch(&input, &other, EqualItem)); }};
    (adjacent_find, $arity:tt, $seed:expr) => {{ setup!($arity, $seed; exec, input, device, columns); prop_assert_eq!(read_optional_index(&exec, massively::vector::adjacent_find(&exec, input_expr!($arity, device), EqualItem).unwrap()), reference::adjacent_find(&input, EqualItem)); }};
    (find_first_of, $arity:tt, $seed:expr) => {{ setup!($arity, $seed; exec, input, device, columns); let needles: Vec<_> = input.iter().step_by(3).copied().collect(); let needles_device = device_rows!($arity, exec, needles); prop_assert_eq!(read_optional_index(&exec, massively::vector::find_first_of(&exec, input_expr!($arity, device), input_expr!($arity, needles_device), EqualItem).unwrap()), reference::find_first_of(&input, &needles, EqualItem)); }};
    (fill, $arity:tt, $seed:expr) => {{ setup!($arity, $seed; exec, input, device, columns); let value = splat!($arity, 42u32); let output = empty_output!(exec, $arity, input.len()); massively::vector::fill(&exec, value, output_expr!($arity, output)).unwrap(); let mut expected = vec![splat!($arity, 0u32); input.len()]; reference::fill(value, &mut expected); assert_output!($arity, exec, output, expected, input.len()); }};
    (replace_where, $arity:tt, $seed:expr) => {{ setup!($arity, $seed; exec, input, device, columns); let flags = flags_for(&$seed); let flags_gpu = exec.to_device(&flags); let output = device_rows!($arity, exec, input); let value = splat!($arity, 42u32); massively::vector::replace_where(&exec, value, as_stencil(lazify(flags_gpu.slice(..))), output_expr!($arity, output)).unwrap(); let mut expected = input.clone(); reference::replace_where(value, &flags, &mut expected); assert_output!($arity, exec, output, expected, input.len()); }};
    ($case:ident, $arity:tt, $seed:expr) => {{ ordering_case!($case, $arity, $seed) }};
}

macro_rules! ordering_case {
    (sort, $arity:tt, $seed:expr) => {{ setup!($arity, $seed; exec, input, device, columns); let output = massively::vector::sort(&exec, input_expr!($arity, device), LessItem).unwrap(); let expected = reference::sort(&input, LessItem); assert_mvec!($arity, exec, output, expected, input.len()); }};
    (merge, $arity:tt, $seed:expr) => {{ setup!($arity, $seed; exec, input, device, columns); let left = reference::sort(&input[..input.len()/2], LessItem); let right = reference::sort(&input[input.len()/2..], LessItem); let left_device = device_rows!($arity, exec, left); let right_device = device_rows!($arity, exec, right); let output = massively::vector::merge(&exec, input_expr!($arity, left_device), input_expr!($arity, right_device), LessItem).unwrap(); let expected = reference::merge(&left, &right, LessItem); assert_mvec!($arity, exec, output, expected, input.len()); }};
    (is_sorted, $arity:tt, $seed:expr) => {{ setup!($arity, $seed; exec, input, device, columns); prop_assert_eq!(read_value(&exec, massively::vector::is_sorted(&exec, input_expr!($arity, device), LessItem).unwrap()), expected_flag(reference::is_sorted(&input, LessItem))); }};
    (is_sorted_until, $arity:tt, $seed:expr) => {{ setup!($arity, $seed; exec, input, device, columns); prop_assert_eq!(read_value(&exec, massively::vector::is_sorted_until(&exec, input_expr!($arity, device), LessItem).unwrap()) as usize, reference::is_sorted_until(&input, LessItem)); }};
    (lexicographical_compare, $arity:tt, $seed:expr) => {{ setup!($arity, $seed; exec, input, device, columns); let mut other = input.clone(); other.reverse(); let other_device = device_rows!($arity, exec, other); prop_assert_eq!(read_value(&exec, massively::vector::lexicographical_compare(&exec, input_expr!($arity, device), input_expr!($arity, other_device), LessItem).unwrap()), expected_flag(reference::lexicographical_compare(&input, &other, LessItem))); }};
    (min_element, $arity:tt, $seed:expr) => {{ setup!($arity, $seed; exec, input, device, columns); prop_assert_eq!(read_optional_index(&exec, massively::vector::min_element(&exec, input_expr!($arity, device), LessItem).unwrap()), reference::min_element(&input, LessItem)); }};
    (max_element, $arity:tt, $seed:expr) => {{ setup!($arity, $seed; exec, input, device, columns); prop_assert_eq!(read_optional_index(&exec, massively::vector::max_element(&exec, input_expr!($arity, device), LessItem).unwrap()), reference::max_element(&input, LessItem)); }};
    (minmax_element, $arity:tt, $seed:expr) => {{ setup!($arity, $seed; exec, input, device, columns); prop_assert_eq!(read_optional_index_pair(&exec, massively::vector::minmax_element(&exec, input_expr!($arity, device), LessItem).unwrap()), reference::minmax_element(&input, LessItem)); }};
    (lower_bound, $arity:tt, $seed:expr) => {{ setup!($arity, $seed; exec, input, device, columns); let sorted = reference::sort(&input, LessItem); let sorted_device = device_rows!($arity, exec, sorted); let output = massively::vector::lower_bound(&exec, input_expr!($arity, sorted_device), input_expr!($arity, device), LessItem).unwrap(); prop_assert_eq!(exec.to_host(&output).unwrap(), reference::lower_bound(&sorted, &input, LessItem)); }};
    (upper_bound, $arity:tt, $seed:expr) => {{ setup!($arity, $seed; exec, input, device, columns); let sorted = reference::sort(&input, LessItem); let sorted_device = device_rows!($arity, exec, sorted); let output = massively::vector::upper_bound(&exec, input_expr!($arity, sorted_device), input_expr!($arity, device), LessItem).unwrap(); prop_assert_eq!(exec.to_host(&output).unwrap(), reference::upper_bound(&sorted, &input, LessItem)); }};
    (unique, $arity:tt, $seed:expr) => {{ setup!($arity, $seed; exec, input, device, columns); let sorted = reference::sort(&input, LessItem); let sorted_device = device_rows!($arity, exec, sorted); let output = massively::vector::unique(&exec, input_expr!($arity, sorted_device), EqualItem).unwrap(); let expected = reference::unique(&sorted, EqualItem); let len = expected.len(); assert_mvec!($arity, exec, output, expected, len); }};
    (set_union, $arity:tt, $seed:expr) => {{ set_case!(set_union, set_union, $arity, $seed) }};
    (set_intersection, $arity:tt, $seed:expr) => {{ set_case!(set_intersection, set_intersection, $arity, $seed) }};
    (set_difference, $arity:tt, $seed:expr) => {{ set_case!(set_difference, set_difference, $arity, $seed) }};
}

macro_rules! set_case {
    ($algorithm:ident, $oracle:ident, $arity:tt, $seed:expr) => {{
        setup!($arity, $seed; exec, input, device, columns); let split = input.len()/2; let left = reference::sort(&input[..split], LessItem); let right = reference::sort(&input[split..], LessItem); let left_device = device_rows!($arity, exec, left); let right_device = device_rows!($arity, exec, right);
        let output = massively::vector::$algorithm(&exec, input_expr!($arity, left_device), input_expr!($arity, right_device), LessItem).unwrap();
        let expected = reference::$oracle(&left, &right, LessItem); let len = expected.len(); assert_mvec!($arity, exec, output, expected, len);
    }};
}

macro_rules! define_arity_module {
    ($module:ident, $case:ident) => {
        mod $module {
            use super::*;
            proptest! {
                #![proptest_config(ProptestConfig { cases: CASES, .. ProptestConfig::default() })]
                #[test] fn arity_1(seed in oracle_vec(0u32..100)) { value_case!($case, 1, seed); }
                #[test] fn arity_2(seed in oracle_vec(0u32..100)) { value_case!($case, 2, seed); }
                #[test] fn arity_3(seed in oracle_vec(0u32..100)) { value_case!($case, 3, seed); }
                #[test] fn arity_4(seed in oracle_vec(0u32..100)) { value_case!($case, 4, seed); }
                #[test] fn arity_5(seed in oracle_vec(0u32..100)) { value_case!($case, 5, seed); }
                #[test] fn arity_6(seed in oracle_vec(0u32..100)) { value_case!($case, 6, seed); }
                #[test] fn arity_7(seed in oracle_vec(0u32..100)) { value_case!($case, 7, seed); }
            }
        }
    };
}

define_arity_module!(reduce_arity, reduce);
define_arity_module!(inclusive_scan_arity, inclusive_scan);
define_arity_module!(exclusive_scan_arity, exclusive_scan);
define_arity_module!(adjacent_difference_arity, adjacent_difference);
define_arity_module!(copy_where_arity, copy_where);
define_arity_module!(remove_where_arity, remove_where);
define_arity_module!(reverse_arity, reverse);
define_arity_module!(count_if_arity, count_if);
define_arity_module!(all_of_arity, all_of);
define_arity_module!(any_of_arity, any_of);
define_arity_module!(none_of_arity, none_of);
define_arity_module!(find_if_arity, find_if);
define_arity_module!(is_partitioned_arity, is_partitioned);
define_arity_module!(partition_arity, partition);
define_arity_module!(permute_arity, permute);
define_arity_module!(gather_arity, gather);
define_arity_module!(gather_where_arity, gather_where);
define_arity_module!(scatter_arity, scatter);
define_arity_module!(scatter_where_arity, scatter_where);
define_arity_module!(equal_arity, equal);
define_arity_module!(mismatch_arity, mismatch);
define_arity_module!(adjacent_find_arity, adjacent_find);
define_arity_module!(find_first_of_arity, find_first_of);
define_arity_module!(fill_arity, fill);
define_arity_module!(replace_where_arity, replace_where);
define_arity_module!(sort_arity, sort);
define_arity_module!(merge_arity, merge);
define_arity_module!(is_sorted_arity, is_sorted);
define_arity_module!(is_sorted_until_arity, is_sorted_until);
define_arity_module!(lexicographical_compare_arity, lexicographical_compare);
define_arity_module!(min_element_arity, min_element);
define_arity_module!(max_element_arity, max_element);
define_arity_module!(minmax_element_arity, minmax_element);
define_arity_module!(lower_bound_arity, lower_bound);
define_arity_module!(upper_bound_arity, upper_bound);
define_arity_module!(unique_arity, unique);
define_arity_module!(set_union_arity, set_union);
define_arity_module!(set_intersection_arity, set_intersection);
define_arity_module!(set_difference_arity, set_difference);

macro_rules! by_key_setup {
    ($key_arity:tt, $value_arity:tt, $pairs:expr;
     $exec:ident, $keys:ident, $values:ident, $key_device:ident, $value_device:ident) => {
        let $exec = exec();
        let key_seed: Vec<_> = $pairs.iter().map(|pair| pair.0).collect();
        let value_seed: Vec<_> = $pairs.iter().map(|pair| pair.1).collect();
        let key_columns = seven_columns(&key_seed);
        let value_columns = seven_columns(&value_seed);
        let $keys = rows!($key_arity, key_columns);
        let $values = rows!($value_arity, value_columns);
        #[allow(unused_variables)]
        let $key_device: Vec<_> = key_columns
            .iter()
            .map(|column| $exec.to_device(column))
            .collect();
        #[allow(unused_variables)]
        let $value_device: Vec<_> = value_columns
            .iter()
            .map(|column| $exec.to_device(column))
            .collect();
    };
}

macro_rules! by_key_case {
    (sort_by_key, $key_arity:tt, $value_arity:tt, $pairs:expr) => {{
        by_key_setup!($key_arity, $value_arity, $pairs; exec, keys, values, key_device, value_device);
        let output = massively::vector::sort_by_key(&exec, input_expr!($key_arity, key_device), input_expr!($value_arity, value_device), LessItem).unwrap();
        let (_, expected_values) = reference::sort_by_key(&keys, &values, LessItem);
        assert_mvec!($value_arity, exec, output, expected_values, values.len());
    }};
    (inclusive_scan_by_key, $key_arity:tt, $value_arity:tt, $pairs:expr) => {{
        by_key_setup!($key_arity, $value_arity, $pairs; exec, keys, values, key_device, value_device);
        let output = massively::vector::inclusive_scan_by_key(&exec, input_expr!($key_arity, key_device), input_expr!($value_arity, value_device), EqualItem, MaxItem).unwrap();
        let expected = reference::inclusive_scan_by_key(&keys, &values, EqualItem, MaxItem);
        assert_mvec!($value_arity, exec, output, expected, values.len());
    }};
    (exclusive_scan_by_key, $key_arity:tt, $value_arity:tt, $pairs:expr) => {{
        by_key_setup!($key_arity, $value_arity, $pairs; exec, keys, values, key_device, value_device);
        let zero = splat!($value_arity, 0u32);
        let device_zero = zero;
        let output = massively::vector::exclusive_scan_by_key(&exec, input_expr!($key_arity, key_device), input_expr!($value_arity, value_device), EqualItem, device_zero, MaxItem).unwrap();
        let expected = reference::exclusive_scan_by_key(&keys, &values, EqualItem, zero, MaxItem);
        assert_mvec!($value_arity, exec, output, expected, values.len());
    }};
    (reduce_by_key, $key_arity:tt, $value_arity:tt, $pairs:expr) => {{
        by_key_setup!($key_arity, $value_arity, $pairs; exec, keys, values, key_device, value_device);
        let zero = splat!($value_arity, 0u32);
        let device_zero = zero;
        let (key_output, value_output) = massively::vector::reduce_by_key(&exec, input_expr!($key_arity, key_device), input_expr!($value_arity, value_device), EqualItem, device_zero, MaxItem).unwrap();
        let (expected_keys, expected_values) = reference::reduce_by_key(&keys, &values, EqualItem, zero, MaxItem); let len = expected_keys.len();
        assert_mvec!($key_arity, exec, key_output, expected_keys, len); assert_mvec!($value_arity, exec, value_output, expected_values, len);
    }};
    (unique_by_key, $key_arity:tt, $value_arity:tt, $pairs:expr) => {{
        by_key_setup!($key_arity, $value_arity, $pairs; exec, keys, values, key_device, value_device);
        let output = massively::vector::unique_by_key(&exec, input_expr!($key_arity, key_device), input_expr!($value_arity, value_device), EqualItem).unwrap();
        let (_, expected_values) = reference::unique_by_key(&keys, &values, EqualItem); let len = expected_values.len();
        assert_mvec!($value_arity, exec, output, expected_values, len);
    }};
    (merge_by_key, $key_arity:tt, $value_arity:tt, $pairs:expr) => {{
        by_key_setup!($key_arity, $value_arity, $pairs; exec, keys, values, key_device, value_device);
        let split = keys.len()/2; let (left_keys, left_values) = reference::sort_by_key(&keys[..split], &values[..split], LessItem); let (right_keys, right_values) = reference::sort_by_key(&keys[split..], &values[split..], LessItem);
        let left_key_device = device_rows!($key_arity, exec, left_keys); let right_key_device = device_rows!($key_arity, exec, right_keys); let left_value_device = device_rows!($value_arity, exec, left_values); let right_value_device = device_rows!($value_arity, exec, right_values);
        let output = massively::vector::merge_by_key(&exec, input_expr!($key_arity, left_key_device), input_expr!($value_arity, left_value_device), input_expr!($key_arity, right_key_device), input_expr!($value_arity, right_value_device), LessItem).unwrap();
        let (_, expected_values) = reference::merge_by_key(&left_keys, &left_values, &right_keys, &right_values, LessItem);
        assert_mvec!($value_arity, exec, output, expected_values, values.len());
    }};
}

macro_rules! define_by_key_product_module {
    ($module:ident, $case:ident) => {
        mod $module {
            use super::*;
            macro_rules! product_case {
                ($name:ident, $keys:tt, $values:tt) => {
                    proptest! {
                        #![proptest_config(ProptestConfig { cases: CASES, .. ProptestConfig::default() })]
                        #[test]
                        fn $name(pairs in oracle_vec((0u32..32, 0u32..100))) {
                            by_key_case!($case, $keys, $values, pairs);
                        }
                    }
                };
            }
            product_case!(key_1_value_1, 1, 1); product_case!(key_1_value_2, 1, 2); product_case!(key_1_value_3, 1, 3); product_case!(key_1_value_4, 1, 4); product_case!(key_1_value_5, 1, 5); product_case!(key_1_value_6, 1, 6); product_case!(key_1_value_7, 1, 7);
            product_case!(key_2_value_1, 2, 1); product_case!(key_2_value_2, 2, 2); product_case!(key_2_value_3, 2, 3); product_case!(key_2_value_4, 2, 4); product_case!(key_2_value_5, 2, 5); product_case!(key_2_value_6, 2, 6); product_case!(key_2_value_7, 2, 7);
            product_case!(key_3_value_1, 3, 1); product_case!(key_3_value_2, 3, 2); product_case!(key_3_value_3, 3, 3); product_case!(key_3_value_4, 3, 4); product_case!(key_3_value_5, 3, 5); product_case!(key_3_value_6, 3, 6); product_case!(key_3_value_7, 3, 7);
        }
    };
}

define_by_key_product_module!(sort_by_key_arity, sort_by_key);
define_by_key_product_module!(inclusive_scan_by_key_arity, inclusive_scan_by_key);
define_by_key_product_module!(exclusive_scan_by_key_arity, exclusive_scan_by_key);
define_by_key_product_module!(reduce_by_key_arity, reduce_by_key);
define_by_key_product_module!(unique_by_key_arity, unique_by_key);
define_by_key_product_module!(merge_by_key_arity, merge_by_key);
