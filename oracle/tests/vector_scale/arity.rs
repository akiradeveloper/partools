use cubecl::prelude::*;
use massively::{op::*, *};
use oracle::{op, vector as reference};

use super::common::{exec, read_value};

const A13_SCALE_LEN: usize = 65_537;

type Twelve = (u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32);

struct MaxTwelve;

macro_rules! max_twelve {
    ($lhs:ident, $rhs:ident) => {
        (
            $lhs.0.max($rhs.0),
            $lhs.1.max($rhs.1),
            $lhs.2.max($rhs.2),
            $lhs.3.max($rhs.3),
            $lhs.4.max($rhs.4),
            $lhs.5.max($rhs.5),
            $lhs.6.max($rhs.6),
            $lhs.7.max($rhs.7),
            $lhs.8.max($rhs.8),
            $lhs.9.max($rhs.9),
            $lhs.10.max($rhs.10),
            $lhs.11.max($rhs.11),
        )
    };
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
        max_twelve!(lhs, rhs)
    }
}

#[test]
fn lazify_reduce_at_read_arity_13() {
    let columns: [Vec<u32>; 12] = core::array::from_fn(|column| {
        (0..A13_SCALE_LEN)
            .map(|index| ((index * (column + 3) + column * 17) % 100_003) as u32)
            .collect()
    });
    let aos: Vec<Twelve> = (0..A13_SCALE_LEN)
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
        .collect();
    let exec = exec();
    let device: Vec<_> = columns
        .iter()
        .map(|column| exec.to_device(column))
        .collect();
    let lazified = lazy::identity(lazy::permute(
        zip12(
            device[0].slice(..),
            device[1].slice(..),
            device[2].slice(..),
            device[3].slice(..),
            device[4].slice(..),
            device[5].slice(..),
            device[6].slice(..),
            device[7].slice(..),
            device[8].slice(..),
            device[9].slice(..),
            device[10].slice(..),
            device[11].slice(..),
        ),
        lazy::counting(0).take(A13_SCALE_LEN as massively::MIndex),
    ));
    let zero = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

    assert_eq!(
        read_value(
            &exec,
            massively::vector::reduce(&exec, lazified, zero, MaxTwelve).unwrap(),
        ),
        reference::reduce(&aos, zero, MaxTwelve),
    );
}
