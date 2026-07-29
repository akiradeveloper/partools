//! Index-sideband reductions over arbitrary read expressions.

use cubecl::prelude::*;

use crate::core::arity::{A13, Dispatch};
use crate::core::bindings::Bindings;
use crate::core::eval::Eval13;
use crate::core::launch::cube_count_1d;
use crate::core::read::{
    Env0, Env13, KernelReadSlots, LowerReadExpression, PaddedReadSlots, ReadExpression, StageRead,
};
use crate::core::value::MStorageElement;
use crate::{DeviceVec, Error, Executor};

const BLOCK_SIZE: u32 = 256;
const ITEMS_PER_UNIT: usize = 256;
const TILE_SIZE: usize = BLOCK_SIZE as usize * ITEMS_PER_UNIT;

/// Chooses one of two indexed semantic values.
#[doc(hidden)]
#[cubecl::cube]
pub trait ArgReductionOp<Item: CubeType>: 'static + Send + Sync {
    /// Returns whether the right candidate wins. The caller presents candidates
    /// in ascending original-index order, so operators can implement stable
    /// first/last tie breaking with one comparison.
    fn rhs_wins(lhs: Item, rhs: Item) -> bool;
}

macro_rules! define_arg_reduce_kernel {
    ($name:ident,$eval:ident,$method:ident; [$( $leaf:ident:$slot:ident ),+]) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $name<
            Item: CubeType + Send + Sync + 'static,
            $( $leaf: CubePrimitive + cubecl::frontend::Scalar, )+
            Expr: $eval<Item, $( $leaf ),+>,
            Op: ArgReductionOp<Item>,
        >(
            $( $slot: &[$leaf], )+
            offsets: &[u32],
            len: &[u32],
            partials: &mut [u32],
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = BLOCK_SIZE as usize;
            let logical_len = len[0] as usize;
            if CUBE_POS as usize >= crate::core::launch::logical_block_count(logical_len, TILE_SIZE) {
                if UNIT_POS == 0u32 && (CUBE_POS as usize) < partials.len() {
                    partials[CUBE_POS as usize] = u32::MAX;
                }
                terminate!();
            }
            let tile_start = CUBE_POS as usize * TILE_SIZE;
            let first_index = tile_start + unit;
            let safe_index = if first_index < logical_len {
                first_index
            } else {
                logical_len - 1usize
            };
            let accumulator = RuntimeCell::<u32>::new(safe_index as u32);
            let valid = RuntimeCell::<u32>::new(
                if first_index < logical_len { 1u32 } else { 0u32 },
            );

            if logical_len - tile_start >= TILE_SIZE {
                for item in 1usize..ITEMS_PER_UNIT {
                    let rhs_index = first_index + item * cube_dim;
                    let lhs_index = accumulator.read() as usize;
                    if Op::rhs_wins(
                        Expr::$method($( $slot, )+ offsets, lhs_index),
                        Expr::$method($( $slot, )+ offsets, rhs_index),
                    ) {
                        accumulator.store(rhs_index as u32);
                    }
                }
            } else {
                for item in 1usize..ITEMS_PER_UNIT {
                    let rhs_index = first_index + item * cube_dim;
                    if rhs_index < logical_len {
                        if valid.read() != 0u32 {
                            let lhs_index = accumulator.read() as usize;
                            if Op::rhs_wins(
                                Expr::$method($( $slot, )+ offsets, lhs_index),
                                Expr::$method($( $slot, )+ offsets, rhs_index),
                            ) {
                                accumulator.store(rhs_index as u32);
                            }
                        } else {
                            accumulator.store(rhs_index as u32);
                            valid.store(1u32);
                        }
                    }
                }
            }

            let offset = RuntimeCell::<u32>::new(PLANE_DIM / 2u32);
            while offset.read() > 0u32 {
                let rhs_index = plane_shuffle_down(accumulator.read(), offset.read());
                let rhs_valid = plane_shuffle_down(valid.read(), offset.read());
                if UNIT_POS_PLANE < offset.read() && rhs_valid != 0u32 {
                    if valid.read() != 0u32 {
                        let lhs_index = accumulator.read();
                        if lhs_index < rhs_index {
                            if Op::rhs_wins(
                                Expr::$method($( $slot, )+ offsets, lhs_index as usize),
                                Expr::$method($( $slot, )+ offsets, rhs_index as usize),
                            ) {
                                accumulator.store(rhs_index);
                            }
                        } else if !Op::rhs_wins(
                                Expr::$method($( $slot, )+ offsets, rhs_index as usize),
                                Expr::$method($( $slot, )+ offsets, lhs_index as usize),
                            ) {
                            accumulator.store(rhs_index);
                        }
                    } else {
                        accumulator.store(rhs_index);
                        valid.store(1u32);
                    }
                }
                offset.store(offset.read() / 2u32);
            }

            let mut plane_indices = Shared::<[u32]>::new_slice(cube_dim);
            let mut plane_valid = Shared::<[u32]>::new_slice(cube_dim);
            if UNIT_POS_PLANE == 0u32 {
                plane_indices[PLANE_POS as usize] = accumulator.read();
                plane_valid[PLANE_POS as usize] = valid.read();
            }
            sync_cube();

            if PLANE_POS == 0u32 {
                let plane_count = CUBE_DIM.div_ceil(PLANE_DIM);
                let source = if UNIT_POS_PLANE < plane_count {
                    UNIT_POS_PLANE as usize
                } else {
                    0usize
                };
                accumulator.store(plane_indices[source]);
                valid.store(if UNIT_POS_PLANE < plane_count {
                    plane_valid[source]
                } else {
                    0u32
                });
                let plane_cursor = RuntimeCell::<u32>::new(UNIT_POS_PLANE + PLANE_DIM);
                while plane_cursor.read() < plane_count {
                    let source = plane_cursor.read() as usize;
                    if valid.read() != 0u32 && plane_valid[source] != 0u32 {
                        let lhs_index = accumulator.read();
                        let rhs_index = plane_indices[source];
                        if lhs_index < rhs_index {
                            if Op::rhs_wins(
                                Expr::$method($( $slot, )+ offsets, lhs_index as usize),
                                Expr::$method($( $slot, )+ offsets, rhs_index as usize),
                            ) {
                                accumulator.store(rhs_index);
                            }
                        } else if !Op::rhs_wins(
                            Expr::$method($( $slot, )+ offsets, rhs_index as usize),
                            Expr::$method($( $slot, )+ offsets, lhs_index as usize),
                        ) {
                            accumulator.store(rhs_index);
                        }
                    }
                    plane_cursor.store(plane_cursor.read() + PLANE_DIM);
                }
                let plane_offset = RuntimeCell::<u32>::new(PLANE_DIM / 2u32);
                while plane_offset.read() > 0u32 {
                    let rhs_index = plane_shuffle_down(accumulator.read(), plane_offset.read());
                    let rhs_valid = plane_shuffle_down(valid.read(), plane_offset.read());
                    if UNIT_POS_PLANE < plane_offset.read() && rhs_valid != 0u32 {
                        if valid.read() != 0u32 {
                            let lhs_index = accumulator.read();
                            if lhs_index < rhs_index {
                                if Op::rhs_wins(
                                    Expr::$method($( $slot, )+ offsets, lhs_index as usize),
                                    Expr::$method($( $slot, )+ offsets, rhs_index as usize),
                                ) {
                                    accumulator.store(rhs_index);
                                }
                            } else if !Op::rhs_wins(
                                    Expr::$method($( $slot, )+ offsets, rhs_index as usize),
                                    Expr::$method($( $slot, )+ offsets, lhs_index as usize),
                                ) {
                                accumulator.store(rhs_index);
                            }
                        } else {
                            accumulator.store(rhs_index);
                            valid.store(1u32);
                        }
                    }
                    plane_offset.store(plane_offset.read() / 2u32);
                }
                if UNIT_POS_PLANE == 0u32 {
                    partials[CUBE_POS as usize] = if valid.read() != 0u32 {
                        accumulator.read()
                    } else {
                        u32::MAX
                    };
                }
            }
        }
    };
}

macro_rules! define_arg_partial_kernel {
    ($name:ident,$eval:ident,$method:ident; [$( $leaf:ident:$slot:ident ),+]) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $name<
            Item: CubeType + Send + Sync + 'static,
            $( $leaf: CubePrimitive + cubecl::frontend::Scalar, )+
            Expr: $eval<Item, $( $leaf ),+>,
            Op: ArgReductionOp<Item>,
        >(
            $( $slot: &[$leaf], )+
            offsets: &[u32],
            input: &[u32],
            len: &[u32],
            partials: &mut [u32],
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = BLOCK_SIZE as usize;
            let logical_len = len[0] as usize;
            if CUBE_POS as usize >= crate::core::launch::logical_block_count(logical_len, TILE_SIZE) {
                if UNIT_POS == 0u32 && (CUBE_POS as usize) < partials.len() {
                    partials[CUBE_POS as usize] = u32::MAX;
                }
                terminate!();
            }
            let tile_start = CUBE_POS as usize * TILE_SIZE;
            let first_position = tile_start + unit;
            let safe_position = if first_position < logical_len {
                first_position
            } else {
                logical_len - 1usize
            };
            let accumulator = RuntimeCell::<u32>::new(input[safe_position]);
            let valid = RuntimeCell::<u32>::new(
                if first_position < logical_len { 1u32 } else { 0u32 },
            );

            for item in 1usize..ITEMS_PER_UNIT {
                let rhs_position = first_position + item * cube_dim;
                if rhs_position < logical_len {
                    let lhs_index = accumulator.read();
                    let rhs_index = input[rhs_position];
                    if lhs_index < rhs_index {
                        if Op::rhs_wins(
                            Expr::$method($( $slot, )+ offsets, lhs_index as usize),
                            Expr::$method($( $slot, )+ offsets, rhs_index as usize),
                        ) {
                            accumulator.store(rhs_index);
                        }
                    } else if !Op::rhs_wins(
                            Expr::$method($( $slot, )+ offsets, rhs_index as usize),
                            Expr::$method($( $slot, )+ offsets, lhs_index as usize),
                        ) {
                        accumulator.store(rhs_index);
                    }
                }
            }

            let offset = RuntimeCell::<u32>::new(PLANE_DIM / 2u32);
            while offset.read() > 0u32 {
                let rhs_index = plane_shuffle_down(accumulator.read(), offset.read());
                let rhs_valid = plane_shuffle_down(valid.read(), offset.read());
                if UNIT_POS_PLANE < offset.read() && rhs_valid != 0u32 {
                    if valid.read() != 0u32 {
                        let lhs_index = accumulator.read();
                        if lhs_index < rhs_index {
                            if Op::rhs_wins(
                                Expr::$method($( $slot, )+ offsets, lhs_index as usize),
                                Expr::$method($( $slot, )+ offsets, rhs_index as usize),
                            ) {
                                accumulator.store(rhs_index);
                            }
                        } else if !Op::rhs_wins(
                                Expr::$method($( $slot, )+ offsets, rhs_index as usize),
                                Expr::$method($( $slot, )+ offsets, lhs_index as usize),
                            ) {
                            accumulator.store(rhs_index);
                        }
                    } else {
                        accumulator.store(rhs_index);
                        valid.store(1u32);
                    }
                }
                offset.store(offset.read() / 2u32);
            }

            let mut plane_indices = Shared::<[u32]>::new_slice(cube_dim);
            let mut plane_valid = Shared::<[u32]>::new_slice(cube_dim);
            if UNIT_POS_PLANE == 0u32 {
                plane_indices[PLANE_POS as usize] = accumulator.read();
                plane_valid[PLANE_POS as usize] = valid.read();
            }
            sync_cube();

            if PLANE_POS == 0u32 {
                let plane_count = CUBE_DIM.div_ceil(PLANE_DIM);
                let source = if UNIT_POS_PLANE < plane_count {
                    UNIT_POS_PLANE as usize
                } else {
                    0usize
                };
                accumulator.store(plane_indices[source]);
                valid.store(if UNIT_POS_PLANE < plane_count {
                    plane_valid[source]
                } else {
                    0u32
                });
                let plane_cursor = RuntimeCell::<u32>::new(UNIT_POS_PLANE + PLANE_DIM);
                while plane_cursor.read() < plane_count {
                    let source = plane_cursor.read() as usize;
                    if valid.read() != 0u32 && plane_valid[source] != 0u32 {
                        let lhs_index = accumulator.read();
                        let rhs_index = plane_indices[source];
                        if lhs_index < rhs_index {
                            if Op::rhs_wins(
                                Expr::$method($( $slot, )+ offsets, lhs_index as usize),
                                Expr::$method($( $slot, )+ offsets, rhs_index as usize),
                            ) {
                                accumulator.store(rhs_index);
                            }
                        } else if !Op::rhs_wins(
                            Expr::$method($( $slot, )+ offsets, rhs_index as usize),
                            Expr::$method($( $slot, )+ offsets, lhs_index as usize),
                        ) {
                            accumulator.store(rhs_index);
                        }
                    }
                    plane_cursor.store(plane_cursor.read() + PLANE_DIM);
                }
                let plane_offset = RuntimeCell::<u32>::new(PLANE_DIM / 2u32);
                while plane_offset.read() > 0u32 {
                    let rhs_index = plane_shuffle_down(accumulator.read(), plane_offset.read());
                    let rhs_valid = plane_shuffle_down(valid.read(), plane_offset.read());
                    if UNIT_POS_PLANE < plane_offset.read() && rhs_valid != 0u32 {
                        if valid.read() != 0u32 {
                            let lhs_index = accumulator.read();
                            if lhs_index < rhs_index {
                                if Op::rhs_wins(
                                    Expr::$method($( $slot, )+ offsets, lhs_index as usize),
                                    Expr::$method($( $slot, )+ offsets, rhs_index as usize),
                                ) {
                                    accumulator.store(rhs_index);
                                }
                            } else if !Op::rhs_wins(
                                    Expr::$method($( $slot, )+ offsets, rhs_index as usize),
                                    Expr::$method($( $slot, )+ offsets, lhs_index as usize),
                                ) {
                                accumulator.store(rhs_index);
                            }
                        } else {
                            accumulator.store(rhs_index);
                            valid.store(1u32);
                        }
                    }
                    plane_offset.store(plane_offset.read() / 2u32);
                }
                if UNIT_POS_PLANE == 0u32 {
                    partials[CUBE_POS as usize] = if valid.read() != 0u32 {
                        accumulator.read()
                    } else {
                        u32::MAX
                    };
                }
            }
        }
    };
}

define_arg_reduce_kernel!(arg_reduce_a13,Eval13,eval13; [L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6,L7:slot7,L8:slot8,L9:slot9,L10:slot10,L11:slot11,L12:slot12]);

define_arg_partial_kernel!(arg_partial_a13,Eval13,eval13; [L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6,L7:slot7,L8:slot8,L9:slot9,L10:slot10,L11:slot11,L12:slot12]);

/// Arity dispatch for an index-sideband reduction.
#[doc(hidden)]
pub trait ArgReduceDispatch<R: Runtime, Input, Op, Slots> {
    fn execute(exec: &Executor<R>, input: &Input) -> Result<DeviceVec<R, u32>, Error>;
}

fn block_count(len: usize) -> usize {
    len.div_ceil(TILE_SIZE).max(1)
}

macro_rules! impl_arg_reduce_dispatch {
    ($arity:ty,$eval:ident,$first:ident,$partial:ident,$env:ty; [$( $leaf:ident:$index:literal ),+]) => {
        impl<R, Input, Item, Op, $( $leaf ),+> ArgReduceDispatch<R, Input, Op, $env>
            for Dispatch<$arity, crate::core::storage::S1>
        where
            R: Runtime,
            Item: CubeType + Send + Sync + 'static,
            Op: ArgReductionOp<Item>,
            $( $leaf: MStorageElement, )+
            Input: ReadExpression<Item = Item> + LowerReadExpression + StageRead<R, Env0>,
            Input::Slots: PaddedReadSlots<
                L0 = L0, L1 = L1, L2 = L2, L3 = L3, L4 = L4, L5 = L5, L6 = L6,
                L7 = L7, L8 = L8, L9 = L9, L10 = L10, L11 = L11, L12 = L12,
            >,
            Input::DeviceExpr: $eval<Item, $( $leaf ),+>,
        {
            fn execute(exec: &Executor<R>, input: &Input) -> Result<DeviceVec<R, u32>, Error> {
                let capacity = input.physical_len()?;
                if capacity == 0 {
                    return Ok(exec.to_device(&[u32::MAX]));
                }
                let extent = input.logical_extent()?;
                let client = exec.client();
                let bindings = Bindings::read(exec, input)?;
                let offsets = client.create_from_slice(u32::as_bytes(&bindings.offsets));
                let len_handle = extent.materialize(exec)?;
                let blocks = block_count(capacity);
                let mut current_extent = extent.ceil_div(exec, TILE_SIZE, blocks)?;
                let mut current = exec.alloc_row::<u32>(blocks);
                current.set_logical_extent(current_extent.clone());
                unsafe {
                    $first::launch_unchecked::<Item, $( $leaf, )+ Input::DeviceExpr, Op, R>(
                        client,
                        cube_count_1d(blocks)?,
                        CubeDim::new_1d(BLOCK_SIZE),
                        $( BufferArg::from_raw_parts(bindings.slots[$index].0.clone(), bindings.slots[$index].1), )+
                        BufferArg::from_raw_parts(offsets.clone(), bindings.offsets.len()),
                        BufferArg::from_raw_parts(len_handle.handle.clone(), 1),
                        BufferArg::from_raw_parts(current.handle.clone(), blocks),
                    );
                }

                let mut current_len = blocks;
                while current_len > 1 {
                    let next_len = block_count(current_len);
                    let next_extent = current_extent.ceil_div(exec, TILE_SIZE, next_len)?;
                    let mut next = exec.alloc_row::<u32>(next_len);
                    next.set_logical_extent(next_extent.clone());
                    let current_len_handle = current_extent.materialize(exec)?;
                    unsafe {
                        $partial::launch_unchecked::<Item, $( $leaf, )+ Input::DeviceExpr, Op, R>(
                            client,
                            cube_count_1d(next_len)?,
                            CubeDim::new_1d(BLOCK_SIZE),
                            $( BufferArg::from_raw_parts(bindings.slots[$index].0.clone(), bindings.slots[$index].1), )+
                            BufferArg::from_raw_parts(offsets.clone(), bindings.offsets.len()),
                            BufferArg::from_raw_parts(current.handle.clone(), current_len),
                            BufferArg::from_raw_parts(current_len_handle.handle.clone(), 1),
                            BufferArg::from_raw_parts(next.handle.clone(), next_len),
                        );
                    }
                    current = next;
                    current_len = next_len;
                    current_extent = next_extent;
                }

                Ok(exec.column_from_handle(current.handle.clone(), 1))
            }
        }
    };
}

impl_arg_reduce_dispatch!(A13,Eval13,arg_reduce_a13,arg_partial_a13,Env13<L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12>; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6,L7:7,L8:8,L9:9,L10:10,L11:11,L12:12]);

/// Returns the selected input index, or `u32::MAX` for an empty input.
pub(crate) fn arg_reduce<R, Input, Op>(
    exec: &Executor<R>,
    input: Input,
    _op: Op,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Input: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Op: ArgReductionOp<Input::Item>,
    Dispatch<A13, crate::core::storage::S1>:
        ArgReduceDispatch<R, Input, Op, KernelReadSlots<Input::Slots>>,
{
    <Dispatch<A13, crate::core::storage::S1> as ArgReduceDispatch<
        R,
        Input,
        Op,
        KernelReadSlots<Input::Slots>,
    >>::execute(exec, &input)
}
