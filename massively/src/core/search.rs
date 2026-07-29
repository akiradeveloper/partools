//! Search controls over independently lowered fixed-ABI inputs.

use core::marker::PhantomData;

use cubecl::prelude::*;

use crate::core::arity::A13;
use crate::core::bindings::Bindings;
use crate::core::eval::Eval13At;
use crate::core::op::{BinaryPredicateOp, PredicateOp};
use crate::core::read::{Env0, Env13, LowerReadExpression, ReadExpression, StageRead};
use crate::core::value::MStorageElement;
use crate::{DeviceVec, Error, Executor, MVec};

const BLOCK_SIZE: u32 = 256;

struct NonZero;

#[cubecl::cube]
impl PredicateOp<u32> for NonZero {
    fn apply(input: u32) -> crate::MFlag {
        crate::flag::from_bool(input != 0u32)
    }
}

#[cubecl::cube(launch_unchecked, explicit_define)]
fn prepare_one_length(length: &[u32], control: &mut [u32]) {
    if ABSOLUTE_POS == 0usize {
        let metadata = control.len() - 1usize;
        control[metadata] = length[0];
    }
}

#[cubecl::cube(launch_unchecked, explicit_define)]
fn prepare_two_lengths(first: &[u32], second: &[u32], control: &mut [u32]) {
    if ABSOLUTE_POS == 0usize {
        let metadata = control.len() - 2usize;
        control[metadata] = first[0];
        control[metadata + 1usize] = second[0];
    }
}

#[cubecl::cube(launch_unchecked)]
fn clamp_first_to_len(first: &[u32], len: &[u32], output: &mut [u32]) {
    if ABSOLUTE_POS == 0usize {
        output[0] = u32::min(first[0], len[0]);
    }
}

#[cubecl::cube]
trait PairCodeOp<Item: CubeType>: 'static + Send + Sync {
    fn code(left: Item, right: Item, left_again: Item, right_again: Item) -> u32;
}

struct MismatchCode<Equal>(PhantomData<fn() -> Equal>);

#[cubecl::cube]
impl<Item, Equal> PairCodeOp<Item> for MismatchCode<Equal>
where
    Item: CubeType + 'static,
    Equal: BinaryPredicateOp<Item>,
{
    fn code(left: Item, right: Item, _left_again: Item, _right_again: Item) -> u32 {
        if crate::core::ordering::binary_predicate::<Item, Equal>(left, right) {
            0u32
        } else {
            1u32
        }
    }
}

struct LexicographicalCode<Less>(PhantomData<fn() -> Less>);

#[cubecl::cube]
impl<Item, Less> PairCodeOp<Item> for LexicographicalCode<Less>
where
    Item: CubeType + 'static,
    Less: BinaryPredicateOp<Item>,
{
    fn code(left: Item, right: Item, left_again: Item, right_again: Item) -> u32 {
        if crate::core::ordering::binary_predicate::<Item, Less>(left, right) {
            1u32
        } else if crate::core::ordering::binary_predicate::<Item, Less>(right_again, left_again) {
            2u32
        } else {
            0u32
        }
    }
}

macro_rules! define_pair_code_kernel {
    (
        $name:ident;
        [$( $left_leaf:ident:$left_slot:ident ),+];
        [$( $right_leaf:ident:$right_slot:ident ),+]
    ) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $name<
            Item: CubeType + Send + Sync + 'static,
            $( $left_leaf: CubePrimitive + cubecl::frontend::Scalar, )+
            $( $right_leaf: CubePrimitive + cubecl::frontend::Scalar, )+
            Left: Eval13At<Item, $( $left_leaf ),+>,
            Right: Eval13At<Item, $( $right_leaf ),+>,
            Op: PairCodeOp<Item>,
        >(
            $( $left_slot: &[$left_leaf], )+
            $( $right_slot: &[$right_leaf], )+
            offsets: &[u32],
            control: &mut [u32],
        ) {
            let index = ABSOLUTE_POS as usize;
            let metadata = control.len() - 1usize;
            if index < control[metadata] as usize {
                control[index] = Op::code(
                    Left::eval13_at($( $left_slot, )+ offsets, 0usize, index),
                    Right::eval13_at($( $right_slot, )+ offsets, 13usize, index),
                    Left::eval13_at($( $left_slot, )+ offsets, 0usize, index),
                    Right::eval13_at($( $right_slot, )+ offsets, 13usize, index),
                );
            }
        }
    };
}

define_pair_code_kernel!(
    pair_code_a13;
    [L0:left0,L1:left1,L2:left2,L3:left3,L4:left4,L5:left5,L6:left6,L7:left7,L8:left8,L9:left9,L10:left10,L11:left11,L12:left12];
    [R0:right0,R1:right1,R2:right2,R3:right3,R4:right4,R5:right5,R6:right6,R7:right7,R8:right8,R9:right9,R10:right10,R11:right11,R12:right12]
);

#[cubecl::cube(launch_unchecked)]
fn lexicographical_result_kernel(
    codes: &[u32],
    best: &[u32],
    shared_len: &[u32],
    left_is_shorter: &[u32],
    output: &mut [u32],
) {
    if ABSOLUTE_POS == 0 {
        let index = best[0];
        output[0] = if index < shared_len[0] {
            crate::flag::from_bool(codes[index as usize] == 1u32)
        } else {
            crate::flag::from_bool(left_is_shorter[0] != 0u32)
        };
    }
}

macro_rules! define_find_first_of_kernel {
    (
        $name:ident;
        [$( $source_leaf:ident:$source_slot:ident ),+];
        [$( $needle_leaf:ident:$needle_slot:ident ),+]
    ) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $name<
            Item: CubeType + Send + Sync + 'static,
            $( $source_leaf: CubePrimitive + cubecl::frontend::Scalar, )+
            $( $needle_leaf: CubePrimitive + cubecl::frontend::Scalar, )+
            Source: Eval13At<Item, $( $source_leaf ),+>,
            Needles: Eval13At<Item, $( $needle_leaf ),+>,
            Equal: BinaryPredicateOp<Item>,
        >(
            $( $source_slot: &[$source_leaf], )+
            $( $needle_slot: &[$needle_leaf], )+
            offsets: &[u32],
            control: &mut [u32],
        ) {
            let index = ABSOLUTE_POS as usize;
            let metadata = control.len() - 2usize;
            let source_len = control[metadata] as usize;
            let needle_len = control[metadata + 1usize] as usize;
            if index < source_len {
                let found = RuntimeCell::<u32>::new(0u32);
                let needle = RuntimeCell::<usize>::new(0usize);
                while needle.read() < needle_len {
                    if crate::core::ordering::binary_predicate::<Item, Equal>(
                        Source::eval13_at($( $source_slot, )+ offsets, 0usize, index),
                        Needles::eval13_at(
                            $( $needle_slot, )+ offsets, 13usize, needle.read(),
                        ),
                    ) {
                        found.store(1u32);
                        needle.store(needle_len);
                    } else {
                        needle.store(needle.read() + 1usize);
                    }
                }
                control[index] = found.read();
            }
        }
    };
}

define_find_first_of_kernel!(
    find_first_of_a13;
    [L0:source0,L1:source1,L2:source2,L3:source3,L4:source4,L5:source5,L6:source6,L7:source7,L8:source8,L9:source9,L10:source10,L11:source11,L12:source12];
    [R0:needle0,R1:needle1,R2:needle2,R3:needle3,R4:needle4,R5:needle5,R6:needle6,R7:needle7,R8:needle8,R9:needle9,R10:needle10,R11:needle11,R12:needle12]
);

#[cubecl::cube]
trait BoundOp<Item: CubeType>: 'static + Send + Sync {
    fn go_right(candidate: Item, value: Item) -> bool;
}

struct LowerBound<Less>(PhantomData<fn() -> Less>);

#[cubecl::cube]
impl<Item, Less> BoundOp<Item> for LowerBound<Less>
where
    Item: CubeType + 'static,
    Less: BinaryPredicateOp<Item>,
{
    fn go_right(candidate: Item, value: Item) -> bool {
        crate::core::ordering::binary_predicate::<Item, Less>(candidate, value)
    }
}

struct UpperBound<Less>(PhantomData<fn() -> Less>);

#[cubecl::cube]
impl<Item, Less> BoundOp<Item> for UpperBound<Less>
where
    Item: CubeType + 'static,
    Less: BinaryPredicateOp<Item>,
{
    fn go_right(candidate: Item, value: Item) -> bool {
        !crate::core::ordering::binary_predicate::<Item, Less>(value, candidate)
    }
}

macro_rules! define_bound_kernel {
    (
        $name:ident;
        [$( $source_leaf:ident:$source_slot:ident ),+];
        [$( $value_leaf:ident:$value_slot:ident ),+]
    ) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $name<
            Item: CubeType + Send + Sync + 'static,
            $( $source_leaf: CubePrimitive + cubecl::frontend::Scalar, )+
            $( $value_leaf: CubePrimitive + cubecl::frontend::Scalar, )+
            Source: Eval13At<Item, $( $source_leaf ),+>,
            Values: Eval13At<Item, $( $value_leaf ),+>,
            Op: BoundOp<Item>,
        >(
            $( $source_slot: &[$source_leaf], )+
            $( $value_slot: &[$value_leaf], )+
            offsets: &[u32],
            control: &mut [u32],
        ) {
            let index = ABSOLUTE_POS as usize;
            let metadata = control.len() - 2usize;
            let source_len = control[metadata] as usize;
            if index < control[metadata + 1usize] as usize {
                let low = RuntimeCell::<usize>::new(0usize);
                let high = RuntimeCell::<usize>::new(source_len);
                while low.read() < high.read() {
                    let mid = low.read() + (high.read() - low.read()) / 2usize;
                    if Op::go_right(
                        Source::eval13_at($( $source_slot, )+ offsets, 0usize, mid),
                        Values::eval13_at($( $value_slot, )+ offsets, 13usize, index),
                    ) {
                        low.store(mid + 1usize);
                    } else {
                        high.store(mid);
                    }
                }
                control[index] = low.read() as u32;
            }
        }
    };
}

define_bound_kernel!(
    bound_a13;
    [L0:source0,L1:source1,L2:source2,L3:source3,L4:source4,L5:source5,L6:source6,L7:source7,L8:source8,L9:source9,L10:source10,L11:source11,L12:source12];
    [R0:value0,R1:value1,R2:value2,R3:value3,R4:value4,R5:value5,R6:value6,R7:value7,R8:value8,R9:value9,R10:value10,R11:value11,R12:value12]
);

struct PairDispatch<Storage>(PhantomData<fn() -> Storage>);

trait PairCodeDispatch<R, Left, Right, Item, LeftSlots, RightSlots, Op>
where
    R: Runtime,
{
    fn run(
        exec: &Executor<R>,
        left: &Left,
        right: &Right,
    ) -> Result<
        (
            DeviceVec<R, u32>,
            DeviceVec<R, u32>,
            crate::core::extent::LogicalExtent,
            crate::core::extent::LogicalExtent,
        ),
        Error,
    >;
}

macro_rules! impl_pair_code_dispatch {
    (
        $storage:ty,$arity:ty,$kernel:ident;
        $left_env:ty => [$( $left_leaf:ident:$left_index:literal ),+];
        $right_env:ty => [$( $right_leaf:ident:$right_index:literal ),+]
    ) => {
        impl<R, Left, Right, Item, Op, $( $left_leaf, )+ $( $right_leaf ),+>
            PairCodeDispatch<R, Left, Right, Item, $left_env, $right_env, Op>
            for PairDispatch<$storage>
        where
            R: Runtime,
            Item: CubeType + Send + Sync + 'static,
            Op: PairCodeOp<Item>,
            $( $left_leaf: MStorageElement, )+
            $( $right_leaf: MStorageElement, )+
            Left: ReadExpression<Item = Item, ReadArity = $arity>
                + LowerReadExpression<Slots = $left_env>
                + StageRead<R, Env0>,
            Right: ReadExpression<Item = Item, ReadArity = $arity>
                + LowerReadExpression<Slots = $right_env>
                + StageRead<R, Env0>,
            Left::DeviceExpr: Eval13At<Item, $( $left_leaf ),+>,
            Right::DeviceExpr: Eval13At<Item, $( $right_leaf ),+>,
        {
            fn run(
                exec: &Executor<R>,
                left: &Left,
                right: &Right,
            ) -> Result<
                (
                    DeviceVec<R, u32>,
                    DeviceVec<R, u32>,
                    crate::core::extent::LogicalExtent,
                    crate::core::extent::LogicalExtent,
                ),
                Error,
            > {
                let left_capacity = left.physical_len()?;
                let right_capacity = right.physical_len()?;
                let capacity = left_capacity.min(right_capacity);
                let left_extent = left.logical_extent()?;
                let right_extent = right.logical_extent()?;
                let shared_extent = crate::core::extent::LogicalExtent::min(
                    exec,
                    &left_extent,
                    &right_extent,
                )?;
                if capacity == 0 {
                    let mut codes = exec.alloc_row::<u32>(0);
                    codes.set_logical_extent(shared_extent.clone());
                    let best = shared_extent.copy_value(exec)?;
                    return Ok((codes, best, left_extent, right_extent));
                }
                let left_bindings = Bindings::read(exec, left)?;
                let right_bindings = Bindings::read(exec, right)?;
                let mut combined_offsets = left_bindings.offsets.clone();
                combined_offsets.extend_from_slice(&right_bindings.offsets);
                let offsets = exec.client().create_from_slice(u32::as_bytes(&combined_offsets));
                let len_handle = shared_extent.materialize(exec)?;
                let control_len = capacity
                    .checked_add(1)
                    .ok_or(Error::LengthTooLarge { len: capacity })?;
                let control = exec.alloc_column::<u32>(control_len);
                let mut codes = exec.column_from_handle::<u32>(control.handle.clone(), capacity);
                codes.set_logical_extent(shared_extent.clone());
                unsafe {
                    prepare_one_length::launch_unchecked::<R>(
                        exec.client(),
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new_1d(1),
                        BufferArg::from_raw_parts(len_handle.handle.clone(), 1),
                        BufferArg::from_raw_parts(control.handle.clone(), control_len),
                    );
                    $kernel::launch_unchecked::<
                        Item,
                        $( $left_leaf, )+
                        $( $right_leaf, )+
                        Left::DeviceExpr,
                        Right::DeviceExpr,
                        Op,
                        R,
                    >(
                        exec.client(),
                        crate::core::launch::cube_count_1d(capacity.div_ceil(BLOCK_SIZE as usize))?,
                        CubeDim::new_1d(BLOCK_SIZE),
                        $( BufferArg::from_raw_parts(left_bindings.slots[$left_index].0.clone(), left_bindings.slots[$left_index].1), )+
                        $( BufferArg::from_raw_parts(right_bindings.slots[$right_index].0.clone(), right_bindings.slots[$right_index].1), )+
                        BufferArg::from_raw_parts(offsets, combined_offsets.len()),
                        BufferArg::from_raw_parts(control.handle.clone(), control_len),
                    );
                }
                let first = crate::core::predicate::find_if(exec, codes.column(), NonZero)?;
                let best = exec.alloc_column::<u32>(1);
                unsafe {
                    clamp_first_to_len::launch_unchecked::<R>(
                        exec.client(),
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new_1d(1),
                        BufferArg::from_raw_parts(first.handle.clone(), 1),
                        BufferArg::from_raw_parts(len_handle.handle.clone(), 1),
                        BufferArg::from_raw_parts(best.handle.clone(), 1),
                    );
                }
                Ok((codes, best, left_extent, right_extent))
            }
        }
    };
}

impl_pair_code_dispatch!(
    crate::core::storage::S12, A13, pair_code_a13;
    Env13<L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12>
        => [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6,L7:7,L8:8,L9:9,L10:10,L11:11,L12:12];
    Env13<R0,R1,R2,R3,R4,R5,R6,R7,R8,R9,R10,R11,R12>
        => [R0:0,R1:1,R2:2,R3:3,R4:4,R5:5,R6:6,R7:7,R8:8,R9:9,R10:10,R11:11,R12:12]
);

trait FindFirstDispatch<R, Source, Needles, Item, SourceSlots, NeedleSlots, Equal>
where
    R: Runtime,
{
    fn run(
        exec: &Executor<R>,
        source: &Source,
        needles: &Needles,
    ) -> Result<DeviceVec<R, u32>, Error>;
}

trait BoundDispatch<R, Source, Values, Item, SourceSlots, ValueSlots, Op>
where
    R: Runtime,
{
    fn run(
        exec: &Executor<R>,
        source: &Source,
        values: &Values,
    ) -> Result<DeviceVec<R, u32>, Error>;
}

macro_rules! impl_range_query_dispatch {
    (
        $storage:ty,$arity:ty,$find_kernel:ident,$bound_kernel:ident;
        $source_env:ty => [$( $source_leaf:ident:$source_index:literal ),+];
        $other_env:ty => [$( $other_leaf:ident:$other_index:literal ),+]
    ) => {
        impl<R, Source, Needles, Item, Equal, $( $source_leaf, )+ $( $other_leaf ),+>
            FindFirstDispatch<R, Source, Needles, Item, $source_env, $other_env, Equal>
            for PairDispatch<$storage>
        where
            R: Runtime,
            Item: CubeType + Send + Sync + 'static,
            Equal: BinaryPredicateOp<Item>,
            $( $source_leaf: MStorageElement, )+
            $( $other_leaf: MStorageElement, )+
            Source: ReadExpression<Item = Item, ReadArity = $arity>
                + LowerReadExpression<Slots = $source_env>
                + StageRead<R, Env0>,
            Needles: ReadExpression<Item = Item, ReadArity = $arity>
                + LowerReadExpression<Slots = $other_env>
                + StageRead<R, Env0>,
            Source::DeviceExpr: Eval13At<Item, $( $source_leaf ),+>,
            Needles::DeviceExpr: Eval13At<Item, $( $other_leaf ),+>,
        {
            fn run(
                exec: &Executor<R>,
                source: &Source,
                needles: &Needles,
            ) -> Result<DeviceVec<R, u32>, Error> {
                let source_capacity = source.physical_len()?;
                let needle_capacity = needles.physical_len()?;
                let best = exec.to_device(&[u32::MAX]);
                if source_capacity == 0 || needle_capacity == 0 {
                    return Ok(best);
                }
                let source_bindings = Bindings::read(exec, source)?;
                let needle_bindings = Bindings::read(exec, needles)?;
                let mut combined_offsets = source_bindings.offsets.clone();
                combined_offsets.extend_from_slice(&needle_bindings.offsets);
                let offsets = exec.client().create_from_slice(u32::as_bytes(&combined_offsets));
                let source_len_handle = source.logical_extent()?.materialize(exec)?;
                let needle_len_handle = needles.logical_extent()?.materialize(exec)?;
                let control_len = source_capacity
                    .checked_add(2)
                    .ok_or(Error::LengthTooLarge { len: source_capacity })?;
                let control = exec.alloc_column::<u32>(control_len);
                let mut flags = exec.column_from_handle::<u32>(control.handle.clone(), source_capacity);
                flags.set_logical_extent(source.logical_extent()?);
                unsafe {
                    prepare_two_lengths::launch_unchecked::<R>(
                        exec.client(),
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new_1d(1),
                        BufferArg::from_raw_parts(source_len_handle.handle.clone(), 1),
                        BufferArg::from_raw_parts(needle_len_handle.handle.clone(), 1),
                        BufferArg::from_raw_parts(control.handle.clone(), control_len),
                    );
                    $find_kernel::launch_unchecked::<
                        Item,
                        $( $source_leaf, )+
                        $( $other_leaf, )+
                        Source::DeviceExpr,
                        Needles::DeviceExpr,
                        Equal,
                        R,
                    >(
                        exec.client(),
                        crate::core::launch::cube_count_1d(source_capacity.div_ceil(BLOCK_SIZE as usize))?,
                        CubeDim::new_1d(BLOCK_SIZE),
                        $( BufferArg::from_raw_parts(source_bindings.slots[$source_index].0.clone(), source_bindings.slots[$source_index].1), )+
                        $( BufferArg::from_raw_parts(needle_bindings.slots[$other_index].0.clone(), needle_bindings.slots[$other_index].1), )+
                        BufferArg::from_raw_parts(offsets, combined_offsets.len()),
                        BufferArg::from_raw_parts(control.handle.clone(), control_len),
                    );
                }
                crate::core::predicate::find_if(exec, flags.column(), NonZero)
            }
        }

        impl<R, Source, Values, Item, Op, $( $source_leaf, )+ $( $other_leaf ),+>
            BoundDispatch<R, Source, Values, Item, $source_env, $other_env, Op>
            for PairDispatch<$storage>
        where
            R: Runtime,
            Item: CubeType + Send + Sync + 'static,
            Op: BoundOp<Item>,
            $( $source_leaf: MStorageElement, )+
            $( $other_leaf: MStorageElement, )+
            Source: ReadExpression<Item = Item, ReadArity = $arity>
                + LowerReadExpression<Slots = $source_env>
                + StageRead<R, Env0>,
            Values: ReadExpression<Item = Item, ReadArity = $arity>
                + LowerReadExpression<Slots = $other_env>
                + StageRead<R, Env0>,
            Source::DeviceExpr: Eval13At<Item, $( $source_leaf ),+>,
            Values::DeviceExpr: Eval13At<Item, $( $other_leaf ),+>,
        {
            fn run(
                exec: &Executor<R>,
                source: &Source,
                values: &Values,
            ) -> Result<DeviceVec<R, u32>, Error> {
                let value_capacity = values.physical_len()?;
                let value_extent = values.logical_extent()?;
                if value_capacity == 0 {
                    let mut bounds = exec.alloc_row::<u32>(0);
                    bounds.set_logical_extent(value_extent.clone());
                    return Ok(bounds);
                }
                let source_bindings = Bindings::read(exec, source)?;
                let value_bindings = Bindings::read(exec, values)?;
                let mut combined_offsets = source_bindings.offsets.clone();
                combined_offsets.extend_from_slice(&value_bindings.offsets);
                let offsets = exec.client().create_from_slice(u32::as_bytes(&combined_offsets));
                let source_len_handle = source.logical_extent()?.materialize(exec)?;
                let value_len_handle = value_extent.materialize(exec)?;
                let control_len = value_capacity
                    .checked_add(2)
                    .ok_or(Error::LengthTooLarge { len: value_capacity })?;
                let control = exec.alloc_column::<u32>(control_len);
                let mut bounds = exec.column_from_handle::<u32>(control.handle.clone(), value_capacity);
                bounds.set_logical_extent(value_extent.clone());
                unsafe {
                    prepare_two_lengths::launch_unchecked::<R>(
                        exec.client(),
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new_1d(1),
                        BufferArg::from_raw_parts(source_len_handle.handle.clone(), 1),
                        BufferArg::from_raw_parts(value_len_handle.handle.clone(), 1),
                        BufferArg::from_raw_parts(control.handle.clone(), control_len),
                    );
                    $bound_kernel::launch_unchecked::<
                        Item,
                        $( $source_leaf, )+
                        $( $other_leaf, )+
                        Source::DeviceExpr,
                        Values::DeviceExpr,
                        Op,
                        R,
                    >(
                        exec.client(),
                        crate::core::launch::cube_count_1d(value_capacity.div_ceil(BLOCK_SIZE as usize))?,
                        CubeDim::new_1d(BLOCK_SIZE),
                        $( BufferArg::from_raw_parts(source_bindings.slots[$source_index].0.clone(), source_bindings.slots[$source_index].1), )+
                        $( BufferArg::from_raw_parts(value_bindings.slots[$other_index].0.clone(), value_bindings.slots[$other_index].1), )+
                        BufferArg::from_raw_parts(offsets, combined_offsets.len()),
                        BufferArg::from_raw_parts(control.handle.clone(), control_len),
                    );
                }
                Ok(bounds)
            }
        }
    };
}

impl_range_query_dispatch!(
    crate::core::storage::S12, A13, find_first_of_a13, bound_a13;
    Env13<L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12>
        => [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6,L7:7,L8:8,L9:9,L10:10,L11:11,L12:12];
    Env13<R0,R1,R2,R3,R4,R5,R6,R7,R8,R9,R10,R11,R12>
        => [R0:0,R1:1,R2:2,R3:3,R4:4,R5:5,R6:6,R7:7,R8:8,R9:9,R10:10,R11:11,R12:12]
);

trait PairCodeInput<R: Runtime, Right, Op>: ReadExpression {
    fn pair_codes(
        self,
        exec: &Executor<R>,
        right: Right,
    ) -> Result<
        (
            DeviceVec<R, u32>,
            DeviceVec<R, u32>,
            crate::core::extent::LogicalExtent,
            crate::core::extent::LogicalExtent,
        ),
        Error,
    >;
}

/// Internal public-API capability for `find_first_of`.
#[doc(hidden)]
pub trait FindFirstOfInput<R: Runtime, Needles, Equal>: ReadExpression {
    fn find_first(self, exec: &Executor<R>, needles: Needles) -> Result<DeviceVec<R, u32>, Error>;
}

impl<R, Source, Needles, Equal> FindFirstOfInput<R, Needles, Equal> for Source
where
    R: Runtime,
    Source: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Needles: ReadExpression<Item = Source::Item> + LowerReadExpression + StageRead<R, Env0>,
    PairDispatch<crate::core::storage::S12>:
        FindFirstDispatch<R, Source, Needles, Source::Item, Source::Slots, Needles::Slots, Equal>,
{
    fn find_first(self, exec: &Executor<R>, needles: Needles) -> Result<DeviceVec<R, u32>, Error> {
        <PairDispatch<crate::core::storage::S12> as FindFirstDispatch<
            R,
            Source,
            Needles,
            Source::Item,
            Source::Slots,
            Needles::Slots,
            Equal,
        >>::run(exec, &self, &needles)
    }
}

/// Finds the first source item equal to any needle.
pub(crate) fn find_first_of<R, Source, Needles, Equal>(
    exec: &Executor<R>,
    source: Source,
    needles: Needles,
    _equal: Equal,
) -> Result<MVec<R, u32>, Error>
where
    R: Runtime,
    Source: FindFirstOfInput<R, Needles, Equal>,
{
    source.find_first(exec, needles)
}

/// Internal public-API capability for batched sorted bounds.
#[doc(hidden)]
pub trait SortedBoundsInput<R: Runtime, Values, Less>: ReadExpression {
    fn lower_bounds(self, exec: &Executor<R>, values: Values) -> Result<DeviceVec<R, u32>, Error>;
    fn upper_bounds(self, exec: &Executor<R>, values: Values) -> Result<DeviceVec<R, u32>, Error>;
}

impl<R, Source, Values, Less> SortedBoundsInput<R, Values, Less> for Source
where
    R: Runtime,
    Source: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Values: ReadExpression<Item = Source::Item> + LowerReadExpression + StageRead<R, Env0>,
    PairDispatch<crate::core::storage::S12>: BoundDispatch<
            R,
            Source,
            Values,
            Source::Item,
            Source::Slots,
            Values::Slots,
            LowerBound<Less>,
        > + BoundDispatch<
            R,
            Source,
            Values,
            Source::Item,
            Source::Slots,
            Values::Slots,
            UpperBound<Less>,
        >,
{
    fn lower_bounds(self, exec: &Executor<R>, values: Values) -> Result<DeviceVec<R, u32>, Error> {
        <PairDispatch<crate::core::storage::S12> as BoundDispatch<
            R,
            Source,
            Values,
            Source::Item,
            Source::Slots,
            Values::Slots,
            LowerBound<Less>,
        >>::run(exec, &self, &values)
    }

    fn upper_bounds(self, exec: &Executor<R>, values: Values) -> Result<DeviceVec<R, u32>, Error> {
        <PairDispatch<crate::core::storage::S12> as BoundDispatch<
            R,
            Source,
            Values,
            Source::Item,
            Source::Slots,
            Values::Slots,
            UpperBound<Less>,
        >>::run(exec, &self, &values)
    }
}

pub(crate) fn lower_bounds_storage<R, Source, Values, Less>(
    exec: &Executor<R>,
    source: Source,
    values: Values,
    _less: Less,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Source: SortedBoundsInput<R, Values, Less>,
{
    source.lower_bounds(exec, values)
}

pub(crate) fn upper_bounds_storage<R, Source, Values, Less>(
    exec: &Executor<R>,
    source: Source,
    values: Values,
    _less: Less,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Source: SortedBoundsInput<R, Values, Less>,
{
    source.upper_bounds(exec, values)
}

impl<R, Left, Right, Op> PairCodeInput<R, Right, Op> for Left
where
    R: Runtime,
    Left: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Right: ReadExpression<Item = Left::Item> + LowerReadExpression + StageRead<R, Env0>,
    PairDispatch<crate::core::storage::S12>:
        PairCodeDispatch<R, Left, Right, Left::Item, Left::Slots, Right::Slots, Op>,
{
    fn pair_codes(
        self,
        exec: &Executor<R>,
        right: Right,
    ) -> Result<
        (
            DeviceVec<R, u32>,
            DeviceVec<R, u32>,
            crate::core::extent::LogicalExtent,
            crate::core::extent::LogicalExtent,
        ),
        Error,
    > {
        <PairDispatch<crate::core::storage::S12> as PairCodeDispatch<
            R,
            Left,
            Right,
            Left::Item,
            Left::Slots,
            Right::Slots,
            Op,
        >>::run(exec, &self, &right)
    }
}

/// Internal capability hiding fixed-ABI pair-code dispatch for equality.
#[doc(hidden)]
pub trait EqualityInput<R: Runtime, Right, Equal>: ReadExpression + Sized {
    fn mismatch_control(
        self,
        exec: &Executor<R>,
        right: Right,
    ) -> Result<
        (
            DeviceVec<R, u32>,
            DeviceVec<R, u32>,
            crate::core::extent::LogicalExtent,
            crate::core::extent::LogicalExtent,
        ),
        Error,
    >;
}

impl<R, Left, Right, Equal> EqualityInput<R, Right, Equal> for Left
where
    R: Runtime,
    Left: ReadExpression + PairCodeInput<R, Right, MismatchCode<Equal>>,
{
    fn mismatch_control(
        self,
        exec: &Executor<R>,
        right: Right,
    ) -> Result<
        (
            DeviceVec<R, u32>,
            DeviceVec<R, u32>,
            crate::core::extent::LogicalExtent,
            crate::core::extent::LogicalExtent,
        ),
        Error,
    > {
        self.pair_codes(exec, right)
    }
}

/// Returns whether two ranges have equal length and equal items.
pub(crate) fn equal<R, Left, Right, Equal>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    _equal: Equal,
) -> Result<MVec<R, u32>, Error>
where
    R: Runtime,
    Left: EqualityInput<R, Right, Equal>,
{
    let (_codes, mismatch, _left_extent, _right_extent) = left.mismatch_control(exec, right)?;
    Ok(mismatch)
}

/// Returns the first mismatch, including the shared end when lengths differ.
pub(crate) fn mismatch<R, Left, Right, Equal>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    _equal: Equal,
) -> Result<MVec<R, u32>, Error>
where
    R: Runtime,
    Left: EqualityInput<R, Right, Equal>,
{
    let (_codes, mismatch, _left_extent, _right_extent) = left.mismatch_control(exec, right)?;
    Ok(mismatch)
}

/// Internal capability hiding fixed-ABI lexicographical dispatch.
#[doc(hidden)]
pub trait LexicographicalInput<R: Runtime, Right, Less>: ReadExpression + Sized {
    fn lexicographical_codes(
        self,
        exec: &Executor<R>,
        right: Right,
    ) -> Result<
        (
            DeviceVec<R, u32>,
            DeviceVec<R, u32>,
            crate::core::extent::LogicalExtent,
            crate::core::extent::LogicalExtent,
        ),
        Error,
    >;
}

impl<R, Left, Right, Less> LexicographicalInput<R, Right, Less> for Left
where
    R: Runtime,
    Left: ReadExpression + PairCodeInput<R, Right, LexicographicalCode<Less>>,
{
    fn lexicographical_codes(
        self,
        exec: &Executor<R>,
        right: Right,
    ) -> Result<
        (
            DeviceVec<R, u32>,
            DeviceVec<R, u32>,
            crate::core::extent::LogicalExtent,
            crate::core::extent::LogicalExtent,
        ),
        Error,
    > {
        self.pair_codes(exec, right)
    }
}

/// Lexicographically compares two semantic item ranges.
pub(crate) fn lexicographical_compare<R, Left, Right, Less>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    _less: Less,
) -> Result<MVec<R, u32>, Error>
where
    R: Runtime,
    Left: LexicographicalInput<R, Right, Less>,
{
    let (codes, mismatch, left_extent, right_extent) = left.lexicographical_codes(exec, right)?;
    let shared_len = crate::core::extent::LogicalExtent::min(exec, &left_extent, &right_extent)?
        .materialize(exec)?;
    let left_is_shorter =
        crate::core::extent::LogicalExtent::less_value(exec, &left_extent, &right_extent)?;
    let output = exec.alloc_row::<u32>(1);
    unsafe {
        lexicographical_result_kernel::launch_unchecked::<R>(
            exec.client(),
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(codes.handle.clone(), codes.capacity()),
            BufferArg::from_raw_parts(mismatch.handle.clone(), 1),
            BufferArg::from_raw_parts(shared_len.handle.clone(), 1),
            BufferArg::from_raw_parts(left_is_shorter.handle.clone(), 1),
            BufferArg::from_raw_parts(output.handle.clone(), 1),
        );
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::iter::Zip;
    use crate::core::read::{Counting, Permute};
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    type Seven = (u32, u32, u32, u32, u32, u32, u32);

    struct EqualSeven;

    #[cubecl::cube]
    impl BinaryPredicateOp<Seven> for EqualSeven {
        fn apply(lhs: Seven, rhs: Seven) -> crate::MFlag {
            crate::flag::from_bool(
                lhs.0 == rhs.0
                    && lhs.1 == rhs.1
                    && lhs.2 == rhs.2
                    && lhs.3 == rhs.3
                    && lhs.4 == rhs.4
                    && lhs.5 == rhs.5
                    && lhs.6 == rhs.6,
            )
        }
    }

    struct LessSeven;

    #[cubecl::cube]
    impl BinaryPredicateOp<Seven> for LessSeven {
        fn apply(lhs: Seven, rhs: Seven) -> crate::MFlag {
            crate::flag::from_bool(lhs.0 < rhs.0)
        }
    }

    struct EqualU32;

    #[cubecl::cube]
    impl BinaryPredicateOp<u32> for EqualU32 {
        fn apply(lhs: u32, rhs: u32) -> crate::MFlag {
            crate::flag::from_bool(lhs == rhs)
        }
    }

    struct LessU32;

    #[cubecl::cube]
    impl BinaryPredicateOp<u32> for LessU32 {
        fn apply(lhs: u32, rhs: u32) -> crate::MFlag {
            crate::flag::from_bool(lhs < rhs)
        }
    }

    macro_rules! assert_scalar_pair_kernel_budget {
        (
            $module:ident::$kernel:ident,
            $operation:ty,
            $name:literal,
            $settings:expr,
            $client:expr,
            $arg:expr
        ) => {{
            type ScalarExpr = <crate::core::read::Column<u32> as LowerReadExpression>::DeviceExpr;
            type Kernel = $module::$kernel<
                u32,
                u32,
                u32,
                u32,
                u32,
                u32,
                u32,
                u32,
                u32,
                u32,
                u32,
                u32,
                u32,
                u32,
                u32,
                u32,
                u32,
                u32,
                u32,
                u32,
                u32,
                u32,
                u32,
                u32,
                u32,
                u32,
                u32,
                ScalarExpr,
                ScalarExpr,
                $operation,
                WgpuRuntime,
            >;
            let generated = crate::core::launch::kernel_with_max_explicit_storage_bindings!(
                Kernel, $settings, $client, $arg
            );
            crate::core::launch::assert_binding_budget($name, &generated);
        }};
    }

    #[test]
    fn generated_two_input_search_kernels_fit_the_binding_budget() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let settings = KernelSettings::new(
            CubeDim::new_1d(BLOCK_SIZE).into(),
            ExecutionMode::Unchecked,
            AddressType::U32,
        );
        let mut launcher = KernelLauncher::<WgpuRuntime>::new(settings.clone());
        let handle = exec.client().empty(core::mem::size_of::<u32>());
        let arg = unsafe {
            <[u32] as LaunchArg>::register(BufferArg::from_raw_parts(handle, 1), &mut launcher)
        };

        assert_scalar_pair_kernel_budget!(
            pair_code_a13::PairCodeA13,
            MismatchCode<EqualU32>,
            "pair comparison",
            settings.clone(),
            exec.client().clone(),
            arg.clone()
        );
        assert_scalar_pair_kernel_budget!(
            find_first_of_a13::FindFirstOfA13,
            EqualU32,
            "find first of",
            settings.clone(),
            exec.client().clone(),
            arg.clone()
        );
        assert_scalar_pair_kernel_budget!(
            bound_a13::BoundA13,
            LowerBound<LessU32>,
            "sorted bound",
            settings,
            exec.client().clone(),
            arg
        );
    }

    #[test]
    fn shared_offset_binding_preserves_each_input_slice() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let left = exec.to_device(&[90_u32, 1, 2, 3, 91]);
        let right = exec.to_device(&[80_u32, 81, 1, 4, 3, 82]);
        let needles = exec.to_device(&[70_u32, 71, 3, 72]);

        assert_eq!(
            crate::api::algorithm::mismatch(&exec, left.slice(1..4), right.slice(2..5), EqualU32,)
                .unwrap(),
            Some(1)
        );
        assert_eq!(
            crate::api::algorithm::find_first_of(
                &exec,
                left.slice(1..4),
                needles.slice(2..3),
                EqualU32,
            )
            .unwrap(),
            Some(2)
        );
    }

    #[test]
    fn pair_queries_compare_two_independent_storage7_inputs() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let left0 = exec.to_device(&[1_u32, 2, 3]);
        let right0 = exec.to_device(&[1_u32, 4, 3]);
        let left_tail: Vec<_> = (1_u32..7)
            .map(|value| exec.to_device(&[value; 3]))
            .collect();
        let right_tail: Vec<_> = (1_u32..7)
            .map(|value| exec.to_device(&[value; 3]))
            .collect();
        let make_left = || {
            Permute::new(
                Zip::new(
                    left0.column(),
                    Zip::new(
                        left_tail[0].column(),
                        Zip::new(
                            left_tail[1].column(),
                            Zip::new(
                                left_tail[2].column(),
                                Zip::new(
                                    left_tail[3].column(),
                                    Zip::new(left_tail[4].column(), left_tail[5].column()),
                                ),
                            ),
                        ),
                    ),
                ),
                Counting::new(0, 3),
            )
        };
        let make_right = || {
            Permute::new(
                Zip::new(
                    right0.column(),
                    Zip::new(
                        right_tail[0].column(),
                        Zip::new(
                            right_tail[1].column(),
                            Zip::new(
                                right_tail[2].column(),
                                Zip::new(
                                    right_tail[3].column(),
                                    Zip::new(right_tail[4].column(), right_tail[5].column()),
                                ),
                            ),
                        ),
                    ),
                ),
                Counting::new(0, 3),
            )
        };

        assert_eq!(
            crate::api::algorithm::equal(&exec, make_left(), make_right(), EqualSeven).unwrap(),
            crate::flag::from_bool(false)
        );
        assert_eq!(
            crate::api::algorithm::mismatch(&exec, make_left(), make_right(), EqualSeven).unwrap(),
            Some(1)
        );
        assert_eq!(
            crate::api::algorithm::lexicographical_compare(
                &exec,
                make_left(),
                make_right(),
                LessSeven,
            )
            .unwrap(),
            crate::flag::from_bool(true)
        );
    }

    #[test]
    fn range_queries_cover_empty_needles_and_batched_bounds() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let source = exec.to_device(&[1_u32, 2, 2, 4]);
        let needles = exec.to_device(&[9_u32, 2]);
        let empty = exec.to_device::<u32>(&[]);
        assert_eq!(
            crate::api::algorithm::find_first_of(
                &exec,
                source.column(),
                needles.column(),
                EqualU32,
            )
            .unwrap(),
            Some(1)
        );
        assert_eq!(
            crate::api::algorithm::find_first_of(&exec, source.column(), empty.column(), EqualU32,)
                .unwrap(),
            None
        );

        let values = exec.to_device(&[0_u32, 2, 3, 5]);
        let lower =
            crate::api::algorithm::lower_bound(&exec, source.column(), values.column(), LessU32)
                .unwrap();
        let upper =
            crate::api::algorithm::upper_bound(&exec, source.column(), values.column(), LessU32)
                .unwrap();
        assert_eq!(exec.to_host(&lower).unwrap(), vec![0, 1, 3, 4]);
        assert_eq!(exec.to_host(&upper).unwrap(), vec![0, 3, 3, 4]);
    }

    #[test]
    fn lexicographical_direction_code_distinguishes_greater_left_item() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let left = exec.to_device(&[93_u32]);
        let right = exec.to_device(&[43_u32]);
        let (codes, _, _, _) =
            <_ as PairCodeInput<WgpuRuntime, _, LexicographicalCode<LessU32>>>::pair_codes(
                crate::core::read::FixedRead::new(left.column()),
                &exec,
                crate::core::read::FixedRead::new(right.column()),
            )
            .unwrap();

        assert_eq!(exec.to_host(&codes).unwrap(), vec![2]);
        assert_eq!(
            crate::api::algorithm::lexicographical_compare(
                &exec,
                left.column(),
                right.column(),
                LessU32,
            )
            .unwrap(),
            crate::flag::from_bool(false)
        );
    }
}
