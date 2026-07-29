//! Recursive output trees and physical storage staging.

use core::marker::PhantomData;
use cubecl::prelude::Runtime;

use crate::core::bindings::Bindings;
use crate::core::iter::Zip;
use crate::core::read::{
    Column, Env0, Env1, Env2, Env3, Env4, Env5, Env6, Env7, Env8, Env9, Env10, Env11, Env12,
};
use crate::core::runtime::ColumnMut;
use crate::core::storage::{
    Concat, FlatLeaves, FlatRow, JoinedRow, S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12,
    StorageArity, StorageLayout,
};
use crate::core::value::MStorageElement;
use crate::{DeviceSliceMut, Error};
use std::ops::RangeBounds;

/// A zero-copy mutable subrange tied to the runtime selected by `MIterMut`.
#[derive(Clone, Copy, Debug)]
pub struct Slice<R, Output> {
    output: Output,
    _runtime: PhantomData<fn() -> R>,
}

impl<R, Output> Slice<R, Output> {
    pub const fn new(output: Output) -> Self {
        Self {
            output,
            _runtime: PhantomData,
        }
    }

    pub(crate) fn into_inner(self) -> Output {
        self.output
    }
}

impl<R, Output> OutputExpression for Slice<R, Output>
where
    Output: OutputExpression,
{
    type Item = Output::Item;
    type StorageArity = Output::StorageArity;

    fn physical_len(&self) -> Result<usize, Error> {
        self.output.physical_len()
    }
}

impl<R, Output> SliceOutput for Slice<R, Output>
where
    Output: SliceOutput,
{
    fn slice_output<Range: RangeBounds<usize>>(&self, range: Range) -> Self {
        Self::new(self.output.slice_output(range))
    }
}

impl<R, Output> ReadOutput for Slice<R, Output>
where
    Output: ReadOutput,
{
    type Read = crate::core::read::Slice<R, Output::Read>;

    fn slice_read<Range: RangeBounds<usize>>(&self, range: Range) -> Self::Read {
        crate::core::read::Slice::new(self.output.slice_read(range))
    }
}

/// A mutable output whose public shape is a binary tree.
pub trait OutputExpression {
    type Item: StorageLayout;
    type StorageArity: StorageArity;

    /// Number of rows available in the preallocated destination.
    fn physical_len(&self) -> Result<usize, Error>;
}

/// Creates same-shaped subviews of a recursive output tree.
pub trait SliceOutput: OutputExpression + Sized {
    fn slice_output<Range: RangeBounds<usize>>(&self, range: Range) -> Self;
}

/// Creates a read-only view over a mutable output tree.
#[doc(hidden)]
pub trait ReadOutput: OutputExpression {
    type Read: crate::core::read::ReadExpression<Item = Self::Item>;

    fn slice_read<Range: RangeBounds<usize>>(&self, range: Range) -> Self::Read;
}

impl<T> SliceOutput for ColumnMut<T>
where
    T: MStorageElement + StorageLayout<StorageArity = S1>,
{
    fn slice_output<Range: RangeBounds<usize>>(&self, range: Range) -> Self {
        self.slice_mut_usize(range)
    }
}

impl<T> OutputExpression for ColumnMut<T>
where
    T: MStorageElement + StorageLayout<StorageArity = S1>,
{
    type Item = T;
    type StorageArity = S1;

    fn physical_len(&self) -> Result<usize, Error> {
        Ok(self.len)
    }
}

impl<T> ReadOutput for ColumnMut<T>
where
    T: MStorageElement + StorageLayout<StorageArity = S1>,
{
    type Read = Column<T>;

    fn slice_read<Range: RangeBounds<usize>>(&self, range: Range) -> Self::Read {
        self.slice_usize(range)
    }
}

impl<R, T> SliceOutput for DeviceSliceMut<R, T>
where
    R: Runtime,
    T: MStorageElement + StorageLayout<StorageArity = S1>,
{
    fn slice_output<Range: RangeBounds<usize>>(&self, range: Range) -> Self {
        DeviceSliceMut::from_output(self.output.slice_mut_usize(range))
    }
}

impl<R, T> OutputExpression for DeviceSliceMut<R, T>
where
    R: Runtime,
    T: MStorageElement + StorageLayout<StorageArity = S1>,
{
    type Item = T;
    type StorageArity = S1;

    fn physical_len(&self) -> Result<usize, Error> {
        Ok(self.output.len)
    }
}

impl<R, T> ReadOutput for DeviceSliceMut<R, T>
where
    R: Runtime,
    T: MStorageElement + StorageLayout<StorageArity = S1>,
{
    type Read = Column<T>;

    fn slice_read<Range: RangeBounds<usize>>(&self, range: Range) -> Self::Read {
        self.output.slice_usize(range)
    }
}

impl<Left, Right> OutputExpression for Zip<Left, Right>
where
    Left: OutputExpression,
    Right: OutputExpression,
    Left::Item: FlatRow,
    Right::Item: FlatRow,
    <Left::Item as StorageLayout>::StorageLeaves:
        FlatLeaves<Item = Left::Item> + Concat<<Right::Item as StorageLayout>::StorageLeaves>,
    <Right::Item as StorageLayout>::StorageLeaves: FlatLeaves<Item = Right::Item>,
    <<Left::Item as StorageLayout>::StorageLeaves as Concat<
        <Right::Item as StorageLayout>::StorageLeaves,
    >>::Output: FlatLeaves,
{
    type Item = JoinedRow<Left::Item, Right::Item>;
    type StorageArity = <Self::Item as StorageLayout>::StorageArity;

    fn physical_len(&self) -> Result<usize, Error> {
        let left = self.0.physical_len()?;
        let right = self.1.physical_len()?;
        if left != right {
            return Err(Error::LengthMismatch { left, right });
        }
        Ok(left)
    }
}

impl<Left, Right> SliceOutput for Zip<Left, Right>
where
    Left: SliceOutput,
    Right: SliceOutput,
    Zip<Left, Right>: OutputExpression,
{
    fn slice_output<Range: RangeBounds<usize>>(&self, range: Range) -> Self {
        let len = self
            .physical_len()
            .expect("output columns have equal lengths");
        let (start, count) = crate::core::read::resolve_slice_range(len, range);
        Zip::new(
            self.0.slice_output(start..start + count),
            self.1.slice_output(start..start + count),
        )
    }
}

impl<Left, Right> ReadOutput for Zip<Left, Right>
where
    Left: ReadOutput,
    Right: ReadOutput,
    Zip<Left, Right>: OutputExpression,
    Zip<Left::Read, Right::Read>:
        crate::core::read::ReadExpression<Item = <Zip<Left, Right> as OutputExpression>::Item>,
{
    type Read = Zip<Left::Read, Right::Read>;

    fn slice_read<Range: RangeBounds<usize>>(&self, range: Range) -> Self::Read {
        let len = self
            .physical_len()
            .expect("output columns have equal lengths");
        let (start, count) = crate::core::read::resolve_slice_range(len, range);
        Zip::new(
            self.0.slice_read(start..start + count),
            self.1.slice_read(start..start + count),
        )
    }
}

/// Binds output leaves left-to-right to an environment.
#[doc(hidden)]
pub trait BindOutputSlots<Env> {
    type NextEnv;
}

macro_rules! impl_output_leaf_binding {
    (impl <$( $env_ty:ident ),*> $env:ty => $next:ty) => {
        impl<T, $( $env_ty ),*> BindOutputSlots<$env> for ColumnMut<T>
        where
            T: MStorageElement,
        {
            type NextEnv = $next;
        }
    };
}

impl<R, T, Env> BindOutputSlots<Env> for DeviceSliceMut<R, T>
where
    R: Runtime,
    T: MStorageElement,
    ColumnMut<T>: BindOutputSlots<Env>,
{
    type NextEnv = <ColumnMut<T> as BindOutputSlots<Env>>::NextEnv;
}

impl_output_leaf_binding!(impl <> Env0 => Env1<T>);
impl_output_leaf_binding!(impl <L0> Env1<L0> => Env2<L0, T>);
impl_output_leaf_binding!(impl <L0, L1> Env2<L0, L1> => Env3<L0, L1, T>);
impl_output_leaf_binding!(impl <L0, L1, L2> Env3<L0, L1, L2> => Env4<L0, L1, L2, T>);
impl_output_leaf_binding!(impl <L0, L1, L2, L3> Env4<L0, L1, L2, L3> => Env5<L0, L1, L2, L3, T>);
impl_output_leaf_binding!(impl <L0, L1, L2, L3, L4> Env5<L0, L1, L2, L3, L4> => Env6<L0, L1, L2, L3, L4, T>);
impl_output_leaf_binding!(impl <L0, L1, L2, L3, L4, L5> Env6<L0, L1, L2, L3, L4, L5> => Env7<L0, L1, L2, L3, L4, L5, T>);
impl_output_leaf_binding!(impl <L0, L1, L2, L3, L4, L5, L6> Env7<L0, L1, L2, L3, L4, L5, L6> => Env8<L0, L1, L2, L3, L4, L5, L6, T>);
impl_output_leaf_binding!(impl <L0, L1, L2, L3, L4, L5, L6, L7> Env8<L0, L1, L2, L3, L4, L5, L6, L7> => Env9<L0, L1, L2, L3, L4, L5, L6, L7, T>);
impl_output_leaf_binding!(impl <L0, L1, L2, L3, L4, L5, L6, L7, L8> Env9<L0, L1, L2, L3, L4, L5, L6, L7, L8> => Env10<L0, L1, L2, L3, L4, L5, L6, L7, L8, T>);
impl_output_leaf_binding!(impl <L0, L1, L2, L3, L4, L5, L6, L7, L8, L9> Env10<L0, L1, L2, L3, L4, L5, L6, L7, L8, L9> => Env11<L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, T>);
impl_output_leaf_binding!(impl <L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10> Env11<L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10> => Env12<L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, T>);

impl<Left, Right, Env> BindOutputSlots<Env> for Zip<Left, Right>
where
    Left: BindOutputSlots<Env>,
    Right: BindOutputSlots<Left::NextEnv>,
{
    type NextEnv = Right::NextEnv;
}

impl<R, Output, Env> BindOutputSlots<Env> for Slice<R, Output>
where
    Output: BindOutputSlots<Env>,
{
    type NextEnv = Output::NextEnv;
}

#[doc(hidden)]
pub trait OutputSlotEnvironment {
    type StorageArity: StorageArity;
}

/// Normalizes the physical output ABI to twelve buffer slots without changing
/// the semantic item or its storage layout. Unused slots are typed as `u32`
/// and are never accessed by device-side padded stores.
#[doc(hidden)]
pub trait PaddedOutputSlots: OutputSlotEnvironment {
    type Leaves: crate::core::storage::StorePadded12;
}

/// Maps an ordered physical leaf list back to its actual (unpadded) output
/// slot environment.
#[doc(hidden)]
pub trait OutputSlotLayout: crate::core::storage::StorePadded12 {
    type Slots: PaddedOutputSlots<Leaves = Self>;
}

#[doc(hidden)]
pub type KernelOutputSlots<Slots> = Env12<
    <<Slots as PaddedOutputSlots>::Leaves as crate::core::storage::StorePadded12>::O0,
    <<Slots as PaddedOutputSlots>::Leaves as crate::core::storage::StorePadded12>::O1,
    <<Slots as PaddedOutputSlots>::Leaves as crate::core::storage::StorePadded12>::O2,
    <<Slots as PaddedOutputSlots>::Leaves as crate::core::storage::StorePadded12>::O3,
    <<Slots as PaddedOutputSlots>::Leaves as crate::core::storage::StorePadded12>::O4,
    <<Slots as PaddedOutputSlots>::Leaves as crate::core::storage::StorePadded12>::O5,
    <<Slots as PaddedOutputSlots>::Leaves as crate::core::storage::StorePadded12>::O6,
    <<Slots as PaddedOutputSlots>::Leaves as crate::core::storage::StorePadded12>::O7,
    <<Slots as PaddedOutputSlots>::Leaves as crate::core::storage::StorePadded12>::O8,
    <<Slots as PaddedOutputSlots>::Leaves as crate::core::storage::StorePadded12>::O9,
    <<Slots as PaddedOutputSlots>::Leaves as crate::core::storage::StorePadded12>::O10,
    <<Slots as PaddedOutputSlots>::Leaves as crate::core::storage::StorePadded12>::O11,
>;

macro_rules! impl_padded_output_slots {
    ($env:ty; [$($actual:ident),+]) => {
        impl<$($actual: MStorageElement),+> PaddedOutputSlots for $env {
            type Leaves = impl_padded_output_slots!(@leaves $($actual),+);
        }

        impl<$($actual: MStorageElement),+> OutputSlotLayout
            for impl_padded_output_slots!(@leaves $($actual),+)
        {
            type Slots = $env;
        }
    };
    (@leaves $last:ty) => { crate::core::storage::Last<$last> };
    (@leaves $head:ty, $($tail:ty),+) => {
        crate::core::storage::More<$head, impl_padded_output_slots!(@leaves $($tail),+)>
    };
}

impl_padded_output_slots!(Env1<O0>; [O0]);
impl_padded_output_slots!(Env2<O0, O1>; [O0, O1]);
impl_padded_output_slots!(Env3<O0, O1, O2>; [O0, O1, O2]);
impl_padded_output_slots!(Env4<O0, O1, O2, O3>; [O0, O1, O2, O3]);
impl_padded_output_slots!(Env5<O0, O1, O2, O3, O4>; [O0, O1, O2, O3, O4]);
impl_padded_output_slots!(Env6<O0, O1, O2, O3, O4, O5>; [O0, O1, O2, O3, O4, O5]);
impl_padded_output_slots!(Env7<O0, O1, O2, O3, O4, O5, O6>; [O0, O1, O2, O3, O4, O5, O6]);
impl_padded_output_slots!(Env8<O0, O1, O2, O3, O4, O5, O6, O7>; [O0, O1, O2, O3, O4, O5, O6, O7]);
impl_padded_output_slots!(Env9<O0, O1, O2, O3, O4, O5, O6, O7, O8>; [O0, O1, O2, O3, O4, O5, O6, O7, O8]);
impl_padded_output_slots!(Env10<O0, O1, O2, O3, O4, O5, O6, O7, O8, O9>; [O0, O1, O2, O3, O4, O5, O6, O7, O8, O9]);
impl_padded_output_slots!(Env11<O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10>; [O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10]);
impl_padded_output_slots!(Env12<O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11>; [O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11]);

impl<L0> OutputSlotEnvironment for Env1<L0> {
    type StorageArity = S1;
}
impl<L0, L1> OutputSlotEnvironment for Env2<L0, L1> {
    type StorageArity = S2;
}
impl<L0, L1, L2> OutputSlotEnvironment for Env3<L0, L1, L2> {
    type StorageArity = S3;
}
impl<L0, L1, L2, L3> OutputSlotEnvironment for Env4<L0, L1, L2, L3> {
    type StorageArity = S4;
}
impl<L0, L1, L2, L3, L4> OutputSlotEnvironment for Env5<L0, L1, L2, L3, L4> {
    type StorageArity = S5;
}
impl<L0, L1, L2, L3, L4, L5> OutputSlotEnvironment for Env6<L0, L1, L2, L3, L4, L5> {
    type StorageArity = S6;
}
impl<L0, L1, L2, L3, L4, L5, L6> OutputSlotEnvironment for Env7<L0, L1, L2, L3, L4, L5, L6> {
    type StorageArity = S7;
}
impl<L0, L1, L2, L3, L4, L5, L6, L7> OutputSlotEnvironment
    for Env8<L0, L1, L2, L3, L4, L5, L6, L7>
{
    type StorageArity = S8;
}
impl<L0, L1, L2, L3, L4, L5, L6, L7, L8> OutputSlotEnvironment
    for Env9<L0, L1, L2, L3, L4, L5, L6, L7, L8>
{
    type StorageArity = S9;
}
impl<L0, L1, L2, L3, L4, L5, L6, L7, L8, L9> OutputSlotEnvironment
    for Env10<L0, L1, L2, L3, L4, L5, L6, L7, L8, L9>
{
    type StorageArity = S10;
}
impl<L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10> OutputSlotEnvironment
    for Env11<L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10>
{
    type StorageArity = S11;
}
impl<L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11> OutputSlotEnvironment
    for Env12<L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11>
{
    type StorageArity = S12;
}

/// Fully bound output tree.
#[doc(hidden)]
pub trait LowerOutputExpression:
    OutputExpression + BindOutputSlots<Env0, NextEnv = Self::Slots>
{
    type Slots: OutputSlotEnvironment<StorageArity = Self::StorageArity>;
}

impl<Output> LowerOutputExpression for Output
where
    Output: OutputExpression + BindOutputSlots<Env0>,
    Output::NextEnv: OutputSlotEnvironment<StorageArity = Output::StorageArity>,
{
    type Slots = Output::NextEnv;
}

/// Stages output buffers using the same left-first traversal as storage layout.
#[doc(hidden)]
pub trait StageOutput<R: Runtime, Env>: BindOutputSlots<Env> {
    fn stage_output(&self, owner: u64, bindings: &mut Bindings) -> Result<(), Error>;
}

impl<R, T, Env> StageOutput<R, Env> for DeviceSliceMut<R, T>
where
    R: Runtime,
    T: MStorageElement,
    ColumnMut<T>: StageOutput<R, Env>,
    DeviceSliceMut<R, T>:
        BindOutputSlots<Env, NextEnv = <ColumnMut<T> as BindOutputSlots<Env>>::NextEnv>,
{
    fn stage_output(&self, owner: u64, bindings: &mut Bindings) -> Result<(), Error> {
        self.output.stage_output(owner, bindings)
    }
}

impl<R, T, Env> StageOutput<R, Env> for ColumnMut<T>
where
    R: Runtime,
    T: MStorageElement,
    ColumnMut<T>: BindOutputSlots<Env>,
{
    fn stage_output(&self, owner: u64, bindings: &mut Bindings) -> Result<(), Error> {
        if self.owner != owner {
            return Err(Error::ForeignExecutor);
        }
        bindings.push(self.handle.clone(), self.buffer_len, self.offset);
        Ok(())
    }
}

impl<R, Left, Right, Env> StageOutput<R, Env> for Zip<Left, Right>
where
    R: Runtime,
    Left: StageOutput<R, Env>,
    Right: StageOutput<R, Left::NextEnv>,
    Zip<Left, Right>: BindOutputSlots<Env>,
{
    fn stage_output(&self, owner: u64, bindings: &mut Bindings) -> Result<(), Error> {
        self.0.stage_output(owner, bindings)?;
        self.1.stage_output(owner, bindings)
    }
}

impl<R, Output, Env> StageOutput<R, Env> for Slice<R, Output>
where
    R: Runtime,
    Output: StageOutput<R, Env>,
    Slice<R, Output>: BindOutputSlots<Env>,
{
    fn stage_output(&self, owner: u64, bindings: &mut Bindings) -> Result<(), Error> {
        self.output.stage_output(owner, bindings)
    }
}
