//! Recursive allocation of flat-row SoA storage.

#![allow(private_interfaces)]

use std::ops::RangeBounds;

use cubecl::prelude::Runtime;

use crate::api::iter::{
    MAlloc, MIter, MIterMut, MStorage, MStorageExtent, StorageSlice, StorageSliceMut,
};
use crate::core::iter::Zip;
use crate::core::output::{
    LowerOutputExpression, OutputExpression, PaddedOutputSlots, SliceOutput, StageOutput,
};
use crate::core::read::{
    Column, Env0, LowerReadExpression, ReadExpression, SlotEnvironment, StageRead,
};
use crate::core::runtime::ColumnMut;
use crate::core::selection::FillOutput;
use crate::core::storage::{Concat, FlatLeaves, FlatRow, JoinedRow, Last, More, S1, StorageLayout};
use crate::core::transform::materialize;
use crate::core::value::MStorageElement;
use crate::{DeviceVec, Error, Executor};

/// Owned storage that can produce read and mutable output trees.
pub trait RowStorage<R: Runtime>: Clone + Send + Sync + 'static {
    type Item: StorageLayout;
    type ReadSlots: SlotEnvironment<Arity = <Self::Read as ReadExpression>::ReadArity>
        + crate::core::read::PaddedReadSlots;
    type WriteSlots: PaddedOutputSlots<Leaves = <Self::Item as StorageLayout>::StorageLeaves>;
    type Read: ReadExpression<Item = Self::Item>
        + LowerReadExpression<Slots = Self::ReadSlots>
        + StageRead<R, Env0>
        + Clone;
    type Write: OutputExpression<Item = Self::Item>
        + LowerOutputExpression<Slots = Self::WriteSlots>
        + StageOutput<R, Env0>
        + SliceOutput
        + FillOutput<R>;

    fn len(&self) -> Result<usize, Error>;
    fn logical_extent(&self) -> crate::core::extent::LogicalExtent;
    fn set_logical_extent(&mut self, extent: crate::core::extent::LogicalExtent);
    fn read(&self) -> Self::Read;
    fn write(&self) -> Self::Write;
    /// Uploads one semantic row by composing the physical column transfers.
    fn from_item(exec: &Executor<R>, value: Self::Item) -> Self;
    fn read_first(&self, exec: &Executor<R>) -> Result<Self::Item, Error>;
    fn slice<Range: RangeBounds<usize>>(&self, range: Range) -> Self::Read;
    fn slice_mut<Range: RangeBounds<usize>>(&self, range: Range) -> Self::Write;
    fn read_item(&self, exec: &Executor<R>, index: usize) -> Result<Self::Item, Error>;
}

/// Copies physical row storage into an equally shaped output tree.
#[doc(hidden)]
pub trait CopyStorage<R: Runtime>: RowStorage<R> {
    fn copy_storage(&self, exec: &Executor<R>, output: Self::Write) -> Result<(), Error>;
}

impl<R, T> RowStorage<R> for DeviceVec<R, T>
where
    R: Runtime,
    T: MStorageElement + StorageLayout<StorageArity = S1, StorageLeaves = Last<T>>,
{
    type Item = T;
    type ReadSlots = crate::core::read::Env1<T>;
    type WriteSlots = crate::core::read::Env1<T>;
    type Read = Column<T>;
    type Write = ColumnMut<T>;

    fn len(&self) -> Result<usize, Error> {
        Ok(self.capacity())
    }
    fn logical_extent(&self) -> crate::core::extent::LogicalExtent {
        DeviceVec::logical_extent(self)
    }
    fn set_logical_extent(&mut self, extent: crate::core::extent::LogicalExtent) {
        DeviceVec::set_logical_extent(self, extent);
    }
    fn read(&self) -> Self::Read {
        self.column()
    }
    fn write(&self) -> Self::Write {
        self.slice_mut_usize(..)
    }
    fn from_item(exec: &Executor<R>, value: Self::Item) -> Self {
        exec.to_device(&[value])
    }
    fn read_first(&self, exec: &Executor<R>) -> Result<Self::Item, Error> {
        exec.to_host(self)?
            .into_iter()
            .next()
            .ok_or(Error::LengthMismatch { left: 0, right: 1 })
    }
    fn slice<Range: RangeBounds<usize>>(&self, range: Range) -> Self::Read {
        self.slice_usize(range)
    }
    fn slice_mut<Range: RangeBounds<usize>>(&self, range: Range) -> Self::Write {
        self.slice_mut_usize(range)
    }

    fn read_item(&self, exec: &Executor<R>, index: usize) -> Result<Self::Item, Error> {
        if index >= self.capacity() {
            return Err(Error::LengthMismatch {
                left: index + 1,
                right: self.capacity(),
            });
        }
        Ok(exec.to_host(&self.slice_usize(index..index + 1))?[0])
    }
}

impl<R, T> CopyStorage<R> for DeviceVec<R, T>
where
    R: Runtime,
    T: MStorageElement + StorageLayout<StorageArity = S1, StorageLeaves = Last<T>>,
    Column<T>: ReadExpression<Item = T, ReadArity = crate::core::arity::A1>
        + LowerReadExpression<Slots = crate::core::read::Env1<T>>
        + StageRead<R, Env0>,
    ColumnMut<T>: OutputExpression<Item = T, StorageArity = S1>
        + LowerOutputExpression<Slots = crate::core::read::Env1<T>>
        + StageOutput<R, Env0>,
{
    fn copy_storage(&self, exec: &Executor<R>, output: Self::Write) -> Result<(), Error> {
        materialize(exec, self.read(), output)
    }
}

impl<R, Left, Right> RowStorage<R> for Zip<Left, Right>
where
    R: Runtime,
    Left: RowStorage<R>,
    Right: RowStorage<R>,
    Left::Item: FlatRow,
    Right::Item: FlatRow,
    <Left::Item as StorageLayout>::StorageLeaves:
        FlatLeaves<Item = Left::Item> + Concat<<Right::Item as StorageLayout>::StorageLeaves>,
    <Right::Item as StorageLayout>::StorageLeaves: FlatLeaves<Item = Right::Item>,
    <<Left::Item as StorageLayout>::StorageLeaves as Concat<
        <Right::Item as StorageLayout>::StorageLeaves,
    >>::Output: FlatLeaves,
    Zip<Left::Read, Right::Read>: ReadExpression<Item = JoinedRow<Left::Item, Right::Item>>
        + LowerReadExpression
        + StageRead<R, Env0>,
    Zip<Left::Write, Right::Write>: OutputExpression<Item = JoinedRow<Left::Item, Right::Item>>
        + LowerOutputExpression
        + StageOutput<R, Env0>
        + SliceOutput
        + FillOutput<R>,
    <Zip<Left::Write, Right::Write> as LowerOutputExpression>::Slots: PaddedOutputSlots<
        Leaves = <JoinedRow<Left::Item, Right::Item> as StorageLayout>::StorageLeaves,
    >,
{
    type Item = JoinedRow<Left::Item, Right::Item>;
    type ReadSlots = <Zip<Left::Read, Right::Read> as LowerReadExpression>::Slots;
    type WriteSlots = <Zip<Left::Write, Right::Write> as LowerOutputExpression>::Slots;
    type Read = Zip<Left::Read, Right::Read>;
    type Write = Zip<Left::Write, Right::Write>;

    fn len(&self) -> Result<usize, Error> {
        let left = self.0.len()?;
        let right = self.1.len()?;
        if left != right {
            Err(Error::LengthMismatch { left, right })
        } else {
            Ok(left)
        }
    }

    fn logical_extent(&self) -> crate::core::extent::LogicalExtent {
        RowStorage::logical_extent(&self.0)
            .zipped(&RowStorage::logical_extent(&self.1))
            .expect("storage columns have equal logical extents")
    }
    fn set_logical_extent(&mut self, extent: crate::core::extent::LogicalExtent) {
        RowStorage::set_logical_extent(&mut self.0, extent.clone());
        RowStorage::set_logical_extent(&mut self.1, extent);
    }

    fn read(&self) -> Self::Read {
        Zip::new(self.0.read(), self.1.read())
    }
    fn write(&self) -> Self::Write {
        Zip::new(self.0.write(), self.1.write())
    }
    fn from_item(exec: &Executor<R>, value: Self::Item) -> Self {
        let (left, right) =
            <Left::Item as StorageLayout>::StorageLeaves::split(value.into_storage_leaves());
        Zip::new(
            Left::from_item(exec, Left::Item::from_storage_leaves(left)),
            Right::from_item(exec, Right::Item::from_storage_leaves(right)),
        )
    }
    fn read_first(&self, exec: &Executor<R>) -> Result<Self::Item, Error> {
        let left = self.0.read_first(exec)?.into_storage_leaves();
        let right = self.1.read_first(exec)?.into_storage_leaves();
        Ok(Self::Item::from_storage_leaves(left.concat(right)))
    }

    fn slice<Range: RangeBounds<usize>>(&self, range: Range) -> Self::Read {
        let len = self.len().expect("storage columns have equal lengths");
        let (start, count) = crate::core::read::resolve_slice_range(len, range);
        Zip::new(
            self.0.slice(start..start + count),
            self.1.slice(start..start + count),
        )
    }

    fn slice_mut<Range: RangeBounds<usize>>(&self, range: Range) -> Self::Write {
        let len = self.len().expect("storage columns have equal lengths");
        let (start, count) = crate::core::read::resolve_slice_range(len, range);
        Zip::new(
            self.0.slice_mut(start..start + count),
            self.1.slice_mut(start..start + count),
        )
    }

    fn read_item(&self, exec: &Executor<R>, index: usize) -> Result<Self::Item, Error> {
        let left = self.0.read_item(exec, index)?.into_storage_leaves();
        let right = self.1.read_item(exec, index)?.into_storage_leaves();
        Ok(Self::Item::from_storage_leaves(left.concat(right)))
    }
}

impl<R, Left, Right> CopyStorage<R> for Zip<Left, Right>
where
    R: Runtime,
    Left: CopyStorage<R>,
    Right: CopyStorage<R>,
    Left::Item: FlatRow,
    Right::Item: FlatRow,
    <Left::Item as StorageLayout>::StorageLeaves:
        FlatLeaves<Item = Left::Item> + Concat<<Right::Item as StorageLayout>::StorageLeaves>,
    <Right::Item as StorageLayout>::StorageLeaves: FlatLeaves<Item = Right::Item>,
    <<Left::Item as StorageLayout>::StorageLeaves as Concat<
        <Right::Item as StorageLayout>::StorageLeaves,
    >>::Output: FlatLeaves,
    Zip<Left::Read, Right::Read>: ReadExpression<Item = JoinedRow<Left::Item, Right::Item>>
        + LowerReadExpression
        + StageRead<R, Env0>,
    Zip<Left::Write, Right::Write>: OutputExpression<Item = JoinedRow<Left::Item, Right::Item>>
        + LowerOutputExpression
        + StageOutput<R, Env0>
        + SliceOutput
        + FillOutput<R>,
    <Zip<Left::Write, Right::Write> as LowerOutputExpression>::Slots: PaddedOutputSlots<
        Leaves = <JoinedRow<Left::Item, Right::Item> as StorageLayout>::StorageLeaves,
    >,
{
    fn copy_storage(&self, exec: &Executor<R>, output: Self::Write) -> Result<(), Error> {
        self.0.copy_storage(exec, output.0)?;
        self.1.copy_storage(exec, output.1)
    }
}

/// Allocates the physical SoA storage for one flat row type.
pub trait RowAlloc<R: Runtime>: FlatRow {
    type RowStorage: RowStorage<R, Item = Self> + CopyStorage<R>;
    fn alloc(exec: &Executor<R>, len: usize) -> Self::RowStorage;
}

#[doc(hidden)]
pub trait AllocColumns<R: Runtime> {
    type Storage: RowStorage<R>;
    fn alloc_columns(exec: &Executor<R>, len: usize) -> Self::Storage;
}

/// Allocates internal scratch columns for one flat row type.
pub(crate) trait ScratchStorage<R: Runtime>: crate::api::iter::KernelRow + FlatRow {
    type Storage: RowStorage<R, Item = Self> + CopyStorage<R>;

    fn alloc_scratch(exec: &Executor<R>, len: usize) -> Self::Storage;
}

impl<R, Item> ScratchStorage<R> for Item
where
    R: Runtime,
    Item: crate::api::iter::KernelRow + FlatRow,
    Item::StorageLeaves: FlatLeaves<Item = Item> + AllocColumns<R>,
    <Item::StorageLeaves as AllocColumns<R>>::Storage: RowStorage<R, Item = Item> + CopyStorage<R>,
{
    type Storage = <Item::StorageLeaves as AllocColumns<R>>::Storage;

    fn alloc_scratch(exec: &Executor<R>, len: usize) -> Self::Storage {
        Item::StorageLeaves::alloc_columns(exec, len)
    }
}

impl<R, L0> AllocColumns<R> for Last<L0>
where
    R: Runtime,
    L0: MStorageElement + StorageLayout<StorageArity = S1, StorageLeaves = Last<L0>>,
{
    type Storage = DeviceVec<R, L0>;
    fn alloc_columns(exec: &Executor<R>, len: usize) -> Self::Storage {
        exec.alloc_column::<L0>(len)
    }
}

macro_rules! alloc_left {
    ($name:ident; $leaves:ty; $storage:ty; $value:expr; $( $leaf:ident ),+) => {
        impl<R, $( $leaf ),+> AllocColumns<R> for $leaves
        where
            R: Runtime,
            $( $leaf: MStorageElement + StorageLayout<StorageArity = S1, StorageLeaves = Last<$leaf>>, )+
        {
            type Storage = $storage;
            fn alloc_columns(exec: &Executor<R>, len: usize) -> Self::Storage { $value(exec, len) }
        }
    };
}

fn alloc2<R: Runtime, A: MStorageElement, B: MStorageElement>(
    exec: &Executor<R>,
    len: usize,
) -> Zip<DeviceVec<R, A>, DeviceVec<R, B>> {
    Zip::new(exec.alloc_column::<A>(len), exec.alloc_column::<B>(len))
}
fn alloc3<R: Runtime, A: MStorageElement, B: MStorageElement, C: MStorageElement>(
    exec: &Executor<R>,
    len: usize,
) -> Zip<Zip<DeviceVec<R, A>, DeviceVec<R, B>>, DeviceVec<R, C>> {
    Zip::new(alloc2::<R, A, B>(exec, len), exec.alloc_column::<C>(len))
}
fn alloc4<
    R: Runtime,
    A: MStorageElement,
    B: MStorageElement,
    C: MStorageElement,
    D: MStorageElement,
>(
    exec: &Executor<R>,
    len: usize,
) -> Zip<Zip<Zip<DeviceVec<R, A>, DeviceVec<R, B>>, DeviceVec<R, C>>, DeviceVec<R, D>> {
    Zip::new(alloc3::<R, A, B, C>(exec, len), exec.alloc_column::<D>(len))
}
fn alloc5<
    R: Runtime,
    A: MStorageElement,
    B: MStorageElement,
    C: MStorageElement,
    D: MStorageElement,
    E: MStorageElement,
>(
    exec: &Executor<R>,
    len: usize,
) -> Zip<
    Zip<Zip<Zip<DeviceVec<R, A>, DeviceVec<R, B>>, DeviceVec<R, C>>, DeviceVec<R, D>>,
    DeviceVec<R, E>,
> {
    Zip::new(
        alloc4::<R, A, B, C, D>(exec, len),
        exec.alloc_column::<E>(len),
    )
}
fn alloc6<
    R: Runtime,
    A: MStorageElement,
    B: MStorageElement,
    C: MStorageElement,
    D: MStorageElement,
    E: MStorageElement,
    F: MStorageElement,
>(
    exec: &Executor<R>,
    len: usize,
) -> Zip<
    Zip<
        Zip<Zip<Zip<DeviceVec<R, A>, DeviceVec<R, B>>, DeviceVec<R, C>>, DeviceVec<R, D>>,
        DeviceVec<R, E>,
    >,
    DeviceVec<R, F>,
> {
    Zip::new(
        alloc5::<R, A, B, C, D, E>(exec, len),
        exec.alloc_column::<F>(len),
    )
}
fn alloc7<
    R: Runtime,
    A: MStorageElement,
    B: MStorageElement,
    C: MStorageElement,
    D: MStorageElement,
    E: MStorageElement,
    F: MStorageElement,
    G: MStorageElement,
>(
    exec: &Executor<R>,
    len: usize,
) -> Zip<
    Zip<
        Zip<
            Zip<Zip<Zip<DeviceVec<R, A>, DeviceVec<R, B>>, DeviceVec<R, C>>, DeviceVec<R, D>>,
            DeviceVec<R, E>,
        >,
        DeviceVec<R, F>,
    >,
    DeviceVec<R, G>,
> {
    Zip::new(
        alloc6::<R, A, B, C, D, E, F>(exec, len),
        exec.alloc_column::<G>(len),
    )
}

fn alloc8<
    R: Runtime,
    L0: MStorageElement,
    L1: MStorageElement,
    L2: MStorageElement,
    L3: MStorageElement,
    L4: MStorageElement,
    L5: MStorageElement,
    L6: MStorageElement,
    L7: MStorageElement,
>(
    exec: &Executor<R>,
    len: usize,
) -> Zip<
    Zip<
        Zip<
            Zip<
                Zip<
                    Zip<Zip<DeviceVec<R, L0>, DeviceVec<R, L1>>, DeviceVec<R, L2>>,
                    DeviceVec<R, L3>,
                >,
                DeviceVec<R, L4>,
            >,
            DeviceVec<R, L5>,
        >,
        DeviceVec<R, L6>,
    >,
    DeviceVec<R, L7>,
> {
    Zip::new(
        alloc7::<R, L0, L1, L2, L3, L4, L5, L6>(exec, len),
        exec.alloc_column::<L7>(len),
    )
}

fn alloc9<
    R: Runtime,
    L0: MStorageElement,
    L1: MStorageElement,
    L2: MStorageElement,
    L3: MStorageElement,
    L4: MStorageElement,
    L5: MStorageElement,
    L6: MStorageElement,
    L7: MStorageElement,
    L8: MStorageElement,
>(
    exec: &Executor<R>,
    len: usize,
) -> Zip<
    Zip<
        Zip<
            Zip<
                Zip<
                    Zip<
                        Zip<Zip<DeviceVec<R, L0>, DeviceVec<R, L1>>, DeviceVec<R, L2>>,
                        DeviceVec<R, L3>,
                    >,
                    DeviceVec<R, L4>,
                >,
                DeviceVec<R, L5>,
            >,
            DeviceVec<R, L6>,
        >,
        DeviceVec<R, L7>,
    >,
    DeviceVec<R, L8>,
> {
    Zip::new(
        alloc8::<R, L0, L1, L2, L3, L4, L5, L6, L7>(exec, len),
        exec.alloc_column::<L8>(len),
    )
}

fn alloc10<
    R: Runtime,
    L0: MStorageElement,
    L1: MStorageElement,
    L2: MStorageElement,
    L3: MStorageElement,
    L4: MStorageElement,
    L5: MStorageElement,
    L6: MStorageElement,
    L7: MStorageElement,
    L8: MStorageElement,
    L9: MStorageElement,
>(
    exec: &Executor<R>,
    len: usize,
) -> Zip<
    Zip<
        Zip<
            Zip<
                Zip<
                    Zip<
                        Zip<
                            Zip<Zip<DeviceVec<R, L0>, DeviceVec<R, L1>>, DeviceVec<R, L2>>,
                            DeviceVec<R, L3>,
                        >,
                        DeviceVec<R, L4>,
                    >,
                    DeviceVec<R, L5>,
                >,
                DeviceVec<R, L6>,
            >,
            DeviceVec<R, L7>,
        >,
        DeviceVec<R, L8>,
    >,
    DeviceVec<R, L9>,
> {
    Zip::new(
        alloc9::<R, L0, L1, L2, L3, L4, L5, L6, L7, L8>(exec, len),
        exec.alloc_column::<L9>(len),
    )
}

fn alloc11<
    R: Runtime,
    L0: MStorageElement,
    L1: MStorageElement,
    L2: MStorageElement,
    L3: MStorageElement,
    L4: MStorageElement,
    L5: MStorageElement,
    L6: MStorageElement,
    L7: MStorageElement,
    L8: MStorageElement,
    L9: MStorageElement,
    L10: MStorageElement,
>(
    exec: &Executor<R>,
    len: usize,
) -> Zip<
    Zip<
        Zip<
            Zip<
                Zip<
                    Zip<
                        Zip<
                            Zip<
                                Zip<Zip<DeviceVec<R, L0>, DeviceVec<R, L1>>, DeviceVec<R, L2>>,
                                DeviceVec<R, L3>,
                            >,
                            DeviceVec<R, L4>,
                        >,
                        DeviceVec<R, L5>,
                    >,
                    DeviceVec<R, L6>,
                >,
                DeviceVec<R, L7>,
            >,
            DeviceVec<R, L8>,
        >,
        DeviceVec<R, L9>,
    >,
    DeviceVec<R, L10>,
> {
    Zip::new(
        alloc10::<R, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9>(exec, len),
        exec.alloc_column::<L10>(len),
    )
}

fn alloc12<
    R: Runtime,
    L0: MStorageElement,
    L1: MStorageElement,
    L2: MStorageElement,
    L3: MStorageElement,
    L4: MStorageElement,
    L5: MStorageElement,
    L6: MStorageElement,
    L7: MStorageElement,
    L8: MStorageElement,
    L9: MStorageElement,
    L10: MStorageElement,
    L11: MStorageElement,
>(
    exec: &Executor<R>,
    len: usize,
) -> Zip<
    Zip<
        Zip<
            Zip<
                Zip<
                    Zip<
                        Zip<
                            Zip<
                                Zip<
                                    Zip<Zip<DeviceVec<R, L0>, DeviceVec<R, L1>>, DeviceVec<R, L2>>,
                                    DeviceVec<R, L3>,
                                >,
                                DeviceVec<R, L4>,
                            >,
                            DeviceVec<R, L5>,
                        >,
                        DeviceVec<R, L6>,
                    >,
                    DeviceVec<R, L7>,
                >,
                DeviceVec<R, L8>,
            >,
            DeviceVec<R, L9>,
        >,
        DeviceVec<R, L10>,
    >,
    DeviceVec<R, L11>,
> {
    Zip::new(
        alloc11::<R, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10>(exec, len),
        exec.alloc_column::<L11>(len),
    )
}

alloc_left!(A2; More<L0,Last<L1>>; Zip<DeviceVec<R,L0>,DeviceVec<R,L1>>; alloc2::<R,L0,L1>; L0,L1);
alloc_left!(A3; More<L0,More<L1,Last<L2>>>; Zip<Zip<DeviceVec<R,L0>,DeviceVec<R,L1>>,DeviceVec<R,L2>>; alloc3::<R,L0,L1,L2>; L0,L1,L2);
alloc_left!(A4; More<L0,More<L1,More<L2,Last<L3>>>>; Zip<Zip<Zip<DeviceVec<R,L0>,DeviceVec<R,L1>>,DeviceVec<R,L2>>,DeviceVec<R,L3>>; alloc4::<R,L0,L1,L2,L3>; L0,L1,L2,L3);
alloc_left!(A5; More<L0,More<L1,More<L2,More<L3,Last<L4>>>>>; Zip<Zip<Zip<Zip<DeviceVec<R,L0>,DeviceVec<R,L1>>,DeviceVec<R,L2>>,DeviceVec<R,L3>>,DeviceVec<R,L4>>; alloc5::<R,L0,L1,L2,L3,L4>; L0,L1,L2,L3,L4);
alloc_left!(A6; More<L0,More<L1,More<L2,More<L3,More<L4,Last<L5>>>>>>; Zip<Zip<Zip<Zip<Zip<DeviceVec<R,L0>,DeviceVec<R,L1>>,DeviceVec<R,L2>>,DeviceVec<R,L3>>,DeviceVec<R,L4>>,DeviceVec<R,L5>>; alloc6::<R,L0,L1,L2,L3,L4,L5>; L0,L1,L2,L3,L4,L5);
alloc_left!(A7; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,Last<L6>>>>>>>; Zip<Zip<Zip<Zip<Zip<Zip<DeviceVec<R,L0>,DeviceVec<R,L1>>,DeviceVec<R,L2>>,DeviceVec<R,L3>>,DeviceVec<R,L4>>,DeviceVec<R,L5>>,DeviceVec<R,L6>>; alloc7::<R,L0,L1,L2,L3,L4,L5,L6>; L0,L1,L2,L3,L4,L5,L6);
alloc_left!(A8; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,More<L6,Last<L7>>>>>>>>; Zip<Zip<Zip<Zip<Zip<Zip<Zip<DeviceVec<R, L0>, DeviceVec<R, L1>>, DeviceVec<R, L2>>, DeviceVec<R, L3>>, DeviceVec<R, L4>>, DeviceVec<R, L5>>, DeviceVec<R, L6>>, DeviceVec<R, L7>>; alloc8::<R,L0,L1,L2,L3,L4,L5,L6,L7>; L0,L1,L2,L3,L4,L5,L6,L7);
alloc_left!(A9; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,More<L6,More<L7,Last<L8>>>>>>>>>; Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<DeviceVec<R, L0>, DeviceVec<R, L1>>, DeviceVec<R, L2>>, DeviceVec<R, L3>>, DeviceVec<R, L4>>, DeviceVec<R, L5>>, DeviceVec<R, L6>>, DeviceVec<R, L7>>, DeviceVec<R, L8>>; alloc9::<R,L0,L1,L2,L3,L4,L5,L6,L7,L8>; L0,L1,L2,L3,L4,L5,L6,L7,L8);
alloc_left!(A10; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,More<L6,More<L7,More<L8,Last<L9>>>>>>>>>>; Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<DeviceVec<R, L0>, DeviceVec<R, L1>>, DeviceVec<R, L2>>, DeviceVec<R, L3>>, DeviceVec<R, L4>>, DeviceVec<R, L5>>, DeviceVec<R, L6>>, DeviceVec<R, L7>>, DeviceVec<R, L8>>, DeviceVec<R, L9>>; alloc10::<R,L0,L1,L2,L3,L4,L5,L6,L7,L8,L9>; L0,L1,L2,L3,L4,L5,L6,L7,L8,L9);
alloc_left!(A11; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,More<L6,More<L7,More<L8,More<L9,Last<L10>>>>>>>>>>>; Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<DeviceVec<R, L0>, DeviceVec<R, L1>>, DeviceVec<R, L2>>, DeviceVec<R, L3>>, DeviceVec<R, L4>>, DeviceVec<R, L5>>, DeviceVec<R, L6>>, DeviceVec<R, L7>>, DeviceVec<R, L8>>, DeviceVec<R, L9>>, DeviceVec<R, L10>>; alloc11::<R,L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10>; L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10);
alloc_left!(A12; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,More<L6,More<L7,More<L8,More<L9,More<L10,Last<L11>>>>>>>>>>>>; Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<DeviceVec<R, L0>, DeviceVec<R, L1>>, DeviceVec<R, L2>>, DeviceVec<R, L3>>, DeviceVec<R, L4>>, DeviceVec<R, L5>>, DeviceVec<R, L6>>, DeviceVec<R, L7>>, DeviceVec<R, L8>>, DeviceVec<R, L9>>, DeviceVec<R, L10>>, DeviceVec<R, L11>>; alloc12::<R,L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11>; L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11);

impl<R, Item> RowAlloc<R> for Item
where
    R: Runtime,
    Item: FlatRow,
    Item::StorageLeaves: FlatLeaves<Item = Item> + AllocColumns<R>,
    <Item::StorageLeaves as AllocColumns<R>>::Storage: RowStorage<R, Item = Item> + CopyStorage<R>,
{
    type RowStorage = <Item::StorageLeaves as AllocColumns<R>>::Storage;

    fn alloc(exec: &Executor<R>, len: usize) -> Self::RowStorage {
        Item::StorageLeaves::alloc_columns(exec, len)
    }
}

/// Appends one owned column to an already-flat column value.
pub trait AppendColumn<Right> {
    type Output;
    fn append_column(self, right: Right) -> Self::Output;
}

impl<R, Left, Right> AppendColumn<DeviceVec<R, Right>> for DeviceVec<R, Left>
where
    R: Runtime,
    Left: MStorageElement,
    Right: MStorageElement,
{
    type Output = (DeviceVec<R, Left>, DeviceVec<R, Right>);

    fn append_column(self, right: DeviceVec<R, Right>) -> Self::Output {
        (self, right)
    }
}

macro_rules! impl_append_column {
    ($( $item:ident : $index:tt ),+ $(,)?) => {
        impl<R, $( $item, )+ Next> AppendColumn<DeviceVec<R, Next>>
            for ($(DeviceVec<R, $item>,)+)
        where
            R: Runtime,
            $( $item: MStorageElement, )+
            Next: MStorageElement,
        {
            type Output = ($(DeviceVec<R, $item>,)+ DeviceVec<R, Next>);

            fn append_column(self, right: DeviceVec<R, Next>) -> Self::Output {
                ($(self.$index,)+ right)
            }
        }
    };
}

impl_append_column!(A: 0, B: 1);
impl_append_column!(A: 0, B: 1, C: 2);
impl_append_column!(A: 0, B: 1, C: 2, D: 3);
impl_append_column!(A: 0, B: 1, C: 2, D: 3, E: 4);
impl_append_column!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5);
impl_append_column!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6);
impl_append_column!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7);
impl_append_column!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8);
impl_append_column!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8, J: 9);
impl_append_column!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8, J: 9, K: 10);

#[doc(hidden)]
#[allow(opaque_hidden_inferred_bound)]
impl<R, T> MStorage<R> for DeviceVec<R, T>
where
    R: Runtime,
    T: MStorageElement + StorageLayout<StorageArity = S1, StorageLeaves = Last<T>>,
    Last<T>: crate::core::facade::KernelValue,
    Column<T>: crate::api::iter::MIter<R, Item = T>,
    crate::DeviceSlice<R, T>: crate::api::iter::MIter<R, Item = T>,
    crate::DeviceSliceMut<R, T>: crate::api::iter::MIterMut<R, Item = T>,
{
    type Item = T;
    type Columns = Self;
    type Slice<'a> = crate::DeviceSlice<R, T>;
    type SliceMut<'a> = crate::DeviceSliceMut<R, T>;

    fn allocate(exec: &Executor<R>, len: crate::MIndex) -> Self {
        exec.alloc_column::<T>(len as usize)
    }

    fn into_columns(self) -> Self::Columns {
        self
    }

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice<'_>
    where
        Bounds: RangeBounds<crate::MIndex>,
    {
        let (start, count) = crate::core::read::resolve_mindex_slice_range(self.capacity(), range);
        crate::DeviceSlice::from_column(self.slice_usize(start..start + count))
    }

    fn slice_mut<Bounds>(&self, range: Bounds) -> Self::SliceMut<'_>
    where
        Bounds: RangeBounds<crate::MIndex>,
    {
        let (start, count) = crate::core::read::resolve_mindex_slice_range(self.capacity(), range);
        crate::DeviceSliceMut::from_output(self.slice_mut_usize(start..start + count))
    }
}

impl<R: Runtime, T> MStorageExtent<R> for DeviceVec<R, T> {
    fn capacity(&self) -> Result<crate::MIndex, Error> {
        crate::api::iter::checked_len(self.capacity())
    }

    fn logical_extent(&self) -> crate::core::extent::LogicalExtent {
        DeviceVec::logical_extent(self)
    }

    fn set_logical_extent(&mut self, extent: crate::core::extent::LogicalExtent) {
        DeviceVec::set_logical_extent(self, extent);
    }
}

#[doc(hidden)]
impl<R, Left, Right> MStorage<R> for Zip<Left, Right>
where
    R: Runtime,
    Left: MStorage<R>,
    Right: MStorage<R>,
    Self: RowStorage<R>,
    <Self as RowStorage<R>>::Item: ScratchStorage<R>,
    <<Self as RowStorage<R>>::Item as StorageLayout>::StorageLeaves:
        crate::core::facade::KernelValue,
    <Self as RowStorage<R>>::Read: crate::api::iter::MIter<R, Item = <Self as RowStorage<R>>::Item>,
    <Self as RowStorage<R>>::Write:
        crate::api::iter::MIterMut<R, Item = <Self as RowStorage<R>>::Item>,
    Left::Columns: AppendColumn<Right::Columns>,
    for<'a> StorageSlice<'a, R, Self>: MIter<R, Item = <Self as RowStorage<R>>::Item>,
    for<'a> StorageSliceMut<'a, R, Self>: MIterMut<R, Item = <Self as RowStorage<R>>::Item>,
{
    type Item = <Self as RowStorage<R>>::Item;
    type Columns = <Left::Columns as AppendColumn<Right::Columns>>::Output;
    type Slice<'a>
        = StorageSlice<'a, R, Self>
    where
        Self: 'a;
    type SliceMut<'a>
        = StorageSliceMut<'a, R, Self>
    where
        Self: 'a;

    fn allocate(exec: &Executor<R>, len: crate::MIndex) -> Self {
        Zip::new(Left::allocate(exec, len), Right::allocate(exec, len))
    }

    fn into_columns(self) -> Self::Columns {
        self.0.into_columns().append_column(self.1.into_columns())
    }

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice<'_>
    where
        Bounds: RangeBounds<crate::MIndex>,
    {
        let len = MStorageExtent::capacity(self).expect("storage columns have equal lengths");
        let (start, count) = crate::core::read::resolve_mindex_slice_range(len as usize, range);
        StorageSlice::new(self, start, count)
    }

    fn slice_mut<Bounds>(&self, range: Bounds) -> Self::SliceMut<'_>
    where
        Bounds: RangeBounds<crate::MIndex>,
    {
        let len = MStorageExtent::capacity(self).expect("storage columns have equal lengths");
        let (start, count) = crate::core::read::resolve_mindex_slice_range(len as usize, range);
        StorageSliceMut::new(self, start, count)
    }
}

impl<R, Left, Right> MStorageExtent<R> for Zip<Left, Right>
where
    R: Runtime,
    Left: MStorage<R>,
    Right: MStorage<R>,
    Self: RowStorage<R>,
{
    fn capacity(&self) -> Result<crate::MIndex, Error> {
        crate::api::iter::checked_len(RowStorage::len(self)?)
    }

    fn logical_extent(&self) -> crate::core::extent::LogicalExtent {
        MStorageExtent::logical_extent(&self.0)
            .zipped(&MStorageExtent::logical_extent(&self.1))
            .expect("storage columns have equal logical extents")
    }

    fn set_logical_extent(&mut self, extent: crate::core::extent::LogicalExtent) {
        MStorageExtent::set_logical_extent(&mut self.0, extent.clone());
        MStorageExtent::set_logical_extent(&mut self.1, extent);
    }
}

impl<R: Runtime> Executor<R> {
    /// Allocates uninitialized device storage for `len` flat rows.
    ///
    /// `len` is an [`crate::MIndex`] so every allocated row has a representable
    /// index and a full-length index or permutation vector can always be
    /// allocated alongside it.
    ///
    /// The storage must be completely written before it is read.
    ///
    /// # Examples
    ///
    /// ```
    /// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    /// use massively::{Executor, lazy, vector::{fill, scatter}};
    ///
    /// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    /// let output = exec.alloc::<u32>(4);
    /// let values = exec.alloc::<u32>(4);
    /// fill(&exec, 7_u32, values.slice_mut(..)).unwrap();
    /// scatter(
    ///     &exec,
    ///     values.slice(..),
    ///     lazy::counting(0).take(4),
    ///     output.slice_mut(..),
    /// ).unwrap();
    ///
    /// assert_eq!(exec.to_host(&output).unwrap(), vec![7, 7, 7, 7]);
    /// ```
    ///
    /// A host-sized length must be checked and converted explicitly:
    ///
    /// ```compile_fail
    /// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    /// use massively::Executor;
    ///
    /// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    /// let host_len: usize = 4;
    /// let _ = exec.alloc::<u32>(host_len);
    /// ```
    pub fn alloc<Item: MAlloc<R>>(&self, len: crate::MIndex) -> crate::MVec<R, Item> {
        <crate::MVec<R, Item> as MStorage<R>>::allocate(self, len)
    }

    pub(crate) fn alloc_row<Item: RowAlloc<R>>(
        &self,
        len: usize,
    ) -> <Item as RowAlloc<R>>::RowStorage {
        <Item as RowAlloc<R>>::alloc(self, len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    use static_assertions::{assert_impl_all, assert_not_impl_any};

    type FlatRow3 = (u32, f32, i32);
    type NestedRow3 = (u32, (f32, i32));

    assert_impl_all!(FlatRow3: MAlloc<WgpuRuntime>);
    assert_not_impl_any!(NestedRow3: MAlloc<WgpuRuntime>);

    #[test]
    fn alloc_stores_a_flat_row_in_independent_columns() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let a = exec.to_device(&[1_u32, 2, 3]);
        let b = exec.to_device(&[10_f32, 20.0, 30.0]);
        let c = exec.to_device(&[-1_i32, -2, -3]);
        let storage = exec.alloc_row::<FlatRow3>(3);

        let input = crate::api::iter::lower::<WgpuRuntime, _>(crate::zip3(
            a.column(),
            b.column(),
            c.column(),
        ));
        crate::core::transform::materialize(&exec, input, storage.write()).unwrap();

        let (a, b, c) = MStorage::into_columns(storage);
        assert_eq!(exec.to_host(&a).unwrap(), vec![1, 2, 3]);
        assert_eq!(exec.to_host(&b).unwrap(), vec![10.0, 20.0, 30.0]);
        assert_eq!(exec.to_host(&c).unwrap(), vec![-1, -2, -3]);
    }

    #[test]
    fn fill_writes_and_exposes_flat_columns() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let storage = exec.alloc::<FlatRow3>(2);
        crate::api::algorithm::fill(&exec, (7, 3.5, -2), MStorage::slice_mut(&storage, ..))
            .unwrap();
        let (a, b, c) = storage.into_columns();

        assert_eq!(exec.to_host(&a).unwrap(), vec![7, 7]);
        assert_eq!(exec.to_host(&b).unwrap(), vec![3.5, 3.5]);
        assert_eq!(exec.to_host(&c).unwrap(), vec![-2, -2]);
    }
}
