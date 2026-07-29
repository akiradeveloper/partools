#![allow(private_interfaces)]

use crate::core::{extent, output, read, storage};

use cubecl::prelude::Runtime;

use crate::Error;

/// Exact-arity form of a logical iterator read.
///
/// The expression keeps its recursively computed [`crate::core::read::ReadExpression::ReadArity`]
/// until a consumer chooses a kernel ABI.  [`KernelInput::into_fixed`] is the
/// current launch adapter; exact-arity consumers can use this type directly.
pub trait KernelInput<R: Runtime>:
    Clone + read::ReadExpression + read::LowerReadExpression + read::StageRead<R, read::Env0>
{
    /// Selects the fixed launch ABI without changing the underlying read tree.
    fn into_fixed(self) -> read::FixedRead<Self> {
        read::FixedRead::new(self)
    }
}

impl<R, Input> KernelInput<R> for Input
where
    R: Runtime,
    Input:
        Clone + read::ReadExpression + read::LowerReadExpression + read::StageRead<R, read::Env0>,
{
}

/// Physical output leaves supported by the current fixed write ABI.
pub trait KernelOutputLeaves:
    storage::StorePadded12
    + cubecl::prelude::CubeType<ExpandType: storage::StorePadded12Expand>
    + Send
    + Sync
    + 'static
{
}

impl<Leaves> KernelOutputLeaves for Leaves
where
    Leaves: storage::StorePadded12 + Send + Sync + 'static,
    <Leaves as cubecl::prelude::CubeType>::ExpandType: storage::StorePadded12Expand,
{
}

/// A preallocated output tree that can be staged through the current
/// twelve-slot write ABI.
///
/// This is purely a property of the destination buffers.  It does not imply
/// that the source value has a storage layout or that new storage can be
/// allocated for either value type.
pub trait KernelOutput<R: Runtime>:
    output::OutputExpression<Item: storage::StorageLayout<StorageLeaves: KernelOutputLeaves>>
    + output::LowerOutputExpression<
        Slots: output::PaddedOutputSlots<
            Leaves = <Self::Item as storage::StorageLayout>::StorageLeaves,
        >,
    > + output::StageOutput<R, read::Env0>
{
}

impl<R, Output> KernelOutput<R> for Output
where
    R: Runtime,
    Output: output::OutputExpression
        + output::LowerOutputExpression
        + output::StageOutput<R, read::Env0>,
    Output::Slots: output::PaddedOutputSlots<
        Leaves = <Output::Item as storage::StorageLayout>::StorageLeaves,
    >,
    <Output::Item as storage::StorageLayout>::StorageLeaves: storage::StorePadded12,
    <<Output::Item as storage::StorageLayout>::StorageLeaves as cubecl::prelude::CubeType>::ExpandType:
        storage::StorePadded12Expand,
{
}

/// Device-side operations that follow directly from an item's physical leaf
/// layout. This trait has no algorithm dispatch methods.
pub trait KernelValue:
    Sized
    + Send
    + Sync
    + 'static
    + storage::SelectLeaves
    + storage::SharedLeaves
    + storage::MutableLeaves
    + storage::PlaneShuffleLeaves
    + storage::LoadPadded12
    + storage::LoadMutPadded12
    + output::OutputSlotLayout<
        Slots: output::OutputSlotEnvironment<StorageArity = Self::StorageArity>,
    >
{
    type StorageArity: storage::StorageArity;
}

impl<Leaves> KernelValue for Leaves
where
    Leaves: Sized
        + Send
        + Sync
        + 'static
        + storage::SelectLeaves
        + storage::SharedLeaves
        + storage::MutableLeaves
        + storage::PlaneShuffleLeaves
        + storage::LoadPadded12
        + storage::LoadMutPadded12
        + output::OutputSlotLayout,
    <Leaves as output::OutputSlotLayout>::Slots: output::OutputSlotEnvironment,
{
    type StorageArity = <<Leaves as output::OutputSlotLayout>::Slots as output::OutputSlotEnvironment>::StorageArity;
}

pub(crate) fn physical_len<R, Input>(input: &Input) -> Result<usize, Error>
where
    R: Runtime,
    Input: KernelInput<R>,
{
    <Input as read::StageRead<R, read::Env0>>::physical_len(input)
}

pub(crate) fn logical_extent<R, Input>(input: &Input) -> Result<extent::LogicalExtent, Error>
where
    R: Runtime,
    Input: KernelInput<R>,
{
    <Input as read::StageRead<R, read::Env0>>::logical_extent(input)
}
