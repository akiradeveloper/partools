//! Host-side staging of read expressions in physical leaf order.

use crate::core::{extent, read};

use cubecl::prelude::*;

use crate::Error;
use crate::core::bindings::Bindings;
use crate::core::iter::Zip;
use crate::core::value::MStorageElement;
use crate::op::{ReductionOp, UnaryOp};

use super::{
    BindSlots, Column, Constant, Counting, DivModCounting, Env0, FixedRead, LowerReadExpression,
    Permute, ReadExpression, ReverseCounting, Stride, Taken, TakenSource, Transform,
};

/// Host-side staging following the same left-first recursion as [`BindSlots`].
#[doc(hidden)]
pub trait StageRead<R: Runtime, Env>: BindSlots<Env> {
    /// Host-known bound for allocation and dispatch, including any inactive tail.
    fn physical_len(&self) -> Result<usize, Error>;

    /// Initialized prefix; resolving it on the host is an explicit boundary.
    fn logical_extent(&self) -> Result<extent::LogicalExtent, Error> {
        Ok(extent::LogicalExtent::fixed(self.physical_len()?))
    }
    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut Bindings,
    ) -> Result<(), Error>;
}

impl<R, T, Env> StageRead<R, Env> for Column<T>
where
    R: Runtime,
    T: MStorageElement,
    Column<T>: BindSlots<Env>,
{
    fn physical_len(&self) -> Result<usize, Error> {
        Ok(self.len)
    }

    fn logical_extent(&self) -> Result<extent::LogicalExtent, Error> {
        Ok(self.extent.clone())
    }

    fn stage_at(
        &self,
        _client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut Bindings,
    ) -> Result<(), Error> {
        if self.owner != Some(owner) {
            return Err(Error::ForeignExecutor);
        }
        let handle = self.handle.clone().ok_or(Error::UnboundColumn)?;
        bindings.push(handle, self.buffer_len, self.offset);
        Ok(())
    }
}

impl<R, T, Env> StageRead<R, Env> for Constant<T>
where
    R: Runtime,
    T: MStorageElement,
    Constant<T>: BindSlots<Env>,
{
    fn physical_len(&self) -> Result<usize, Error> {
        Ok(self.len)
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        _owner: u64,
        bindings: &mut Bindings,
    ) -> Result<(), Error> {
        let handle = client.create_from_slice(T::as_bytes(&[self.value]));
        bindings.push(handle, 1, 0);
        Ok(())
    }
}

impl<R, T, Env> StageRead<R, Env> for read::Value<T>
where
    R: Runtime,
    T: MStorageElement,
    read::Value<T>: BindSlots<Env>,
{
    fn physical_len(&self) -> Result<usize, Error> {
        Ok(1)
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut Bindings,
    ) -> Result<(), Error> {
        let handle = match self {
            read::Value::Host(value) => client.create_from_slice(T::as_bytes(&[*value])),
            read::Value::Device {
                handle,
                owner: value_owner,
            } => {
                if *value_owner != owner {
                    return Err(Error::ForeignExecutor);
                }
                handle.clone()
            }
        };
        bindings.push(handle, 1, 0);
        Ok(())
    }
}

impl<R, Env> StageRead<R, Env> for Counting
where
    R: Runtime,
    Counting: BindSlots<Env>,
{
    fn physical_len(&self) -> Result<usize, Error> {
        Ok(self.len)
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        _owner: u64,
        bindings: &mut Bindings,
    ) -> Result<(), Error> {
        let handle = client.create_from_slice(u32::as_bytes(&[self.start]));
        bindings.push(handle, 1, 0);
        Ok(())
    }
}

impl<R, Env> StageRead<R, Env> for Stride
where
    R: Runtime,
    Stride: BindSlots<Env>,
{
    fn physical_len(&self) -> Result<usize, Error> {
        Ok(self.len)
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        _owner: u64,
        bindings: &mut Bindings,
    ) -> Result<(), Error> {
        let handle = client.create_from_slice(u32::as_bytes(&[self.start, self.step]));
        bindings.push(handle, 2, 0);
        Ok(())
    }
}

impl<R, Env> StageRead<R, Env> for DivModCounting
where
    R: Runtime,
    DivModCounting: BindSlots<Env>,
{
    fn physical_len(&self) -> Result<usize, Error> {
        Ok(self.len)
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        _owner: u64,
        bindings: &mut Bindings,
    ) -> Result<(), Error> {
        let handle =
            client.create_from_slice(u32::as_bytes(&[self.start, self.divisor, self.modulus]));
        bindings.push(handle, 3, 0);
        Ok(())
    }
}

impl<R, Env> StageRead<R, Env> for ReverseCounting
where
    R: Runtime,
    ReverseCounting: BindSlots<Env>,
{
    fn physical_len(&self) -> Result<usize, Error> {
        Ok(self.len)
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        _owner: u64,
        bindings: &mut Bindings,
    ) -> Result<(), Error> {
        let start = u32::try_from(self.start).map_err(|_| Error::LengthTooLarge {
            len: self.start.saturating_add(1),
        })?;
        let handle = client.create_from_slice(u32::as_bytes(&[start]));
        bindings.push(handle, 1, 0);
        Ok(())
    }
}

impl<R, Source, Env> StageRead<R, Env> for Taken<Source>
where
    R: Runtime,
    Source: TakenSource,
    Source::Read: StageRead<R, Env>,
    Taken<Source>: BindSlots<Env>,
{
    fn physical_len(&self) -> Result<usize, Error> {
        Ok(self.len as usize)
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut Bindings,
    ) -> Result<(), Error> {
        self.lower().stage_at(client, owner, bindings)
    }
}

impl<R, Left, Right, Env> StageRead<R, Env> for Zip<Left, Right>
where
    R: Runtime,
    Left: StageRead<R, Env>,
    Right: StageRead<R, Left::NextEnv>,
    Zip<Left, Right>: BindSlots<Env>,
{
    fn physical_len(&self) -> Result<usize, Error> {
        let left = self.0.physical_len()?;
        let right = self.1.physical_len()?;
        if left != right {
            return Err(Error::LengthMismatch { left, right });
        }
        Ok(left)
    }

    fn logical_extent(&self) -> Result<extent::LogicalExtent, Error> {
        self.0.logical_extent()?.zipped(&self.1.logical_extent()?)
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut Bindings,
    ) -> Result<(), Error> {
        self.0.stage_at(client, owner, bindings)?;
        self.1.stage_at(client, owner, bindings)
    }
}

impl<R, Values, Offsets, Env> StageRead<R, Env> for crate::seg::SegmentRead<Values, Offsets>
where
    R: Runtime,
    Values: StageRead<R, Env>,
    Offsets: StageRead<R, Values::NextEnv>,
    crate::seg::SegmentRead<Values, Offsets>: BindSlots<Env>,
{
    fn physical_len(&self) -> Result<usize, Error> {
        crate::seg::segment_count(self.offsets().physical_len()?)
    }

    fn logical_extent(&self) -> Result<extent::LogicalExtent, Error> {
        Ok(self
            .offsets()
            .logical_extent()?
            .slice(1, self.physical_len()?))
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut Bindings,
    ) -> Result<(), Error> {
        self.values().stage_at(client, owner, bindings)?;
        self.offsets().stage_at(client, owner, bindings)
    }
}

impl<R, Input, Op, Env> StageRead<R, Env> for Transform<Input, Op>
where
    R: Runtime,
    Input: ReadExpression + StageRead<R, Env>,
    Op: UnaryOp<Input::Item>,
    Transform<Input, Op>: BindSlots<Env>,
{
    fn physical_len(&self) -> Result<usize, Error> {
        self.input.physical_len()
    }

    fn logical_extent(&self) -> Result<extent::LogicalExtent, Error> {
        self.input.logical_extent()
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut Bindings,
    ) -> Result<(), Error> {
        self.input.stage_at(client, owner, bindings)
    }
}

impl<R, Input, Op, Env> StageRead<R, Env> for read::IndexedTransform<Input, Op>
where
    R: Runtime,
    Input: ReadExpression + StageRead<R, Env>,
    Op: crate::op::IndexedUnaryOp<Input::Item>,
    read::IndexedTransform<Input, Op>: BindSlots<Env>,
{
    fn physical_len(&self) -> Result<usize, Error> {
        self.input.physical_len()
    }

    fn logical_extent(&self) -> Result<extent::LogicalExtent, Error> {
        self.input.logical_extent()
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut Bindings,
    ) -> Result<(), Error> {
        self.input.stage_at(client, owner, bindings)
    }
}

impl<R, Input, Op, Env> StageRead<R, Env> for read::AdjacentIndexedTransform<Input, Op>
where
    R: Runtime,
    Input: ReadExpression + StageRead<R, Env>,
    Op: crate::op::IndexedBinaryOp<Input::Item>,
    read::AdjacentIndexedTransform<Input, Op>: BindSlots<Env>,
{
    fn physical_len(&self) -> Result<usize, Error> {
        self.input.physical_len()
    }

    fn logical_extent(&self) -> Result<extent::LogicalExtent, Error> {
        self.input.logical_extent()
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut Bindings,
    ) -> Result<(), Error> {
        self.input.stage_at(client, owner, bindings)
    }
}

impl<R, Input, Op, Env> StageRead<R, Env> for read::Adjacent<Input, Op>
where
    R: Runtime,
    Input: ReadExpression + StageRead<R, Env>,
    Op: ReductionOp<Input::Item>,
    read::Adjacent<Input, Op>: BindSlots<Env>,
{
    fn physical_len(&self) -> Result<usize, Error> {
        self.input.physical_len()
    }

    fn logical_extent(&self) -> Result<extent::LogicalExtent, Error> {
        self.input.logical_extent()
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut Bindings,
    ) -> Result<(), Error> {
        self.input.stage_at(client, owner, bindings)
    }
}

impl<R, Input, Env> StageRead<R, Env> for read::Slice<R, Input>
where
    R: Runtime,
    Input: StageRead<R, Env>,
    read::Slice<R, Input>: BindSlots<Env>,
{
    fn physical_len(&self) -> Result<usize, Error> {
        self.input.physical_len()
    }

    fn logical_extent(&self) -> Result<extent::LogicalExtent, Error> {
        self.input.logical_extent()
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut Bindings,
    ) -> Result<(), Error> {
        self.input.stage_at(client, owner, bindings)
    }
}

impl<R, Values, Indices, Env> StageRead<R, Env> for Permute<Values, Indices>
where
    R: Runtime,
    Values: StageRead<R, Env>,
    Indices: StageRead<R, Values::NextEnv>,
    Permute<Values, Indices>: BindSlots<Env>,
{
    fn physical_len(&self) -> Result<usize, Error> {
        self.indices.physical_len()
    }

    fn logical_extent(&self) -> Result<extent::LogicalExtent, Error> {
        self.indices.logical_extent()
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut Bindings,
    ) -> Result<(), Error> {
        self.values.stage_at(client, owner, bindings)?;
        self.indices.stage_at(client, owner, bindings)
    }
}

impl<R, Input, Env> StageRead<R, Env> for read::Repeat<Input>
where
    R: Runtime,
    Input: StageRead<R, Env>,
    read::Repeat<Input>: BindSlots<Env>,
{
    fn physical_len(&self) -> Result<usize, Error> {
        Ok(self.len)
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut Bindings,
    ) -> Result<(), Error> {
        self.input.stage_at(client, owner, bindings)
    }
}

impl<R, Values, Env> StageRead<R, Env> for read::Reverse<Values>
where
    R: Runtime,
    Values: StageRead<R, Env>,
    read::Reverse<Values>: BindSlots<Env>,
{
    fn physical_len(&self) -> Result<usize, Error> {
        match self.len {
            Some(len) => Ok(len),
            None => self.values.physical_len(),
        }
    }

    fn logical_extent(&self) -> Result<extent::LogicalExtent, Error> {
        let capacity = self.physical_len()?;
        Ok(self.values.logical_extent()?.slice(self.offset, capacity))
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut Bindings,
    ) -> Result<(), Error> {
        let exec = crate::Executor::from_client(client, owner);
        let start = self
            .values
            .logical_extent()?
            .reverse_start(&exec, self.offset)?;
        self.values.stage_at(client, owner, bindings)?;
        bindings.push(start.handle.clone(), 1, 0);
        Ok(())
    }
}

impl<R, Input> StageRead<R, Env0> for FixedRead<Input>
where
    R: Runtime,
    Input: LowerReadExpression + StageRead<R, Env0>,
{
    fn physical_len(&self) -> Result<usize, Error> {
        StageRead::physical_len(&self.input)
    }

    fn logical_extent(&self) -> Result<extent::LogicalExtent, Error> {
        match &self.extent {
            Some(extent) => Ok(extent.clone()),
            None => StageRead::logical_extent(&self.input),
        }
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut Bindings,
    ) -> Result<(), Error> {
        StageRead::stage_at(&self.input, client, owner, bindings)
    }
}
