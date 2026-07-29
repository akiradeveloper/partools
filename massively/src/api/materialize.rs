//! Explicit materialization boundaries for owned storage.
//!
//! Ordinary vector algorithms propagate device extents. Consumers that need a
//! host-known size, such as validated segmentations, use `exact` to copy only
//! the initialized prefix after resolving its logical extent.

use cubecl::prelude::Runtime;

use super::{
    algorithm::transform::copy,
    iter::{MStorageExtent, checked_len},
};
use crate::core::extent::LogicalExtent;
use crate::core::read::SliceExpression;
use crate::{Error, Executor, MAlloc, MIndex, MIter, MStorage, MVec};

pub(crate) fn exact<R, Input>(
    exec: &Executor<R>,
    input: Input,
) -> Result<MVec<R, Input::Item>, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Input::Item: MAlloc<R>,
{
    let len = checked_len(input.logical_extent()?.read(exec)?)?;
    let output = exec.alloc::<Input::Item>(len);
    let input = input.lower_read().slice_expression(0, len as usize);
    copy(exec, input, output.slice_mut(..))?;
    Ok(output)
}

/// Trims an allocation to an already known initialized prefix across all columns.
pub(crate) fn into_exact_prefix<R, Item>(
    exec: &Executor<R>,
    mut storage: MVec<R, Item>,
    len: MIndex,
) -> Result<MVec<R, Item>, Error>
where
    R: Runtime,
    Item: MAlloc<R>,
{
    let capacity = storage.capacity()?;
    if len > capacity {
        return Err(Error::OutputTooShort {
            input: len as usize,
            output: capacity as usize,
        });
    }
    if len == capacity {
        storage.set_logical_extent(LogicalExtent::fixed(len as usize));
        return Ok(storage);
    }

    let output = exec.alloc::<Item>(len);
    copy(exec, storage.slice(..len), output.slice_mut(..))?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{vector::copy_where, zip2};
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn exact_copies_the_logical_slice_of_every_column() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let left = exec.to_device(&[10_u32, 20, 30, 40, 50]);
        let right = exec.to_device(&[100_u32, 200, 300, 400, 500]);
        let flags = exec.to_device(&[1_u32, 0, 1, 0, 1]);
        let selected = copy_where(
            &exec,
            zip2(left.slice(..), right.slice(..)),
            flags.slice(..),
        )
        .unwrap();
        let output = exact(&exec, selected.slice(1..5)).unwrap();

        assert_eq!(output.capacity().unwrap(), 2);
        let (left, right) = output.into_columns();
        assert_eq!(exec.to_host(&left).unwrap(), vec![30, 50]);
        assert_eq!(exec.to_host(&right).unwrap(), vec![300, 500]);
    }
}
