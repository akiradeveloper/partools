use cubecl::prelude::{CubeType, Runtime};

use crate::api::iter::MStorageExtent;
use crate::{Error, Executor, MAlloc, MFlag, MIter, MIterMut, MStorage, MVec};

struct GatherOperation<'a, R: Runtime, Values, Indices> {
    exec: &'a Executor<R>,
    values: Values,
    indices: Indices,
}

impl<R, Item, Values, Indices> crate::api::iter::OutputOperation<R, Item>
    for GatherOperation<'_, R, Values, Indices>
where
    R: Runtime,
    Item: CubeType + Send + Sync + 'static,
    Values: MIter<R, Item = Item>,
    Indices: MIter<R, Item = crate::MIndex>,
{
    type Result = Result<(), Error>;

    fn run<Output>(self, output: Output) -> Self::Result
    where
        Item: crate::api::iter::KernelRow + crate::core::allocation::ScratchStorage<R>,
        Output: crate::api::iter::ConcreteOutput<R, Item>,
    {
        crate::core::indexed::gather_direct(
            self.exec,
            crate::api::iter::lower::<R, _>(self.values),
            crate::api::iter::lower::<R, _>(self.indices),
            output,
        )
    }
}

struct GatherWhereOperation<'a, R: Runtime, Values, Indices, Stencil> {
    exec: &'a Executor<R>,
    values: Values,
    indices: Indices,
    stencil: Stencil,
}

struct ApplyPermutationOperation<'a, R: Runtime, Input> {
    exec: &'a Executor<R>,
    input: Input,
    indices: crate::core::read::Column<u32>,
    active_len: Option<&'a crate::DeviceVec<R, u32>>,
}

impl<R, Item, Input> crate::api::iter::OutputOperation<R, Item>
    for ApplyPermutationOperation<'_, R, Input>
where
    R: Runtime,
    Item: CubeType + Send + Sync + 'static,
    Input: crate::core::facade::KernelInput<R, Item = Item>,
{
    type Result = Result<(), Error>;

    fn run<Output>(self, output: Output) -> Self::Result
    where
        Item: crate::api::iter::KernelRow + crate::core::allocation::ScratchStorage<R>,
        Output: crate::api::iter::ConcreteOutput<R, Item>,
    {
        crate::core::indexed::PermutationCopyInput::permutation_copy(
            self.input,
            self.exec,
            self.indices,
            self.active_len,
            output,
        )
    }
}

pub(crate) fn apply_permutation_into<R, Input, Output>(
    exec: &Executor<R>,
    input: Input,
    indices: crate::core::read::Column<u32>,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: crate::core::facade::KernelInput<R, Item = Output::Item>,
    Output: MIterMut<R>,
{
    output.run_output_operation(ApplyPermutationOperation {
        exec,
        input,
        indices,
        active_len: None,
    })
}

pub(crate) fn apply_permutation_prefix_into<R, Input, Output>(
    exec: &Executor<R>,
    input: Input,
    indices: crate::core::read::Column<u32>,
    active_len: &crate::DeviceVec<R, u32>,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: crate::core::facade::KernelInput<R, Item = Output::Item>,
    Output: MIterMut<R>,
{
    output.run_output_operation(ApplyPermutationOperation {
        exec,
        input,
        indices,
        active_len: Some(active_len),
    })
}

impl<R, Item, Values, Indices, Stencil> crate::api::iter::OutputOperation<R, Item>
    for GatherWhereOperation<'_, R, Values, Indices, Stencil>
where
    R: Runtime,
    Item: CubeType + Send + Sync + 'static,
    Values: MIter<R, Item = Item>,
    Indices: MIter<R, Item = crate::MIndex>,
    Stencil: MIter<R, Item = MFlag>,
{
    type Result = Result<(), Error>;

    fn run<Output>(self, output: Output) -> Self::Result
    where
        Item: crate::api::iter::KernelRow + crate::core::allocation::ScratchStorage<R>,
        Output: crate::api::iter::ConcreteOutput<R, Item>,
    {
        let control = crate::core::selection::FlagInput::selected_control(
            crate::api::iter::lower::<R, _>(self.stencil),
            self.exec,
        )?;
        crate::core::indexed::IndexedCopyInput::indexed_copy_selected(
            crate::api::iter::lower::<R, _>(self.values),
            self.exec,
            crate::api::iter::lower::<R, _>(self.indices),
            Some(crate::core::indexed::IndexSelection::Routing {
                control: &control,
                retain_output_position: true,
            }),
            Some(control.count()),
            true,
            output,
        )
    }
}

/// Gathers `values[indices[i]]` into owned device storage.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, vector::gather};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let values = exec.to_device(&[10_u32, 20, 30]);
/// let indices = exec.to_device(&[2_u32, 0, 1]);
/// let output = gather(&exec, values.slice(..), indices.slice(..)).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![30, 10, 20]);
/// ```
pub fn gather<R, Values, Item, Indices>(
    exec: &Executor<R>,
    values: Values,
    indices: Indices,
) -> Result<MVec<R, Item>, Error>
where
    R: Runtime,
    Values: MIter<R, Item = Item>,
    Item: MAlloc<R>,
    Indices: MIter<R, Item = crate::MIndex>,
{
    let len = indices.capacity()?;
    let extent = indices.logical_extent()?;
    let mut output = exec.alloc::<Item>(len);
    gather_into(exec, values, indices, output.slice_mut(..))?;
    output.set_logical_extent(extent);
    Ok(output)
}

/// Gathers values into caller-provided storage.
#[doc(hidden)]
pub(crate) fn gather_into<R, Values, Indices, Output>(
    exec: &Executor<R>,
    values: Values,
    indices: Indices,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: MIter<R, Item = Output::Item>,
    Indices: MIter<R, Item = crate::MIndex>,
    Output: MIterMut<R>,
{
    output.run_output_operation(GatherOperation {
        exec,
        values,
        indices,
    })
}

/// Gathers selected rows while preserving other output rows.
///
/// A false stencil leaves the corresponding output position unchanged.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, vector::gather_where};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let values = exec.to_device(&[10_u32, 20, 30]);
/// let indices = exec.to_device(&[2_u32, 0, 1]);
/// let stencil = exec.to_device(&[1_u32, 0, 1]);
/// let output = exec.to_device(&[99_u32, 99, 99]);
///
/// gather_where(
///     &exec,
///     values.slice(..),
///     indices.slice(..),
///     stencil.slice(..),
///     output.slice_mut(..),
/// )
/// .unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![30, 99, 20]);
/// ```
pub fn gather_where<R, Values, Indices, Stencil, Output>(
    exec: &Executor<R>,
    values: Values,
    indices: Indices,
    stencil: Stencil,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: MIter<R, Item = Output::Item>,
    Indices: MIter<R, Item = crate::MIndex>,
    Stencil: MIter<R, Item = MFlag>,
    Output: MIterMut<R>,
{
    let indices_len = indices.capacity()?;
    let stencil_len = stencil.capacity()?;
    if indices_len != stencil_len {
        return Err(Error::LengthMismatch {
            left: indices_len as usize,
            right: stencil_len as usize,
        });
    }
    let output_len = output.capacity()?;
    if output_len < indices_len {
        return Err(Error::OutputTooShort {
            input: indices_len as usize,
            output: output_len as usize,
        });
    }

    output.run_output_operation(GatherWhereOperation {
        exec,
        values,
        indices,
        stencil,
    })
}

/// Reverses values into owned device storage.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, vector::reverse};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[1_u32, 2, 3, 4]);
/// let output = reverse(&exec, input.slice(..)).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![4, 3, 2, 1]);
/// ```
pub fn reverse<R, Values, Item>(exec: &Executor<R>, values: Values) -> Result<MVec<R, Item>, Error>
where
    R: Runtime,
    Values: MIter<R, Item = Item>,
    crate::lazy::Reverse<Values>: MIter<R, Item = Item>,
    Item: MAlloc<R>,
{
    let len = values.capacity()?;
    let extent = values.logical_extent()?;
    let mut output = exec.alloc::<Item>(len);
    reverse_into(exec, values, output.slice_mut(..))?;
    output.set_logical_extent(extent);
    Ok(output)
}

/// Reverses values into caller-provided storage.
#[doc(hidden)]
pub(crate) fn reverse_into<R, Values, Output>(
    exec: &Executor<R>,
    values: Values,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: MIter<R, Item = Output::Item>,
    crate::lazy::Reverse<Values>: MIter<R, Item = Output::Item>,
    Output: MIterMut<R>,
{
    crate::api::algorithm::transform::transform_into(
        exec,
        crate::lazy::Reverse::new(values),
        crate::op::Identity,
        output,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn gather_where_does_not_evaluate_rejected_indices() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let values = exec.to_device(&[10_u32, 20, 30]);
        let encoded_indices = exec.to_device(&[2_u32, u32::MAX, 1]);
        let encoded_stencil = exec.to_device(&[1_u32, 0, 1]);
        let output = exec.to_device(&[99_u32; 3]);

        gather_where(
            &exec,
            values.slice(..),
            encoded_indices.slice(..),
            encoded_stencil.slice(..),
            output.slice_mut(..),
        )
        .unwrap();

        assert_eq!(exec.to_host(&output).unwrap(), vec![30, 99, 20]);
    }
}
