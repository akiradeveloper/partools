//! Scatter primitives with logical index expressions.

use cubecl::prelude::Runtime;

use crate::core::indexed::IndexedCopyInput;
use crate::core::read::{Env0, ReadExpression};
use crate::{Error, Executor};

/// Writes each input item to the output position given by its index.
pub(crate) fn scatter<R, Values, Indices, Output>(
    exec: &Executor<R>,
    values: Values,
    indices: Indices,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: IndexedCopyInput<R, Indices, Output>,
{
    values.indexed_copy(exec, indices, false, output)
}

/// Scatters rows whose logical stencil is true.
pub(crate) fn scatter_where<R, Values, Indices, Stencil, Output>(
    exec: &Executor<R>,
    values: Values,
    indices: Indices,
    stencil: Stencil,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: IndexedCopyInput<R, Indices, Output> + crate::core::read::StageRead<R, Env0>,
    Indices: ReadExpression<Item = crate::MIndex> + crate::core::read::StageRead<R, Env0>,
    Stencil: crate::core::selection::FlagInput<R>,
    Output: crate::core::output::OutputExpression,
{
    let values_len = values.physical_len()?;
    let indices_len = indices.physical_len()?;
    let stencil_len = stencil.flag_len()?;
    if values_len != indices_len {
        return Err(Error::LengthMismatch {
            left: values_len,
            right: indices_len,
        });
    }
    if values_len != stencil_len {
        return Err(Error::LengthMismatch {
            left: values_len,
            right: stencil_len,
        });
    }
    values
        .logical_extent()?
        .zipped(&indices.logical_extent()?)?
        .zipped(&stencil.flag_extent()?)?;
    let control = stencil.selected_control(exec)?;
    values.indexed_copy_selected(
        exec,
        indices,
        Some(crate::core::indexed::IndexSelection::Routing {
            control: &control,
            retain_output_position: false,
        }),
        Some(control.count()),
        false,
        output,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::iter::Zip;
    use crate::core::read::{Counting, Permute};
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn scatter_accepts_multicolumn_values_and_logical_indices() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let left = exec.to_device(&[1_u32, 2, 3]);
        let right = exec.to_device(&[10_u32, 20, 30]);
        let encoded = exec.to_device(&[2_u32, 0, 3]);
        let indices = encoded.column();
        let out_left = exec.to_device(&[0_u32; 4]);
        let out_right = exec.to_device(&[0_u32; 4]);

        scatter(
            &exec,
            Zip::new(left.column(), right.column()),
            indices,
            Zip::new(out_left.slice_mut(..), out_right.slice_mut(..)),
        )
        .unwrap();

        assert_eq!(exec.to_host(&out_left).unwrap(), vec![2, 0, 1, 3]);
        assert_eq!(exec.to_host(&out_right).unwrap(), vec![20, 0, 10, 30]);
    }

    #[test]
    fn scatter_keeps_value_and_index_expressions_independent() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let values = exec.to_device(&[10_u32, 20, 30]);
        let indices = exec.to_device(&[2_u32, 0, 1]);
        let output = exec.to_device(&[99_u32; 3]);

        scatter(
            &exec,
            Permute::new(values.column(), Counting::new(0, 3)),
            Permute::new(indices.column(), Counting::new(0, 3)),
            output.slice_mut(..),
        )
        .unwrap();

        assert_eq!(exec.to_host(&output).unwrap(), vec![20, 30, 10]);
    }

    #[test]
    fn scatter_where_does_not_evaluate_rejected_indices() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let values = exec.to_device(&[10_u32, 20, 30]);
        let encoded_indices = exec.to_device(&[2_u32, u32::MAX, 1]);
        let encoded_stencil = exec.to_device(&[1_u32, 0, 1]);
        let output = exec.to_device(&[0_u32; 3]);

        scatter_where(
            &exec,
            values.column(),
            encoded_indices.column(),
            encoded_stencil.column(),
            output.slice_mut(..),
        )
        .unwrap();

        assert_eq!(exec.to_host(&output).unwrap(), vec![0, 30, 10]);
    }
}
