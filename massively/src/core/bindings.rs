//! Host-side buffers and offsets for the fixed kernel ABI.
//!
//! Read and output trees use the same binding representation. Staging validates
//! executor ownership before padding unused slots with one shared dummy buffer.
//! Kernel-specific metadata stays with the algorithm that interprets it.

use cubecl::prelude::*;

use crate::{Error, Executor};

use super::{
    output::StageOutput,
    read::{Env0, StageRead},
};

const READ_SLOTS: usize = 13;
const WRITE_SLOTS: usize = 12;

pub(crate) const TRANSFER_READ_OFFSET: usize = WRITE_SLOTS;
pub(crate) const TRANSFER_PARAMETERS_OFFSET: usize = WRITE_SLOTS + READ_SLOTS;

#[doc(hidden)]
pub struct Bindings {
    pub(crate) slots: Vec<(cubecl::server::Handle, usize)>,
    pub(crate) offsets: Vec<u32>,
}

impl Bindings {
    pub(crate) fn new() -> Self {
        Self {
            slots: Vec::with_capacity(READ_SLOTS),
            offsets: Vec::with_capacity(READ_SLOTS),
        }
    }

    pub(crate) fn read<R, Input>(exec: &Executor<R>, input: &Input) -> Result<Self, Error>
    where
        R: Runtime,
        Input: StageRead<R, Env0>,
    {
        let mut bindings = Self::new();
        input.stage_at(exec.client(), exec.id(), &mut bindings)?;
        bindings.pad_to(exec.client(), READ_SLOTS);
        Ok(bindings)
    }

    pub(crate) fn write<R, Output>(exec: &Executor<R>, output: &Output) -> Result<Self, Error>
    where
        R: Runtime,
        Output: StageOutput<R, Env0>,
    {
        let mut bindings = Self::new();
        output.stage_output(exec.id(), &mut bindings)?;
        bindings.pad_to(exec.client(), WRITE_SLOTS);
        Ok(bindings)
    }

    pub(crate) fn push(&mut self, handle: cubecl::server::Handle, len: usize, offset: u32) {
        self.slots.push((handle, len));
        self.offsets.push(offset);
    }

    /// One metadata binding for a row transfer: output offsets first for
    /// StorePadded12, then input offsets for Eval13At, then operation parameters.
    pub(crate) fn transfer_metadata(reads: &Self, writes: &Self, parameters: &[u32]) -> Vec<u32> {
        let mut metadata = Vec::with_capacity(TRANSFER_PARAMETERS_OFFSET + parameters.len());
        metadata.extend_from_slice(&writes.offsets);
        metadata.extend_from_slice(&reads.offsets);
        metadata.extend_from_slice(parameters);
        metadata
    }

    /// Unused slots are never indexed by the corresponding device expression.
    fn pad_to<R: Runtime>(&mut self, client: &ComputeClient<R>, count: usize) {
        assert!(
            self.slots.len() <= count,
            "kernel binding capacity exceeded"
        );
        if self.slots.len() == count {
            return;
        }
        let dummy = client.empty(core::mem::size_of::<u32>());
        while self.slots.len() < count {
            self.push(dummy.clone(), 1, 0);
        }
    }
}
