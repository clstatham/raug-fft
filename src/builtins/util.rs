use raug::prelude::*;

use crate::{prelude::FftProcessor, signal::Fft};

pub struct Null<F: Fft> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Fft> Null<F> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F: Fft> Default for Null<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Fft> FftProcessor for Null<F> {
    fn name(&self) -> &str {
        "Null"
    }

    fn input_spec(&self) -> Vec<SignalSpec> {
        vec![]
    }

    fn output_spec(&self) -> Vec<SignalSpec> {
        vec![SignalSpec::new("output", F::AudioBlock::signal_type())]
    }

    fn create_output_buffers(&self, size: usize) -> Vec<AnyBuffer> {
        vec![AnyBuffer::zeros::<F::AudioBlock>(size)]
    }

    fn process(&mut self, _inputs: ProcessorInputs, _outputs: ProcessorOutputs) -> ProcResult<()> {
        Ok(())
    }
}
