use std::sync::Arc;

use raug::prelude::*;

use crate::{
    processor::FftProcessor,
    signal::{Complex32, Fft},
};

pub struct RealFft<F: Fft> {
    plan: Arc<dyn realfft::RealToComplex<f32>>,
    scratch: Vec<Complex32>,
    rfft_input: Vec<f32>,
    rfft_output: Vec<Complex32>,
    out_signal: Box<F::RealFft>,
}

impl<F: Fft> RealFft<F> {
    pub fn new() -> Self {
        let mut planner = realfft::RealFftPlanner::new();
        let plan = planner.plan_fft_forward(F::N_FFT);
        let scratch = plan.make_scratch_vec();
        let rfft_input = plan.make_input_vec();
        let rfft_output = plan.make_output_vec();
        Self {
            plan,
            scratch,
            rfft_input,
            rfft_output,
            out_signal: Box::new(F::RealFft::default()),
        }
    }
}

impl<F: Fft> Default for RealFft<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Fft> FftProcessor for RealFft<F> {
    fn input_spec(&self) -> Vec<SignalSpec> {
        vec![SignalSpec::new("input", F::AudioBlock::signal_type())]
    }

    fn output_spec(&self) -> Vec<SignalSpec> {
        vec![SignalSpec::new("output", F::RealFft::signal_type())]
    }

    fn create_output_buffers(&self, size: usize) -> Vec<AnyBuffer> {
        vec![AnyBuffer::zeros::<F::RealFft>(size)]
    }

    fn process(
        &mut self,
        inputs: ProcessorInputs,
        mut outputs: ProcessorOutputs,
    ) -> ProcResult<()> {
        let input = inputs.input_as::<F::AudioBlock>(0).unwrap();

        for (i, input) in input.iter().enumerate() {
            self.rfft_input.copy_from_slice(input);

            let res = self.plan.process_with_scratch(
                &mut self.rfft_input,
                &mut self.rfft_output,
                &mut self.scratch,
            );

            self.scratch.fill(Complex32::ZERO);

            if let Err(e) = res {
                return Err(ProcessorError::ProcessingError(Box::new(e)));
            }

            self.out_signal.copy_from_slice(&self.rfft_output);

            outputs.set_output_as::<F::RealFft>(0, i, &*self.out_signal)?;
        }

        Ok(())
    }
}

pub struct InverseRealFft<F: Fft> {
    plan: Arc<dyn realfft::ComplexToReal<f32>>,
    scratch: Vec<Complex32>,
    irfft_input: Vec<Complex32>,
    irfft_output: Vec<f32>,
    out_signal: Box<F::AudioBlock>,
}

impl<F: Fft> InverseRealFft<F> {
    pub fn new() -> Self {
        let mut planner = realfft::RealFftPlanner::new();
        let plan = planner.plan_fft_inverse(F::N_FFT);
        let scratch = plan.make_scratch_vec();
        let irfft_input = plan.make_input_vec();
        let irfft_output = plan.make_output_vec();
        Self {
            plan,
            scratch,
            irfft_input,
            irfft_output,
            out_signal: Box::new(F::AudioBlock::default()),
        }
    }
}

impl<F: Fft> Default for InverseRealFft<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Fft> FftProcessor for InverseRealFft<F> {
    fn input_spec(&self) -> Vec<SignalSpec> {
        vec![SignalSpec::new("input", F::RealFft::signal_type())]
    }

    fn output_spec(&self) -> Vec<SignalSpec> {
        vec![SignalSpec::new("output", F::AudioBlock::signal_type())]
    }

    fn create_output_buffers(&self, size: usize) -> Vec<AnyBuffer> {
        vec![AnyBuffer::zeros::<F::AudioBlock>(size)]
    }

    fn process(
        &mut self,
        inputs: ProcessorInputs,
        mut outputs: ProcessorOutputs,
    ) -> ProcResult<()> {
        let input = inputs.input_as::<F::RealFft>(0).unwrap();

        for (i, input) in input.iter().enumerate() {
            self.irfft_input.copy_from_slice(input);

            self.irfft_input[0].im = 0.0;
            self.irfft_input[F::N_REAL_BINS - 1].im = 0.0;

            let res = self.plan.process_with_scratch(
                &mut self.irfft_input,
                &mut self.irfft_output,
                &mut self.scratch,
            );

            self.scratch.fill(Complex32::ZERO);

            if let Err(e) = res {
                return Err(ProcessorError::ProcessingError(Box::new(e)));
            }

            self.out_signal.copy_from_slice(&self.irfft_output);

            outputs.set_output_as(0, i, &*self.out_signal)?;
        }

        Ok(())
    }
}
