use std::{collections::VecDeque, fmt::Debug, marker::PhantomData};

use raug::{graph::node::ProcessNodeError, prelude::*};
use raug_graph::prelude::*;

use crate::{processor::FftProcessor, signal::Fft};

pub struct FftProcessorNode {
    pub(crate) processor: Box<dyn FftProcessor>,
    pub(crate) input_spec: Vec<SignalSpec>,
    pub(crate) output_spec: Vec<SignalSpec>,
    pub(crate) outputs: Vec<AnyBuffer>,
}

impl Debug for FftProcessorNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.processor.name())
    }
}

impl FftProcessorNode {
    pub fn new(processor: impl FftProcessor) -> Self {
        Self::new_from_boxed(Box::new(processor))
    }

    pub fn new_from_boxed(processor: Box<dyn FftProcessor>) -> Self {
        let input_spec = processor.input_spec();
        let output_spec = processor.output_spec();
        let outputs = processor.create_output_buffers(0);
        Self {
            processor,
            input_spec,
            output_spec,
            outputs,
        }
    }

    #[inline]
    pub fn name(&self) -> &str {
        self.processor.name()
    }

    #[inline]
    pub fn input_spec(&self) -> &[SignalSpec] {
        &self.input_spec
    }

    #[inline]
    pub fn output_spec(&self) -> &[SignalSpec] {
        &self.output_spec
    }

    /// Returns the number of input signals of the processor.
    #[inline]
    pub fn num_inputs(&self) -> usize {
        self.input_spec.len()
    }

    /// Returns the number of output signals of the processor.
    #[inline]
    pub fn num_outputs(&self) -> usize {
        self.output_spec.len()
    }

    /// Returns a reference to the processor.
    #[inline]
    pub fn processor(&self) -> &dyn FftProcessor {
        &*self.processor
    }

    /// Returns a mutable reference to the processor.
    #[inline]
    pub fn processor_mut(&mut self) -> &mut dyn FftProcessor {
        &mut *self.processor
    }

    /// Allocates memory for the processor.
    #[inline]
    pub fn allocate(&mut self, sample_rate: f32) {
        self.processor.allocate(sample_rate);
        self.outputs = self.processor.create_output_buffers(1);
    }

    /// Resizes the internal buffers of the processor and updates the sample rate and FFT size.
    ///
    /// This function is NOT ALLOWED to allocate memory.
    #[inline]
    pub fn resize_buffers(&mut self, sample_rate: f32) {
        self.processor.resize_buffers(sample_rate);
    }

    /// Processes the input signals and writes the output signals to the given buffers.
    #[inline]
    pub(crate) fn process(
        &mut self,
        inputs: &[Option<*const AnyBuffer>],
        env: ProcEnv,
    ) -> Result<(), ProcessNodeError> {
        let inputs = ProcessorInputs {
            input_specs: &self.input_spec,
            inputs,
            env,
        };
        let outputs = ProcessorOutputs {
            output_spec: &self.output_spec,
            outputs: &mut self.outputs,
            mode: env.mode,
        };
        if let Err(e) = self.processor.process(inputs, outputs) {
            return Err(ProcessNodeError {
                error: e,
                node_name: self.name().to_string(),
            });
        }

        Ok(())
    }
}

impl AbstractNode for FftProcessorNode {
    fn name(&self) -> Option<String> {
        Some(self.processor.name().to_string())
    }

    fn num_inputs(&self) -> usize {
        self.input_spec.len()
    }

    fn num_outputs(&self) -> usize {
        self.output_spec.len()
    }

    fn input_type(&self, index: u32) -> Option<raug_graph::TypeInfo> {
        self.input_spec
            .get(index as usize)
            .map(|v| v.signal_type.into())
    }

    fn output_type(&self, index: u32) -> Option<raug_graph::TypeInfo> {
        self.output_spec
            .get(index as usize)
            .map(|v| v.signal_type.into())
    }

    fn input_name(&self, index: u32) -> Option<&str> {
        self.input_spec.get(index as usize).map(|v| v.name.as_str())
    }

    fn output_name(&self, index: u32) -> Option<&str> {
        self.output_spec
            .get(index as usize)
            .map(|v| v.name.as_str())
    }
}

pub struct FftInput<F: Fft> {
    pub(crate) ring_buffer: VecDeque<f32>,
    pub(crate) time_domain: F::AudioBlock,
}

impl<F: Fft> Default for FftInput<F> {
    fn default() -> Self {
        Self {
            ring_buffer: VecDeque::new(),
            time_domain: F::AudioBlock::default(),
        }
    }
}

pub struct FftOutput<F: Fft> {
    pub(crate) ring_buffer: VecDeque<f32>,
    pub(crate) overlap_buffer: VecDeque<f32>,
    _f: PhantomData<F>,
}

impl<F: Fft> Default for FftOutput<F> {
    fn default() -> Self {
        Self {
            ring_buffer: VecDeque::with_capacity(F::N_FFT),
            overlap_buffer: vec![0.0; F::N_FFT].into(),
            _f: PhantomData,
        }
    }
}
