use std::{collections::BTreeMap, ops::Deref};

use raug::{graph::GraphRunResult, prelude::*, processor::io::ProcessMode};

use crate::{
    WindowFunction,
    builtins::transforms::{InverseRealFft, RealFft},
    node::{FftInput, FftOutput, FftProcessorNode},
    prelude::util::Null,
    processor::FftProcessor,
    signal::Fft,
};

use raug_graph::{
    graph::{AbstractGraph, DuplicateConnectionMode, Graph, NodeIndex, VisitResult},
    petgraph::{Direction, visit::EdgeRef},
    prelude::{GraphBuilder, NodeBuilder},
};

pub struct FftGraph<F: Fft> {
    graph: Graph<Self>,

    sample_rate: f32,
    block_size: usize,
    hop_length: usize,
    window: Vec<f32>,

    inputs: BTreeMap<NodeIndex, FftInput<F>>,
    outputs: BTreeMap<NodeIndex, FftOutput<F>>,
}

impl<F: Fft> AbstractGraph for FftGraph<F> {
    type Node = FftProcessorNode;
    type Edge = ();

    fn duplicate_connection_mode() -> DuplicateConnectionMode {
        DuplicateConnectionMode::Disconnect
    }

    fn graph(&self) -> &Graph<Self> {
        &self.graph
    }

    fn graph_mut(&mut self) -> &mut Graph<Self> {
        &mut self.graph
    }
}

impl<F: Fft> FftGraph<F> {
    pub fn new(hop_length: usize, window_fn: WindowFunction) -> Self {
        let mut window = window_fn.generate(F::N_FFT);

        // center the window around 0
        window.rotate_right(F::N_FFT / 2);

        let overlapping_frames = F::N_FFT / hop_length;
        let mut window_sum: f32 = window.iter().map(|x| x * x).sum();
        window_sum *= overlapping_frames as f32;
        assert_ne!(window_sum, 0.0);

        for x in window.iter_mut() {
            *x /= window_sum.sqrt();
        }

        Self {
            graph: Graph::new(),
            sample_rate: 0.0,
            block_size: 0,
            hop_length,
            window,
            inputs: BTreeMap::new(),
            outputs: BTreeMap::new(),
        }
    }

    pub fn fft_length(&self) -> usize {
        F::N_FFT
    }

    pub fn hop_length(&self) -> usize {
        self.hop_length
    }

    pub fn add_audio_input(&mut self) -> NodeIndex {
        let null = self.add_processor(Null::<F>::new());
        let fft = self.add_processor(RealFft::<F>::new());
        self.graph.connect(null, 0, fft, 0).unwrap();
        self.inputs.insert(null, FftInput::<F>::default());
        fft
    }

    pub fn add_audio_output(&mut self) -> NodeIndex {
        let idx = self.add_processor(InverseRealFft::<F>::new());
        self.outputs.insert(idx, FftOutput::<F>::default());
        idx
    }

    pub fn add_processor(&mut self, processor: impl FftProcessor) -> NodeIndex {
        let mut node = FftProcessorNode::new(processor);
        node.allocate(self.sample_rate);
        node.resize_buffers(self.sample_rate);

        self.graph.add_node(node)
    }

    pub fn allocate(&mut self, sample_rate: f32, block_size: usize) {
        self.sample_rate = sample_rate;
        self.block_size = block_size;

        self.graph.visit_mut(|_i, node| {
            node.allocate(sample_rate);
            VisitResult::Continue::<()>
        });
    }

    pub fn resize_buffers(&mut self, sample_rate: f32, block_size: usize) {
        self.sample_rate = sample_rate;
        self.block_size = block_size;

        self.graph.visit_mut(|_i, node| {
            node.resize_buffers(sample_rate);
            VisitResult::Continue::<()>
        });
    }

    #[allow(clippy::needless_range_loop)]
    fn process_inner(
        &mut self,
        inputs: ProcessorInputs,
        mut outputs: ProcessorOutputs,
    ) -> ProcResult<()> {
        self.graph.reset_visitor();

        // if there are no inputs, we can't process anything
        if self.inputs.is_empty() {
            return Ok(());
        }

        let fft_length = self.fft_length();
        let hop_length = self.hop_length();

        let mut input_buffer_length = usize::MAX;

        // fill our input buffers with the input signals
        for (input_index, fft_input) in self.inputs.values_mut().enumerate() {
            let audio_input = inputs.input_as::<f32>(input_index).unwrap();

            fft_input
                .ring_buffer
                .extend(&audio_input[..self.block_size]);

            input_buffer_length = input_buffer_length.min(fft_input.ring_buffer.len());
        }

        // while we still have enough samples to process...
        while input_buffer_length >= fft_length {
            for (&node_index, fft_input) in self.inputs.iter_mut() {
                // window the input
                for i in 0..fft_length {
                    fft_input.time_domain[i] = fft_input.ring_buffer[i] * self.window[i];
                }

                // copy the time domain signal to the FFT input
                self.graph[node_index].outputs[0]
                    .get_mut_as::<F::AudioBlock>(0)
                    .unwrap()
                    .copy_from_slice(&fft_input.time_domain);

                // advance time for the input
                fft_input.ring_buffer.drain(..hop_length);
            }

            // update the input buffer length
            input_buffer_length -= hop_length;

            // traverse the graph and process each node
            for i in 0..self.graph.visit_path().len() {
                let node_id = self.graph.visit_path()[i];
                if let Err(e) = self.process_node(node_id) {
                    return Err(ProcessorError::SubGraphError(Box::new(e)));
                }
            }

            // copy the FFT output to the output buffers
            for (&output_node_idx, fft_output) in self.outputs.iter_mut() {
                let output_buf = &self.graph[output_node_idx].outputs[0]
                    .as_slice::<F::AudioBlock>()
                    .unwrap()[0];

                // overlap-add
                for i in 0..fft_length {
                    fft_output.overlap_buffer[i] += output_buf[i] * self.window[i];
                }

                // advance time for the output
                fft_output
                    .ring_buffer
                    .extend(fft_output.overlap_buffer.drain(..hop_length));

                for _ in 0..hop_length {
                    // zero out the overlap buffer for the next iteration
                    fft_output.overlap_buffer.push_back(0.0);
                }
            }
        }

        // for each output, write as many samples as we can to the block's corresponding output
        for (output_index, fft_output) in self.outputs.values_mut().enumerate() {
            if fft_output.ring_buffer.len() < inputs.block_size() {
                log::debug!(
                    "FftGraph underrun at output index {output_index}, not enough samples in ring buffer"
                );
                continue;
            }
            for sample_index in 0..inputs.block_size() {
                let sample = fft_output.ring_buffer.pop_front().unwrap_or(
                    // shouldn't happen due to the check above, but just in case
                    0.0,
                );
                outputs.set_output_as::<f32>(output_index, sample_index, &sample)?;
            }
        }

        Ok(())
    }

    fn process_node(&mut self, node_id: NodeIndex) -> GraphRunResult<()> {
        let mut inputs: [Option<*const AnyBuffer>; 32] = [None; 32];

        for (source_id, edge) in self
            .graph
            .digraph()
            .edges_directed(node_id, Direction::Incoming)
            .map(|edge| (edge.source(), edge.weight()))
        {
            let source_buffers = &self.graph[source_id].outputs;
            let buffer = &source_buffers[edge.source_output as usize] as *const AnyBuffer;

            inputs[edge.target_input as usize] = Some(buffer);
        }

        let node = &mut self.graph[node_id];

        node.process(
            &inputs[..],
            ProcEnv {
                sample_rate: self.sample_rate,
                block_size: self.block_size,
                mode: ProcessMode::Block,
            },
        )?;

        Ok(())
    }
}

impl<F: Fft> Processor for FftGraph<F> {
    fn input_spec(&self) -> Vec<SignalSpec> {
        let mut specs = Vec::with_capacity(self.inputs.len());
        for i in 0..self.inputs.len() {
            specs.push(SignalSpec::new(i.to_string(), f32::signal_type()));
        }
        specs
    }

    fn output_spec(&self) -> Vec<SignalSpec> {
        let mut specs = Vec::with_capacity(self.outputs.len());
        for i in 0..self.outputs.len() {
            specs.push(SignalSpec::new(i.to_string(), f32::signal_type()));
        }
        specs
    }

    fn create_output_buffers(&self, size: usize) -> Vec<AnyBuffer> {
        let mut buffers = Vec::with_capacity(self.outputs.len());
        for _ in 0..self.outputs.len() {
            buffers.push(AnyBuffer::zeros::<f32>(size));
        }
        buffers
    }

    fn allocate(&mut self, sample_rate: f32, max_block_size: usize) {
        FftGraph::allocate(self, sample_rate, max_block_size);
    }

    fn resize_buffers(&mut self, sample_rate: f32, block_size: usize) {
        FftGraph::resize_buffers(self, sample_rate, block_size);
    }

    fn process(
        &mut self,
        inputs: ProcessorInputs,
        outputs: ProcessorOutputs,
    ) -> Result<(), ProcessorError> {
        self.process_inner(inputs, outputs)
    }
}

pub struct FftGraphBuilder<F: Fft>(GraphBuilder<FftGraph<F>>);

impl<F: Fft> Deref for FftGraphBuilder<F> {
    type Target = GraphBuilder<FftGraph<F>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F: Fft> FftGraphBuilder<F> {
    pub fn new(hop_length: usize, window_fn: WindowFunction) -> Self {
        Self(GraphBuilder::from_inner(FftGraph::new(
            hop_length, window_fn,
        )))
    }

    pub fn add_audio_input(&self) -> NodeBuilder<FftGraph<F>> {
        let node_id = self.with_inner(|graph| graph.add_audio_input());
        NodeBuilder::new(self.0.clone(), node_id)
    }

    pub fn add_audio_output(&self) -> NodeBuilder<FftGraph<F>> {
        let node_id = self.with_inner(|graph| graph.add_audio_output());
        NodeBuilder::new(self.0.clone(), node_id)
    }
}

impl<F: Fft> Processor for FftGraphBuilder<F> {
    fn input_spec(&self) -> Vec<SignalSpec> {
        self.with_inner(|graph| graph.input_spec())
    }

    fn output_spec(&self) -> Vec<SignalSpec> {
        self.with_inner(|graph| graph.output_spec())
    }

    fn create_output_buffers(&self, size: usize) -> Vec<AnyBuffer> {
        self.with_inner(|graph| graph.create_output_buffers(size))
    }

    fn allocate(&mut self, sample_rate: f32, max_block_size: usize) {
        self.with_inner(|graph| graph.allocate(sample_rate, max_block_size));
    }

    fn resize_buffers(&mut self, sample_rate: f32, block_size: usize) {
        self.with_inner(|graph| graph.resize_buffers(sample_rate, block_size));
    }

    fn process(
        &mut self,
        inputs: ProcessorInputs,
        outputs: ProcessorOutputs,
    ) -> Result<(), ProcessorError> {
        self.with_inner(|graph| graph.process(inputs, outputs))
    }
}
