use raug::prelude::*;

pub trait FftProcessor
where
    Self: Send + 'static,
{
    fn name(&self) -> &str {
        raug::util::interned_short_type_name::<Self>()
    }

    fn input_spec(&self) -> Vec<SignalSpec>;
    fn output_spec(&self) -> Vec<SignalSpec>;

    fn create_output_buffers(&self, size: usize) -> Vec<AnyBuffer>;

    #[allow(unused)]
    fn allocate(&mut self, sample_rate: f32) {}
    #[allow(unused)]
    fn resize_buffers(&mut self, sample_rate: f32) {}

    fn process(&mut self, inputs: ProcessorInputs, outputs: ProcessorOutputs) -> ProcResult<()>;
}
