use thiserror::Error;

pub mod builtins;
pub mod graph;
pub mod node;
pub mod processor;
pub mod signal;

pub mod prelude {
    pub use super::builtins::*;
    pub use super::graph::*;
    pub use super::node::*;
    pub use super::processor::*;
    pub use super::signal::*;
}

#[derive(Debug, Error)]
#[error("FFT error: {0}")]
pub enum FftError {
    RealFft(#[from] realfft::FftError),
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum WindowFunction {
    Rectangular,
    #[default]
    Hann,
    Hamming,
    Blackman,
    Nuttall,
    Triangular,
}

impl WindowFunction {
    pub fn generate(&self, length: usize) -> Vec<f32> {
        let mut buf = vec![1.0; length];
        self.apply(&mut buf);
        buf
    }

    pub fn apply(&self, buf: &mut [f32]) {
        let size = buf.len();
        match self {
            Self::Rectangular => {}
            Self::Hann => {
                for (x, y) in buf.iter_mut().zip(apodize::hanning_iter(size)) {
                    *x *= y as f32;
                }
            }
            Self::Hamming => {
                for (x, y) in buf.iter_mut().zip(apodize::hamming_iter(size)) {
                    *x *= y as f32;
                }
            }
            Self::Blackman => {
                for (x, y) in buf.iter_mut().zip(apodize::blackman_iter(size)) {
                    *x *= y as f32;
                }
            }
            Self::Nuttall => {
                for (x, y) in buf.iter_mut().zip(apodize::nuttall_iter(size)) {
                    *x *= y as f32;
                }
            }
            Self::Triangular => {
                for (x, y) in buf.iter_mut().zip(apodize::triangular_iter(size)) {
                    *x *= y as f32;
                }
            }
        }
    }
}
