use std::ops::{Deref, DerefMut};

use raug::signal::Signal;
pub use realfft::num_complex::Complex32;

mod sealed {
    pub trait Sealed {}
}

pub trait Fft: sealed::Sealed + Send + 'static {
    const N_FFT: usize;
    const N_REAL_BINS: usize = Self::N_FFT / 2 + 1;
    type AudioBlock: Signal + Deref<Target = [f32]> + DerefMut;
    type RealFft: Signal + Deref<Target = [Complex32]> + DerefMut;
    type ComplexFft: Signal + Deref<Target = [Complex32]> + DerefMut;
}

macro_rules! impl_fft_frame {
    ($($n:literal => $frame:ident, $audio_block:ident, $real:ident, $complex:ident),* $(,)?) => {
        $(
            pub struct $frame;

            impl sealed::Sealed for $frame {}

            impl Fft for $frame {
                const N_FFT: usize = $n;
                const N_REAL_BINS: usize = $n / 2 + 1;
                type AudioBlock = $audio_block;
                type RealFft = $real;
                type ComplexFft = $complex;
            }

            #[derive(Clone, Copy)]
            #[repr(transparent)]
            pub struct $audio_block([f32; $n]);

            impl Default for $audio_block {
                fn default() -> Self {
                    Self([0.0; $n])
                }
            }

            impl Signal for $audio_block {}

            impl Deref for $audio_block {
                type Target = [f32];

                fn deref(&self) -> &[f32] {
                    &self.0
                }
            }

            impl DerefMut for $audio_block {
                fn deref_mut(&mut self) -> &mut [f32] {
                    &mut self.0
                }
            }

            #[derive(Clone, Copy)]
            #[repr(transparent)]
            pub struct $real([Complex32; $n / 2 + 1]);

            impl Default for $real {
                fn default() -> Self {
                    Self([Complex32::ZERO; $n / 2 + 1])
                }
            }

            impl Signal for $real {}

            impl Deref for $real {
                type Target = [Complex32];

                fn deref(&self) -> &[Complex32] {
                    &self.0
                }
            }

            impl DerefMut for $real {
                fn deref_mut(&mut self) -> &mut [Complex32] {
                    &mut self.0
                }
            }


            #[derive(Clone, Copy)]
            #[repr(transparent)]
            pub struct $complex([Complex32; $n]);

            impl Default for $complex {
                fn default() -> Self {
                    Self([Complex32::ZERO; $n])
                }
            }

            impl Signal for $complex {}

            impl Deref for $complex {
                type Target = [Complex32];

                fn deref(&self) -> &[Complex32] {
                    &self.0
                }
            }

            impl DerefMut for $complex {
                fn deref_mut(&mut self) -> &mut [Complex32] {
                    &mut self.0
                }
            }
        )*
    };
}

impl_fft_frame! {
    64 => Fft64, Audio64, RealFft64, ComplexFft64,
    128 => Fft128, Audio128, RealFft128, ComplexFft128,
    256 => Fft256, Audio256, RealFft256, ComplexFft256,
    512 => Fft512, Audio512, RealFft512, ComplexFft512,
    1024 => Fft1024, Audio1024, RealFft1024, ComplexFft1024,
    2048 => Fft2048, Audio2048, RealFft2048, ComplexFft2048,
    4096 => Fft4096, Audio4096, RealFft4096, ComplexFft4096,
    8192 => Fft8192, Audio8192, RealFft8192, ComplexFft8192,
}
