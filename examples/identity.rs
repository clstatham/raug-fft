use raug::prelude::*;
use raug_ext::prelude::*;
use raug_fft::prelude::*;

fn main() {
    env_logger::init();

    let graph = Graph::new(0, 2);

    let sine = BlSawOscillator::default().node(&graph, 440.0);

    let fft = graph.node({
        let fft_graph = FftGraphBuilder::<Fft1024>::new(256, raug_fft::WindowFunction::Hann);
        let fft_input = fft_graph.add_audio_input();
        let fft_output = fft_graph.add_audio_output();

        fft_output
            .input(0)
            .unwrap()
            .connect(fft_input.output(0).unwrap())
            .unwrap();
        fft_graph
    });

    fft.input(0).connect(sine.output(0));

    let fft = fft * 0.2;

    graph.dac((&fft, &fft));

    graph
        .play(
            CpalOut::spawn(&AudioBackend::Default, &AudioDevice::Default)
                .record_to_wav("identity.wav"),
            // WavFileOut::new("identity.wav", 48000.0, 1024, 2, None),
        )
        .unwrap()
        .run_for(Duration::from_secs(10))
        .unwrap();
}
