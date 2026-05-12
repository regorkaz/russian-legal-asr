// AudioWorklet: downsample from the device sample rate to 16 kHz and
// post Int16 PCM frames to the main thread.

class PCMDownsampler extends AudioWorkletProcessor {
    constructor(options) {
        super();
        this.targetRate = 16000;
        this.srcRate = sampleRate;  // provided by AudioWorkletGlobalScope
        this.ratio = this.srcRate / this.targetRate;
        this.residual = 0;      // fractional source index carried between process() calls
        this.batchSize = 1600;  // 100 ms of 16 kHz Int16 -> 3200 bytes
        this.batch = new Int16Array(this.batchSize);
        this.batchIdx = 0;
    }

    process(inputs) {
        const input = inputs[0];
        if (!input || input.length === 0) return true;

        const channel = input[0];  // mono (first channel)
        if (!channel) return true;

        // Linear-interpolate source -> target rate
        let i = this.residual;
        for (; i < channel.length; i += this.ratio) {
            const idx = Math.floor(i);
            const frac = i - idx;
            const a = channel[idx] || 0;
            const b = channel[idx + 1] !== undefined ? channel[idx + 1] : a;
            const sample = a + (b - a) * frac;
            const clipped = Math.max(-1, Math.min(1, sample));
            this.batch[this.batchIdx++] = clipped < 0 ? clipped * 0x8000 : clipped * 0x7FFF;

            if (this.batchIdx >= this.batchSize) {
                this.port.postMessage(this.batch.buffer.slice(0), [this.batch.buffer.slice(0)]);
                // Send a copy and replace buffer to avoid aliasing
                this.batch = new Int16Array(this.batchSize);
                this.batchIdx = 0;
            }
        }
        this.residual = i - channel.length;

        return true;
    }
}

registerProcessor("pcm-downsampler", PCMDownsampler);
