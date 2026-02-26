class StreamingAudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];    // array of Float32Array chunks
    this.offset = 0;    // read position in first chunk
    this.finishing = false;
    this.port.onmessage = (e) => {
      if (e.data.type === 'chunk') {
        this.queue.push(e.data.samples);
      } else if (e.data.type === 'finish') {
        this.finishing = true;
      } else if (e.data.type === 'clear') {
        this.queue = [];
        this.offset = 0;
        this.finishing = false;
      }
    };
  }

  process(inputs, outputs) {
    const out = outputs[0][0];
    let written = 0;
    while (written < out.length && this.queue.length > 0) {
      const chunk = this.queue[0];
      const available = chunk.length - this.offset;
      const needed = out.length - written;
      const n = Math.min(available, needed);
      out.set(chunk.subarray(this.offset, this.offset + n), written);
      written += n;
      this.offset += n;
      if (this.offset >= chunk.length) {
        this.queue.shift();
        this.offset = 0;
      }
    }
    // Fill remainder with silence
    for (let i = written; i < out.length; i++) out[i] = 0;
    if (this.finishing && this.queue.length === 0) {
      this.port.postMessage({ type: 'ended' });
      this.finishing = false;
    }
    return true;
  }
}

registerProcessor('streaming-audio', StreamingAudioProcessor);
