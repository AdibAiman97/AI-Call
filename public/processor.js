class PCMProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0][0];
    if (input) {
      const buffer = new ArrayBuffer(input.length * 2);
      const view = new DataView(buffer);
      for (let i = 0; i < input.length; i++) {
        let s = Math.max(-1, Math.min(1, input[i]));
        view.setInt16(i * 2, s * 0x7fff, true);
      }
      this.port.postMessage(buffer);
    }
    return true;
  }
}
registerProcessor("pcm-processor", PCMProcessor);
