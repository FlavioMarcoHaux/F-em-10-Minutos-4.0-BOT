

// Helper function to decode base64 string to Uint8Array
export function decode(base64: string): Uint8Array {
    const binaryString = atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes;
}

// Helper function to create a WAV file Blob from raw PCM data
export function createWavFile(pcmData: Uint8Array, numChannels: number, sampleRate: number, bitsPerSample: number): Blob {
    const dataSize = pcmData.length;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    function writeString(view: DataView, offset: number, string: string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    // RIFF chunk descriptor
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numChannels * (bitsPerSample / 8), true); // byteRate
    view.setUint16(32, numChannels * (bitsPerSample / 8), true); // blockAlign
    view.setUint16(34, bitsPerSample, true);
    writeString(view, 36, 'data');
    view.setUint32(40, dataSize, true);
    new Uint8Array(buffer, 44).set(pcmData);

    return new Blob([buffer], { type: 'audio/wav' });
}

// Helper function to create just the WAV header as a Blob
export function createWavHeader(dataSize: number, numChannels: number, sampleRate: number, bitsPerSample: number): Blob {
    const buffer = new ArrayBuffer(44);
    const view = new DataView(buffer);

    function writeString(view: DataView, offset: number, string: string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    // RIFF chunk descriptor
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numChannels * (bitsPerSample / 8), true); // byteRate
    view.setUint16(32, numChannels * (bitsPerSample / 8), true); // blockAlign
    view.setUint16(34, bitsPerSample, true);
    writeString(view, 36, 'data');
    view.setUint32(40, dataSize, true);

    return new Blob([buffer], { type: 'audio/wav' });
}

// Helper function to decode raw PCM data into an AudioBuffer for playback
export async function decodePcmAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number,
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

// Helper to calculate duration from bytes (useful for debug)
export function getWavDuration(bytesLength: number, sampleRate: number = 24000, numChannels: number = 1, bitDepth: number = 16): number {
    const bytesPerSample = bitDepth / 8;
    const totalSamples = bytesLength / (numChannels * bytesPerSample);
    return totalSamples / sampleRate;
}