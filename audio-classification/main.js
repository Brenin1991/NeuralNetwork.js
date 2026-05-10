import fs from "fs";
import { decode as wavDecode } from "wav-decoder";
import Meyda from "meyda";
import { NeuralNetwork } from "../NeuralNetwork/NeuralNetwork.js";

const FRAME_SIZE = 256;

const model = JSON.parse(fs.readFileSync("./model/audio_model.json"));

const classes    = model.classes;
const inputSize  = model.weights[0][0].length;
const mean       = model.mean;
const std        = model.std;
const MFCC_COUNT = model.mfccCount || (inputSize - 5);

Meyda.numberOfMFCCCoefficients = MFCC_COUNT;

const nn = new NeuralNetwork(
    [inputSize, 128, 64, classes.length],
    { lr: 0.001 }
);

nn.weights = model.weights;
nn.biases  = model.biases;

console.log(`Modelo: classes=[${classes}] | inputSize=${inputSize} | MFCC=${MFCC_COUNT}`);

function extractFeatures(audioBuffer) {
    const frame = new Float32Array(FRAME_SIZE);
    const src = audioBuffer.slice(0, FRAME_SIZE);
    frame.set(src);

    const mfcc = Meyda.extract("mfcc", frame) || Array(MFCC_COUNT).fill(0);

    const raw = [
        Meyda.extract("rms", frame) ?? 0,
        Meyda.extract("zcr", frame) ?? 0,
        Meyda.extract("spectralCentroid", frame) ?? 0,
        Meyda.extract("spectralFlatness", frame) ?? 0,
        Meyda.extract("spectralRolloff", frame) ?? 0,
        ...mfcc
    ].map(v => isFinite(v) ? v : 0);

    return raw.map((v, i) => (v - mean[i]) / std[i]);
}

const file = process.argv[2];

if (!file) {
    console.error("Uso: node main.js <arquivo.wav>");
    process.exit(1);
}

const buffer = fs.readFileSync(file);

const decoded = await wavDecode(
    buffer.buffer.slice(
        buffer.byteOffset,
        buffer.byteOffset + buffer.byteLength
    )
);

const audio = decoded.channelData[0];

const features = extractFeatures(audio);

const output = nn.predict(features);

const idx = output.indexOf(Math.max(...output));

console.log("\nPredição:");

classes.forEach((c, i) => {
    console.log(`  ${c.padEnd(10)} ${(output[i] * 100).toFixed(1)}%`);
});

console.log("\nInstrumento:", classes[idx]);