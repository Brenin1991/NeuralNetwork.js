import fs from "fs";
import path from "path";
import { decode as wavDecode } from "wav-decoder";
import Meyda from "meyda";
import { NeuralNetwork } from "../NeuralNetwork/NeuralNetwork.js";

const DATASET_DIR  = "./dataset";
const SILENCE_DIR  = "./dataset/silence";
const SAMPLE_SIZE  = 256;
const MFCC_COUNT   = 40;
const EPOCHS       = 500;
const LR           = 0.0003;
const LR_DECAY     = 0.995;

Meyda.numberOfMFCCCoefficients = MFCC_COUNT;

const instrumentClasses = fs.readdirSync(DATASET_DIR).filter(f => {
    const full = path.join(DATASET_DIR, f);
    return fs.statSync(full).isDirectory() && f !== "silence";
});

const classes = [...instrumentClasses, "silence"];

console.log("Classes:", classes);
console.log(`  → ${instrumentClasses.length} instrumentos + 1 silence = ${classes.length} classes\n`);

function oneHot(index, size) {
    const arr = Array(size).fill(0);
    arr[index] = 1;
    return arr;
}

function extractRawFeatures(audioBuffer) {
    const frames = [];
    for (let i = 0; i + SAMPLE_SIZE <= audioBuffer.length; i += SAMPLE_SIZE) {
        const frame = audioBuffer.slice(i, i + SAMPLE_SIZE);
        const mfcc = Meyda.extract("mfcc", frame) || Array(MFCC_COUNT).fill(0);
        const rms = Meyda.extract("rms", frame) ?? 0;
        const zcr = Meyda.extract("zcr", frame) ?? 0;
        const centroid = Meyda.extract("spectralCentroid", frame) ?? 0;
        const flatness = Meyda.extract("spectralFlatness", frame) ?? 0;
        const rolloff = Meyda.extract("spectralRolloff",  frame) ?? 0;
        const vector = [rms, zcr, centroid, flatness, rolloff, ...mfcc].map(v => (isFinite(v) ? v : 0));

        frames.push(vector);
    }
    return frames;
}

function generateSilenceFrames(count, inputSize) {
    const frames = [];
    for (let i = 0; i < count; i++) {
        const audio = new Float32Array(SAMPLE_SIZE);
        for (let j = 0; j < SAMPLE_SIZE; j++) {
            audio[j] = (Math.random() * 2 - 1) * 0.002;
        }
        frames.push(extractRawFeatures(audio)[0] || new Array(inputSize).fill(0));
    }
    return frames;
}

async function loadWav(filePath) {
    const buffer  = fs.readFileSync(filePath);
    const decoded = await wavDecode(
        buffer.buffer.slice(
            buffer.byteOffset,
            buffer.byteOffset + buffer.byteLength
        )
    );
    return {
        audio:      decoded.channelData[0],
        sampleRate: decoded.sampleRate
    };
}

const rawDataset = [];

for (const className of instrumentClasses) {
    const classDir = path.join(DATASET_DIR, className);
    const files = fs.readdirSync(classDir).filter(f => f.toLowerCase().endsWith(".wav"));

    for (const file of files) {
        const fullPath = path.join(classDir, file);
        console.log("Loading:", fullPath);

        let audio;
        try {
            ({ audio } = await loadWav(fullPath));
        } catch (err) {
            console.warn(`Ignorado: ${err.message}`);
            continue;
        }

        for (const frame of extractRawFeatures(audio)) {
            rawDataset.push({ input: frame, target: className });
        }
    }
}

console.log(`\nSamples de instrumentos: ${rawDataset.length}`);
if (rawDataset.length === 0) {
    console.error("Nenhum sample de instrumento encontrado.");
    process.exit(1);
}

const inputSize = rawDataset[0].input.length;
console.log(`Input size: ${inputSize}  (esperado: ${5 + MFCC_COUNT})`);

if (inputSize !== 5 + MFCC_COUNT) {
    console.error("Input size errado. Verifique Meyda.numberOfMFCCCoefficients.");
    process.exit(1);
}

let silenceFrames = [];

if (fs.existsSync(SILENCE_DIR)) {
    const silenceFiles = fs.readdirSync(SILENCE_DIR)
        .filter(f => f.toLowerCase().endsWith(".wav"));

    for (const file of silenceFiles) {
        const fullPath = path.join(SILENCE_DIR, file);
        console.log("Loading silence:", fullPath);
        try {
            const { audio } = await loadWav(fullPath);
            silenceFrames.push(...extractRawFeatures(audio));
        } catch (err) {
            console.warn(`  ⚠️  Ignorado: ${err.message}`);
        }
    }
}

const samplesPerClass = Math.round(rawDataset.length / instrumentClasses.length);
const silenceNeeded   = Math.max(samplesPerClass - silenceFrames.length, 0);

if (silenceNeeded > 0) {
    console.log(`Gerando ${silenceNeeded} frames de silêncio sintético...`);
    silenceFrames.push(...generateSilenceFrames(silenceNeeded, inputSize));
}

for (const frame of silenceFrames) {
    rawDataset.push({ input: frame, target: "silence" });
}

console.log(`Total samples (com silence): ${rawDataset.length}\n`);

const mean = new Array(inputSize).fill(0);
for (const s of rawDataset)
    for (let i = 0; i < inputSize; i++) mean[i] += s.input[i];
for (let i = 0; i < inputSize; i++) mean[i] /= rawDataset.length;

const std = new Array(inputSize).fill(0);
for (const s of rawDataset)
    for (let i = 0; i < inputSize; i++) std[i] += (s.input[i] - mean[i]) ** 2;
for (let i = 0; i < inputSize; i++) std[i] = Math.sqrt(std[i] / rawDataset.length) || 1;

console.log("Z-score calculado.");

const dataset = rawDataset.map(s => ({
    input:  s.input.map((v, i) => (v - mean[i]) / std[i]),
    target: s.target
}));

const nn = new NeuralNetwork(
    [inputSize, 256, 128, classes.length],
    { lr: LR, momentum: 0.9 }
);

console.log(`\nTreinando... (${EPOCHS} epochs, lr=${LR}, decay=${LR_DECAY}, momentum=0.9)\n`);

let lr = LR;

for (let epoch = 0; epoch < EPOCHS; epoch++) {
    let error = 0;

    for (let i = dataset.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [dataset[i], dataset[j]] = [dataset[j], dataset[i]];
    }

    for (const sample of dataset) {
        const target = oneHot(classes.indexOf(sample.target), classes.length);
        error += nn.train(sample.input, target);
    }

    lr *= LR_DECAY;
    if (nn.lr !== undefined) nn.lr = lr;

    if (epoch % 10 === 0 || epoch < 10) {
        const avg = error / dataset.length;
        console.log(`Epoch ${String(epoch).padStart(3)} | Loss: ${error.toFixed(2).padStart(10)} | Avg: ${avg.toFixed(4)} | lr: ${lr.toFixed(6)}`);
    }
}

const modelData = {
    classes,
    inputSize,
    mfccCount: MFCC_COUNT,
    mean:      Array.from(mean),
    std:       Array.from(std),
    weights:   nn.weights,
    biases:    nn.biases
};

fs.writeFileSync("./model/audio_model.json", JSON.stringify(modelData));
console.log("\nModelo salvo!");
console.log(`Classes: ${classes.join(", ")}`);