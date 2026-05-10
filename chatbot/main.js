import { NeuralNetwork } from '../NeuralNetwork/NeuralNetwork.js';
import { Tokenizer } from '../NeuralNetwork/utils/tokenizer.js';
import fs from "fs";
import readline from "readline";

const data = JSON.parse(fs.readFileSync("./model/model.json", "utf-8"));

const tokenizer = new Tokenizer();
tokenizer.import(data.tokenizer);

const { vocabSize, inputSize } = data;

const nn = new NeuralNetwork(
    [inputSize, 256, 256, vocabSize],
    { lr: 0.01 }
);
nn.weights = data.weights;
nn.biases = data.biases;

function oneHot(index) {
    const v = new Array(vocabSize).fill(0);
    v[index] = 1;
    return v;
}

function bagOfWords(words) {
    const v = new Array(vocabSize).fill(0);
    for (const w of words) {
        const idx = tokenizer.wordToIndex(w);
        if (idx !== -1) v[idx]++;
    }
    const total = words.length || 1;
    return v.map(x => x / total);
}

function sampleWithTemperature(probs, temperature = 0.5) {
    const scaled = probs.map(p => Math.pow(p + 1e-10, 1 / temperature));
    const sum = scaled.reduce((a, b) => a + b, 0);
    const norm = scaled.map(p => p / sum);
    let r = Math.random(), acc = 0;
    for (let i = 0; i < norm.length; i++) {
        acc += norm[i];
        if (r < acc) return i;
    }
    return norm.length - 1;
}

function chat(userMessage, maxWords = 20, temperature = 0.3) {
    const inputWords = userMessage.toLowerCase().trim().split(/\s+/);
    const bow = bagOfWords(inputWords);

    const startIdx = tokenizer.wordToIndex("<start>");
    const endIdx   = tokenizer.wordToIndex("<end>");
    const stopSet  = new Set(["<pad>", "<unk>"]);

    let prevIdx = startIdx;
    const result = [];

    for (let step = 0; step < maxWords; step++) {
        const input = [...bow, ...oneHot(prevIdx)];
        const probs = nn.predict(input);
        const idx = sampleWithTemperature(probs, temperature);
        const word = tokenizer.indexToWord(idx);

        if (idx === endIdx || !word) break;
        if (stopSet.has(word)) { prevIdx = idx; continue; }

        result.push(word);
        prevIdx = idx;
    }

    return result.length > 0 ? result.join(" ") : "não entendi, pode reformular?";
}

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

console.log('\nDigite sua mensagem (ou "sair" para encerrar)\n');

function ask() {
    rl.question("você: ", (input) => {
        const trimmed = input.trim();
        if (!trimmed) { ask(); return; }
        if (trimmed.toLowerCase() === "sair") {
            console.log("\nassistente: até logo! 👋\n");
            rl.close();
            return;
        }
        console.log(`\nassistente: ${chat(trimmed)}\n`);
        ask();
    });
}

ask();