import { NeuralNetwork } from '../NeuralNetwork/NeuralNetwork.js';
import { Tokenizer } from '../NeuralNetwork/utils/tokenizer.js';
import fs from "fs";
import readline from "readline";

const data = JSON.parse(fs.readFileSync("./model/model.json", "utf-8"));

const tokenizer = new Tokenizer();
tokenizer.import(data.tokenizer);

const vocabSize = tokenizer.getVocabSize();

const nn = new NeuralNetwork(
    [vocabSize, 128, 128, vocabSize],
    { lr: 0.01 }
);
nn.weights = data.weights;
nn.biases = data.biases;

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

function autocomplete(seed, steps = 10, temperature = 0.5) {
    const words = seed.toLowerCase().trim().split(/\s+/);
    let word = words[words.length - 1];
    const result = [...words];

    for (let i = 0; i < steps; i++) {
        const wordIdx = tokenizer.wordToIndex(word);

        if (wordIdx === -1) {
            console.log(`Palavra desconhecida: "${word}"`);
            break;
        }

        const input  = tokenizer.oneHot(wordIdx, vocabSize);
        const probs  = nn.predict(input);
        const nextIdx = sampleWithTemperature(probs, temperature);
        const nextWord = tokenizer.indexToWord(nextIdx);

        if (!nextWord || nextWord === "<end>" || nextWord === "<start>") break;

        result.push(nextWord);
        word = nextWord;
    }

    return result.join(" ");
}

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

console.log('\nDigite uma palavra inicial (ou "sair")\n');

function ask() {
    rl.question("> ", (input) => {
        const trimmed = input.trim();
        if (!trimmed) { ask(); return; }
        if (trimmed.toLowerCase() === "sair") {
            rl.close();
            return;
        }
        console.log("\n" + autocomplete(trimmed) + "\n");
        ask();
    });
}

ask();