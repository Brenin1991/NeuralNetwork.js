import { NeuralNetwork } from '../NeuralNetwork/NeuralNetwork.js';
import { Tokenizer } from '../NeuralNetwork/utils/tokenizer.js';

import fs from "fs";
import readline from "readline";

const data = JSON.parse(
    fs.readFileSync(
        "./model/model.json",
        "utf-8"
    )
);

const tokenizer = new Tokenizer();

tokenizer.import(data.tokenizer);

const vocabSize = tokenizer.getVocabSize();

const nn = new NeuralNetwork(
    [vocabSize, 64, 64, vocabSize],
    { lr: 0.001 }
);

nn.weights = data.weights;
nn.biases = data.biases;

function sample(probs) {
    let r = Math.random();
    let sum = 0;
    for (let i = 0; i < probs.length; i++) {
        sum += probs[i];
        if (r < sum) {
            return i;
        }
    }
    return probs.length - 1;
}

function autocomplete(
    seed,
    steps = 10,
    temperature = 0.1
) {
    let words = seed
        .toLowerCase()
        .trim()
        .split(/\s+/);

    let word = words[words.length - 1];
    let result = [...words];
    for (let i = 0; i < steps; i++) {
        const wordIndex = tokenizer.wordToIndex(word);

        if (wordIndex === -1) {
            console.log(`Palavra desconhecida: ${word}`);
            break;
        }
        const input = tokenizer.oneHot(
            wordIndex,
            vocabSize
        );

        const output = nn.predict(input);

        let probs = nn.softmax(output);

        // temperature
        probs = probs.map(
            p => Math.pow(
                p,
                1 / temperature
            )
        );

        // renormaliza
        const sum = probs.reduce(
            (a, b) => a + b,
            0
        );

        probs = probs.map(p => p / sum);

        // sample
        const nextIndex = sample(probs);

        const nextWord = tokenizer.indexToWord(nextIndex);

        if (!nextWord || nextWord === "<end>") {
            break;
        }
        result.push(nextWord);

        word = nextWord;
    }
    return result.join(" ");
}

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

console.log("Digite uma palavra inicial\n");

function ask() {
    rl.question("> ", (input) => {
        if (input === "exit") {
            rl.close();
            return;
        }
        const response = autocomplete(input);

        console.log("\n" + response + "\n");

        ask();
    });
}

ask();