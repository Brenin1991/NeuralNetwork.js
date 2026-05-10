import { NeuralNetwork } from '../NeuralNetwork/NeuralNetwork.js';
import { Tokenizer } from '../NeuralNetwork/utils/tokenizer.js';

import fs from "fs";

// load dataset
const texts = fs.readFileSync("./dataset/words.txt", "utf-8")
    .split("\n")
    .map(t => t.trim())
    .filter(Boolean);

console.log(texts);

//tokenize
const tokenizer = new Tokenizer();

// tokens especiais
tokenizer.addWord("<start>");
tokenizer.addWord("<end>");

// build vocab
tokenizer.buildVocab(texts);

const vocabSize = tokenizer.getVocabSize();

console.log("Vocab size:", vocabSize);

// init nn
const nn = new NeuralNetwork(
    [vocabSize, 64, 64, vocabSize],
    { lr: 0.001 }
);

function createSamples(text) {
    let words = [
        "<start>",
        ...text.split(" "),
        "<end>"
    ];

    let samples = [];

    for (let i = 0; i < words.length - 1; i++) {

        samples.push({
            input: words[i],
            target: words[i + 1]
        });
    }

    return samples;
}

// train
console.log("Treinando...\n");

for (let epoch = 0; epoch < 10000; epoch++) {

    let error = 0;

    for (let text of texts) {

        const samples =
            createSamples(text);

        for (let s of samples) {
            const inputIndex = tokenizer.wordToIndex(s.input);
            const targetIndex = tokenizer.wordToIndex(s.target);

            // proteção
            if (inputIndex === -1 || targetIndex === -1) {
                continue;
            }

            const input = tokenizer.oneHot(
                inputIndex,
                vocabSize
            );

            const target = tokenizer.oneHot(
                targetIndex,
                vocabSize
            );
            error += nn.train(input, target);
        }
    }
    if (epoch % 500 === 0) {
        console.log(`Epoch: ${epoch} | ` + `Error: ${error.toFixed(4)}`);
    }
}

// save model
const model = {
    tokenizer: {
        vocab: tokenizer.vocab,
        reverseVocab: tokenizer.reverseVocab,
        nextIndex: tokenizer.nextIndex
    },

    weights: nn.weights,
    biases: nn.biases
};

fs.writeFileSync(
    "./model/model.json",
    JSON.stringify(model)
);

console.log("\nModelo salvo em model.json");