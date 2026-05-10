import { NeuralNetwork } from '../NeuralNetwork/NeuralNetwork.js';
import { Tokenizer } from '../NeuralNetwork/utils/tokenizer.js';
import fs from "fs";

const texts = fs.readFileSync("./dataset/words.txt", "utf-8")
    .split("\n")
    .map(t => t.trim())
    .filter(Boolean);

console.log(`${texts.length} frases carregadas.\n`);

const tokenizer = new Tokenizer();
tokenizer.addWord("<start>");
tokenizer.addWord("<end>");
tokenizer.buildVocab(texts);

const vocabSize = tokenizer.getVocabSize();
console.log("Vocab size:", vocabSize);

// Pré-computa todas as amostras uma vez (mais eficiente)
function buildSamples(texts) {
    const samples = [];
    for (const text of texts) {
        const words = ["<start>", ...text.split(" "), "<end>"];
        for (let i = 0; i < words.length - 1; i++) {
            const inputIdx  = tokenizer.wordToIndex(words[i]);
            const targetIdx = tokenizer.wordToIndex(words[i + 1]);
            if (inputIdx === -1 || targetIdx === -1) continue;

            const input  = tokenizer.oneHot(inputIdx,  vocabSize);
            const target = tokenizer.oneHot(targetIdx, vocabSize);
            samples.push({ input, target });
        }
    }
    return samples;
}

const samples = buildSamples(texts);
console.log(`${samples.length} amostras geradas.\n`);

const nn = new NeuralNetwork(
    [vocabSize, 128, 128, vocabSize],
    { lr: 0.01 }
);

console.log("Treinando...\n");

for (let epoch = 0; epoch < 10000; epoch++) {
    let error = 0;

    // Embaralha a cada epoch para melhor generalização
    const shuffled = samples
        .map(s => ({ s, r: Math.random() }))
        .sort((a, b) => a.r - b.r)
        .map(x => x.s);

    for (const s of shuffled) {
        error += nn.train(s.input, s.target);
    }

    if (epoch % 500 === 0) {
        console.log(`Epoch: ${epoch} | Error: ${error.toFixed(4)}`);
    }
}

const model = {
    tokenizer: {
        vocab: tokenizer.vocab,
        reverseVocab: tokenizer.reverseVocab,
        nextIndex: tokenizer.nextIndex
    },
    weights: nn.weights,
    biases: nn.biases
};

fs.mkdirSync("./model", { recursive: true });
fs.writeFileSync("./model/model.json", JSON.stringify(model));
console.log("\nModelo salvo em ./model/model.json");