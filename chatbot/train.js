import { NeuralNetwork } from '../NeuralNetwork/NeuralNetwork.js';
import { Tokenizer } from '../NeuralNetwork/utils/tokenizer.js';
import fs from "fs";

const lines = fs.readFileSync("./dataset/chat.txt", "utf-8")
    .split("\n").map(l => l.trim()).filter(Boolean);

const pairs = lines.map(line => {
    const [input, output] = line.split("|").map(s => s.trim());
    return { input, output };
});

console.log(`${pairs.length} pares carregados.\n`);

const tokenizer = new Tokenizer();
tokenizer.addWord("<pad>");
tokenizer.addWord("<unk>");
tokenizer.addWord("<start>");
tokenizer.addWord("<end>");

const allTexts = pairs.flatMap(p => [p.input, p.output]);
tokenizer.buildVocab(allTexts);

const vocabSize = tokenizer.getVocabSize();
console.log("Vocab size:", vocabSize);

const inputSize = vocabSize * 2;

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

function buildSamples(pairs) {
    const samples = [];
    const startIdx = tokenizer.wordToIndex("<start>");

    for (const pair of pairs) {
        const inputWords = pair.input.toLowerCase().trim().split(/\s+/);
        const outputWords = pair.output.toLowerCase().trim().split(/\s+/);

        const bow = bagOfWords(inputWords);
        const sequence = [startIdx, ...outputWords.map(w => tokenizer.wordToIndex(w)).filter(i => i !== -1)];
        const endIdx = tokenizer.wordToIndex("<end>");

        for (let i = 0; i < sequence.length - 1 || i === 0; i++) {
            const prevIdx = sequence[i] ?? startIdx;
            const nextIdx = i < sequence.length - 1 ? sequence[i + 1] : endIdx;

            if (nextIdx === -1) continue;

            samples.push({
                input: [...bow, ...oneHot(prevIdx)],
                target: oneHot(nextIdx)
            });

            if (i >= sequence.length - 1) break;
        }

        // Adiciona amostra para o <end>
        const lastIdx = sequence[sequence.length - 1];
        if (lastIdx !== -1) {
            samples.push({
                input: [...bow, ...oneHot(lastIdx)],
                target: oneHot(endIdx)
            });
        }
    }
    return samples;
}

const samples = buildSamples(pairs);
console.log(`${samples.length} amostras de treino geradas.\n`);

const nn = new NeuralNetwork(
    [inputSize, 256, 256, vocabSize],
    { lr: 0.01 }
);

const EPOCHS = 1000;
console.log("Treinando...\n");

for (let epoch = 0; epoch <= EPOCHS; epoch++) {
    let totalError = 0;

    const shuffled = samples
        .map(s => ({ s, r: Math.random() }))
        .sort((a, b) => a.r - b.r)
        .map(x => x.s);

    for (const sample of shuffled) {
        totalError += nn.train(sample.input, sample.target);
    }

    if (epoch % 500 === 0) {
        console.log(`Epoch: ${epoch} | Error: ${totalError.toFixed(4)}`);
    }
}

const model = {
    vocabSize,
    inputSize,
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