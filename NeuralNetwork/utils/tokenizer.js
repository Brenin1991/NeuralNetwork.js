export class Tokenizer {
    constructor() {
        this.vocab = {};
        this.reverseVocab = {};
        this.nextIndex = 0;
    }

    addWord(word) {
        if (!(word in this.vocab)) {
            this.vocab[word] = this.nextIndex;
            this.reverseVocab[this.nextIndex] = word;
            this.nextIndex++;
        }
    }

    buildVocab(texts) {
        texts.forEach(text => {
            text.split(" ").forEach(word => {
                this.addWord(word);
            });
        });
    }

    wordToIndex(word) {
        return this.vocab[word] ?? -1;
    }

    indexToWord(index) {
        return this.reverseVocab[index] ?? null;
    }

    oneHot(index, size = this.nextIndex) {
        let vec = Array(size).fill(0);
        if (index >= 0) {
            vec[index] = 1;
        }
        return vec;
    }

    getVocabSize() {
        return this.nextIndex;
    }

    export() {
        return {
            vocab: this.vocab,
            reverseVocab: this.reverseVocab,
            nextIndex: this.nextIndex
        };
    }

    import(data) {
        this.vocab = data.vocab;
        this.reverseVocab = data.reverseVocab;
        this.nextIndex = data.nextIndex;
    }
}