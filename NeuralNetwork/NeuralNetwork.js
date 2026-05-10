export class NeuralNetwork {
    constructor(layers, options = {}) {
        this.layers = layers;
        this.lr = options.lr || 0.001;
        this.leakyAlpha = options.leakyAlpha || 0.01;
        this.weights = [];
        this.biases = [];
        this.initWeights();
    }

    relu(x)       { return x > 0 ? x : this.leakyAlpha * x; }
    relu_deriv(x) { return x > 0 ? 1 : this.leakyAlpha; }

    softmax(arr) {
        const max = Math.max(...arr);
        const exps = arr.map(x => Math.exp(x - max));
        const sum = exps.reduce((a, b) => a + b, 0) || 1e-8;
        return exps.map(e => e / sum);
    }

    heRandom(fanIn) {
        const u1 = Math.random() + 1e-10;
        const u2 = Math.random();
        const normal = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        return normal * Math.sqrt(2 / (1 + this.leakyAlpha ** 2) / fanIn);
    }

    randomMatrix(rows, cols) {
        return Array.from({ length: rows }, () =>
            Array.from({ length: cols }, () => this.heRandom(cols))
        );
    }

    initWeights() {
        for (let i = 0; i < this.layers.length - 1; i++) {
            this.weights.push(
                this.randomMatrix(this.layers[i + 1], this.layers[i])
            );
            // Biases em zero — padrão recomendado
            this.biases.push(new Array(this.layers[i + 1]).fill(0));
        }
    }

    forward(input) {
        let activations = [input];
        let zs = [];

        for (let l = 0; l < this.weights.length; l++) {
            const w = this.weights[l];
            const b = this.biases[l];
            const prev = activations[l];
            const isLastLayer = l === this.weights.length - 1;
            const z = [];
            let a = [];

            for (let i = 0; i < w.length; i++) {
                let soma = b[i];
                for (let j = 0; j < w[i].length; j++) {
                    soma += w[i][j] * prev[j];
                }
                z.push(soma);
                if (!isLastLayer) {
                    a.push(this.relu(soma));
                }
            }

            if (isLastLayer) {
                a = this.softmax(z);
            }

            zs.push(z);
            activations.push(a);
        }

        return { activations, zs };
    }

    train(x, t) {
        const { activations, zs } = this.forward(x);
        const output = activations.at(-1);
        const deltas = [];

        deltas.push(output.map((y, i) => y - t[i]));

        for (let l = this.weights.length - 2; l >= 0; l--) {
            const newDelta = [];
            const nextDelta = deltas[0];
            const nextWeights = this.weights[l + 1];

            for (let i = 0; i < this.weights[l].length; i++) {
                let sum = 0;
                for (let j = 0; j < nextDelta.length; j++) {
                    sum += nextWeights[j][i] * nextDelta[j];
                }
                newDelta.push(sum * this.relu_deriv(zs[l][i]));
            }
            deltas.unshift(newDelta);
        }

        for (let l = 0; l < this.weights.length; l++) {
            const a_prev = activations[l];
            for (let i = 0; i < this.weights[l].length; i++) {
                for (let j = 0; j < this.weights[l][i].length; j++) {
                    this.weights[l][i][j] -= this.lr * deltas[l][i] * a_prev[j];
                }
                this.biases[l][i] -= this.lr * deltas[l][i];
            }
        }

        // Cross-Entropy loss
        let loss = 0;
        for (let i = 0; i < output.length; i++) {
            loss -= t[i] * Math.log(output[i] + 1e-8);
        }
        return loss;
    }

    predict(x) {
        return this.forward(x).activations.at(-1);
    }
}