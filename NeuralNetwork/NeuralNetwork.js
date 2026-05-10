export class NeuralNetwork {
    constructor(layers, options = {}) {
        this.layers = layers;
        this.lr = options.lr || 0.001;
        this.weights = [];
        this.biases = [];
        this.initWeights();
    }

    softmax(arr) {
        const max = Math.max(...arr);
        const exps = arr.map(x => Math.exp(x - max));
        const sum = exps.reduce((a, b) => a + b, 0);
        return exps.map(x => x / sum);
    }

    relu(x) { return Math.max(0, x); }
    relu_deriv(x) { return x > 0 ? 1 : 0; }

    softmax(arr) {
        let max = Math.max(...arr);
        let exps = arr.map(x => Math.exp(x - max));
        let sum = exps.reduce((a, b) => a + b, 0) || 1e-8;
        return exps.map(e => e / sum);
    }

    randomMatrix(rows, cols) {
        return Array.from({ length: rows }, () =>
            Array.from({ length: cols }, () => Math.random() * 0.2 - 0.1)
        );
    }

    randomVector(size) {
        return Array.from({ length: size }, () => Math.random() * 0.2 - 0.1);
    }

    initWeights() {
        for (let i = 0; i < this.layers.length - 1; i++) {
            this.weights.push(
                this.randomMatrix(this.layers[i + 1], this.layers[i])
            );
            this.biases.push(this.randomVector(this.layers[i + 1]));
        }
    }

    forward(input) {
        let activations = [input];
        let zs = [];

        for (let l = 0; l < this.weights.length; l++) {
            let w = this.weights[l];
            let b = this.biases[l];
            let prev = activations[l];
            let z = [];
            let a = [];
            for (let i = 0; i < w.length; i++) {
                let soma = 0;
                for (let j = 0; j < w[i].length; j++) {
                    soma += w[i][j] * prev[j];
                }
                soma += b[i];
                z.push(soma);
                if (l !== this.weights.length - 1) {
                    a.push(this.relu(soma));
                }
            }

            // saída → softmax
            if (l === this.weights.length - 1) {
                a = this.softmax(z);
            }
            zs.push(z);
            activations.push(a);
        }
        return { activations, zs };
    }

    train(x, t) {
        let { activations, zs } = this.forward(x);
        let deltas = [];
        let output = activations.at(-1);

        // cross-entropy + softmax
        let delta = output.map((y, i) => y - t[i]);
        deltas.push(delta);

        // backprop
        for (let l = this.weights.length - 2; l >= 0; l--) {
            let newDelta = [];
            for (let i = 0; i < this.weights[l].length; i++) {
                let sum = 0;
                for (let j = 0; j < deltas[0].length; j++) {
                    sum += this.weights[l + 1][j][i] * deltas[0][j];
                }
                sum *= this.relu_deriv(zs[l][i]);
                newDelta.push(sum);
            }
            deltas.unshift(newDelta);
        }

        // update
        for (let l = 0; l < this.weights.length; l++) {
            let a_prev = activations[l];
            for (let i = 0; i < this.weights[l].length; i++) {
                for (let j = 0; j < this.weights[l][i].length; j++) {
                    this.weights[l][i][j] -= this.lr * deltas[l][i] * a_prev[j];
                }
                this.biases[l][i] -= this.lr * deltas[l][i];
            }
        }

        // loss
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