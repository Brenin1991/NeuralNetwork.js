# NeuralNetwork.js

> Implementação leve de redes neurais artificiais em JavaScript nativo — otimizada para rodar em múltiplas plataformas, incluindo mobile.

---

## Sobre

**NeuralNetwork.js** é uma implementação minimalista de uma **MLP (Multi-Layer Perceptron)**, também conhecida como *Feedforward Fully Connected Neural Network*. O objetivo é ser leve, sem dependências externas, e funcionar em qualquer ambiente JavaScript — browser, Node.js ou mobile.

---

## Características

- Camadas densas (*fully connected*)
- *Forward propagation*
- *Backpropagation*
- Ativação **ReLU** nas camadas ocultas
- Ativação **Softmax** na camada de saída
- Função de perda **Cross-Entropy**
- Otimização via **Gradient Descent**

---

## Arquitetura

```
Entrada -> Dense + ReLU -> Dense + ReLU -> Dense + Softmax -> Saída
```

---

## Uso

```js
import { NeuralNetwork } from "./NeuralNetwork/NeuralNetwork.js";

const nn = new NeuralNetwork(
    [100, 256, 128, 5],
    { lr: 0.0003 }
);
```

### Parâmetros

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `layers[0]` | `100` | Dimensão da entrada |
| `layers[1]` | `256` | Neurônios da camada oculta 1 |
| `layers[2]` | `128` | Neurônios da camada oculta 2 |
| `layers[3]` | `5`   | Dimensão da saída (classes) |
| `lr`        | `0.0003` | Taxa de aprendizado (*learning rate*) |

---

## Aplicações

- Classificação de texto, categorias, áudio e imagens
- Autocomplete
- Chatbot
- Detector de intenção
- Análise de sentimento
- Geração procedural simples (texto, notas musicais, sequências)


## Licença

MIT