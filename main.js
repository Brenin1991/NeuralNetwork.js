import { exec } from "child_process";
import readline from "readline";

const apps = ["audio-classification", "auto-complete", "chatbot"];

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

console.log("\nEscolha qual aplicação de Redes Neurais testar:\n");

apps.forEach((app, index) => {
    console.log(`${index + 1} - ${app}`);
});

function dialog() {
    rl.question("\nDigite o número da aplicação: ", (answer) => {
        const index = parseInt(answer) - 1;

        if (isNaN(index) || !apps[index]) {
            console.log("\nAplicação inválida.");
            dialog();
            return;
        }

        const selectedApp = apps[index];

        console.log(`\nAplicação selecionada: ${selectedApp}\n`);
        console.log("1 - Executar main.js");
        console.log("2 - Treinar train.js");

        rl.question("\nEscolha a ação: ", (action) => {

            let file = "";

            if (action === "1") {
                file = "main.js";
            }
            else if (action === "2") {
                file = "train.js";
            }
            else {
                console.log("\nAção inválida.");
                dialog();
                return;
            }

            console.log(`\nExecutando ${selectedApp}/${file}...\n`);

            exec(
                `start cmd /k "cd ${selectedApp} && node ${file}"`,
                (error) => {
                    if (error) {
                        console.error("Erro:", error.message);
                    }
                }
            );

            dialog();
        });
    });
}

dialog();