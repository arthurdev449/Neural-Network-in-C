#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "config_nn.c"
#include "foward_pass.c"
#include "back_propagation.c"
#include "sigmoid_fuctions.c"

// --- Definitions of globals declared in config_nn.c ---
double weights_input_hidden[NUM_INPUT_NODES][NUM_HIDDEN_NODES];
double weights_output_hidden[NUM_HIDDEN_NODES][NUM_OUTPUT_NODES];
double xor_inputs[NUM_TRAINING_SETS][NUM_INPUT_NODES];
double xor_expected[NUM_TRAINING_SETS][NUM_OUTPUT_NODES];
double hidden_layer_output[NUM_HIDDEN_NODES];
double final_output;
double learning_rate;
long epochs;

// Initialize the globals (in config_nn.c)
void nn_config(void) {
    srand(105); // deterministic seed; change to time(NULL) for varied runs
    learning_rate = 0.1;
    epochs = 1000000;

    // Initialize weights to small random values [0,1)
    for (int i = 0; i < NUM_INPUT_NODES; ++i) {
        for (int j = 0; j < NUM_HIDDEN_NODES; ++j) {
            weights_input_hidden[i][j] = ((double)rand() / RAND_MAX);
        }
    }
    for (int i = 0; i < NUM_HIDDEN_NODES; ++i) {
        for (int j = 0; j < NUM_OUTPUT_NODES; ++j) {
            weights_output_hidden[i][j] = ((double)rand() / RAND_MAX);
        }
    }

    // XOR dataset
    xor_inputs[0][0] = 0; xor_inputs[0][1] = 0; xor_expected[0][0] = 0;
    xor_inputs[1][0] = 0; xor_inputs[1][1] = 1; xor_expected[1][0] = 1;
    xor_inputs[2][0] = 1; xor_inputs[2][1] = 0; xor_expected[2][0] = 1;
    xor_inputs[3][0] = 1; xor_inputs[3][1] = 1; xor_expected[3][0] = 0;
}



int main () {
    nn_config();

    // --- 2. Loop de Treinamento Principal ---
    for (int e = 0; e < epochs; e++) {
        
        // Loop que passa por cada um dos 4 exemplos do dataset
        for (int i = 0; i < NUM_TRAINING_SETS; i++) {

            double current_input[NUM_INPUT_NODES] = {xor_inputs[i][0], xor_inputs[i][1]};
            double current_expected_output = xor_expected[i][0];

            double hidden_layer_output[NUM_HIDDEN_NODES];
            double final_output;

            forward_pass(current_input, weights_input_hidden, weights_output_hidden, &final_output, hidden_layer_output);

            back_propagation(current_input, current_expected_output, learning_rate, weights_input_hidden, weights_output_hidden, hidden_layer_output, final_output);
        }
    }

    // --- 3. Teste Final (DEPOIS de todo o treinamento) ---
    printf("\n--- Testando a rede treinada ---\n");
    for (int i = 0; i < NUM_TRAINING_SETS; i++) {
        double current_input[NUM_INPUT_NODES];
        for (int k = 0; k < NUM_INPUT_NODES; ++k) current_input[k] = xor_inputs[i][k];

        double hidden_layer_output[NUM_HIDDEN_NODES];
        double final_output;
        forward_pass(current_input, weights_input_hidden, weights_output_hidden, &final_output, hidden_layer_output);

        int final_prediction = (final_output > 0.98) ? 1 : 0;
        printf("Entrada: {%.0f, %.0f} -> Saida Esperada: %.0f, Previsao da Rede: %f, Precisao de 98 ou mais: %d\n",
               current_input[0], current_input[1], xor_expected[i][0], final_output, final_prediction);
    }

    return 0;
}