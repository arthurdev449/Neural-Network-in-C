// Forward pass module — use config_nn for configuration and externs
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "config_nn.c"

void forward_pass(const double current_input[NUM_INPUT_NODES], double weights_input_hidden[NUM_INPUT_NODES][NUM_HIDDEN_NODES], double weights_output_hidden[NUM_HIDDEN_NODES][NUM_OUTPUT_NODES], double *out_final, double hidden_layer_output[NUM_HIDDEN_NODES]) {
    // 1. Calcular a camada oculta
    for (int h = 0; h < NUM_HIDDEN_NODES; h++) { // Para cada neurônio oculto 'h'
        double weighted_sum = 0;
        for (int j = 0; j < NUM_INPUT_NODES; j++) { // Para cada neurônio de entrada 'j'
             weighted_sum += current_input[j] * weights_input_hidden[j][h];
        }
        hidden_layer_output[h] = sigmoid(weighted_sum);
    }

    // 2. Calcular a camada de saída (APENAS 1 NEURÔNIO)
    double output_weighted_sum = 0;
    for (int h = 0; h < NUM_HIDDEN_NODES; h++) { // Para cada neurônio oculto 'h'
        output_weighted_sum += hidden_layer_output[h] * weights_output_hidden[h][0];
    }
    *out_final = sigmoid(output_weighted_sum);
}
