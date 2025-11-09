// Backpropagation module — use shared configuration from config_nn.c
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "config_nn.c"

void back_propagation(const double current_input[NUM_INPUT_NODES], double current_expected_output, double learning_rate, double weights_input_hidden[NUM_INPUT_NODES][NUM_HIDDEN_NODES], double weights_output_hidden[NUM_HIDDEN_NODES][NUM_OUTPUT_NODES], double hidden_layer_output[NUM_HIDDEN_NODES], double final_output) {
    
    // 1. Calcular o erro da camada de saída
    double output_error_gradient = sigmoid_derivative(final_output) * (current_expected_output - final_output);
    // 2. Calcular o erro da camada oculta
    double hidden_error_gradient[NUM_HIDDEN_NODES];
    for (int h = 0; h < NUM_HIDDEN_NODES; h++) {
        double weighted_error_sum = output_error_gradient * weights_output_hidden[h][0];
        hidden_error_gradient[h] = sigmoid_derivative(hidden_layer_output[h]) * weighted_error_sum;
    }
    
    // 3. Atualizar os pesos da camada oculta -> saída
    for (int h = 0; h < NUM_HIDDEN_NODES; h++) {
        double delta_weight = learning_rate * output_error_gradient * hidden_layer_output[h];
        weights_output_hidden[h][0] += delta_weight;
    }
    // 4. Atualizar os pesos da camada de entrada -> oculta
    for (int h = 0; h < NUM_HIDDEN_NODES; h++) { // Para cada neurônio oculto 'h'
        for (int j = 0; j < NUM_INPUT_NODES; j++) { // Para cada neurônio de entrada 'j'
            double delta_weight = learning_rate * hidden_error_gradient[h] * current_input[j];
            weights_input_hidden[j][h] += delta_weight;
        }
    }
}