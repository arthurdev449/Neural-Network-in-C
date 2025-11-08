#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

int main () {
    // pesos do input e camada oculta
    double weights_input_hidden[2][2] = {{7.0, 5.0}, 
                                         {7.0, 5.0}};
    
    // pesos do output e camada oculta
    double weights_output_hidden[2][2] = {{1.0, 3.0}, 
                                          {1.0, 3.0}};

    int input[2] = {1.0, 0.0};
    
    // --- Forward Pass ---

    double hidden_layer_output[2];
    // Para cada neurônio oculto
    for (int i = 0; i < 2; i++) {
        double weighted_sum = 0;
        for (int j = 0; j < 2; j++) {
             weighted_sum += input[j] * weights_input_hidden[j][i];
        }
        hidden_layer_output[i] = sigmoid(weighted_sum);
        printf("Saida do Neuronio Oculto %d: %f\n", i, hidden_layer_output[i]);
    }

    // --- Estação 2: Calcular a camada de saída ---
    double final_outputs[2];

    // Para cada neurônio de saída
    for (int k = 0; k < 2; k++) { 
        double output_weighted_sum = 0;
        for (int i = 0; i < 2; i++) {
            output_weighted_sum += hidden_layer_output[i] * weights_output_hidden[i][k];
        }

        final_outputs[k] = sigmoid(output_weighted_sum);
        printf("Saida do Neuronio Final %d: %f\n", k, final_outputs[k]);
    }

    // back propagation
    double expected_output[2] = {1.0, 0.0}; 
     double output_error_gradient[2];

    for (int i = 0; i < 2; i++) {
        // TODO: Implemente a fórmula do gradiente de erro aqui para cada neurônio de saída 'i'.
        // Use a variável final_outputs[i] e expected_output[i]
        // output_error_gradient[i] = ...
        double gradient_output_error = final_outputs[i] * (1 - final_outputs[i]) * (expected_output - final_outputs[i]);
        printf("Gradiente de Erro do Neurônio de Saída %d: %f\n", i, gradient_output_error[i]);
    }
    
    return 0;
}
