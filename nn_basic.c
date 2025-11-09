#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Função de ativação Sigmoid
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// A derivada da sigmoid, usada na backpropagation
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

int main () {
    // --- 1. Definição da Arquitetura e Inicialização dos Pesos ---
    srand(105); // Semente para números aleatórios para que os pesos sejam diferentes a cada execução

    double learning_rate = 0.1;
    long epochs = 1000000;

    // Pesos são criados UMA VEZ aqui e inicializados com valores aleatórios pequenos
    double weights_input_hidden[2][2];
    double weights_output_hidden[2][1];
    
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            weights_input_hidden[i][j] = ((double)rand() / RAND_MAX);
        }
    }
    for (int i = 0; i < 2; i++) {
        weights_output_hidden[i][0] = ((double)rand() / RAND_MAX);
    }

    // Dataset XOR
    double xor_inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    double xor_expected[4][1] = {{0}, {1}, {1}, {0}};

    // --- 2. Loop de Treinamento Principal ---
    for (int e = 0; e < epochs; e++) {
        
        // Loop que passa por cada um dos 4 exemplos do dataset
        for (int i = 0; i < 4; i++) {
            
            double current_input[2] = {xor_inputs[i][0], xor_inputs[i][1]};
            double current_expected_output = xor_expected[i][0];

            // =========================================================================
            // FORWARD PASS
            // =========================================================================

            // 1. Calcular a camada oculta
            double hidden_layer_output[2];
            for (int h = 0; h < 2; h++) { // Para cada neurônio oculto 'h'
                double weighted_sum = 0;
                for (int j = 0; j < 2; j++) { // Para cada neurônio de entrada 'j'
                     weighted_sum += current_input[j] * weights_input_hidden[j][h];
                }
                hidden_layer_output[h] = sigmoid(weighted_sum);
            }

            // 2. Calcular a camada de saída (APENAS 1 NEURÔNIO)
            double final_output;
            double output_weighted_sum = 0;
            for (int h = 0; h < 2; h++) { // Para cada neurônio oculto 'h'
                output_weighted_sum += hidden_layer_output[h] * weights_output_hidden[h][0];
            }
            final_output = sigmoid(output_weighted_sum);

            // =========================================================================
            // BACKPROPAGATION
            // =========================================================================

            // 1. Calcular o erro da camada de saída
            double output_error_gradient = sigmoid_derivative(final_output) * (current_expected_output - final_output);

            // 2. Calcular o erro da camada oculta
            double hidden_error_gradient[2];
            for (int h = 0; h < 2; h++) {
                double weighted_error_sum = output_error_gradient * weights_output_hidden[h][0];
                hidden_error_gradient[h] = sigmoid_derivative(hidden_layer_output[h]) * weighted_error_sum;
            }
            
            // 3. Atualizar os pesos da camada oculta -> saída
            for (int h = 0; h < 2; h++) {
                double delta_weight = learning_rate * output_error_gradient * hidden_layer_output[h];
                weights_output_hidden[h][0] += delta_weight;
            }

            // 4. Atualizar os pesos da camada de entrada -> oculta
            for (int h = 0; h < 2; h++) { // Para cada neurônio oculto 'h'
                for (int j = 0; j < 2; j++) { // Para cada neurônio de entrada 'j'
                    double delta_weight = learning_rate * hidden_error_gradient[h] * current_input[j];
                    weights_input_hidden[j][h] += delta_weight;
                }
            }
        }
    }

    // --- 3. Teste Final (DEPOIS de todo o treinamento) ---
    printf("\n--- Testando a rede treinada ---\n");
    for (int i = 0; i < 4; i++) {
        double current_input[2] = {xor_inputs[i][0], xor_inputs[i][1]};
        
        // Forward pass (sem backprop) para obter a previsão
        double hidden_layer_output[2];
        for (int h = 0; h < 2; h++) {
            double weighted_sum = 0;
            for (int j = 0; j < 2; j++) {
                 weighted_sum += current_input[j] * weights_input_hidden[j][h];
            }
            hidden_layer_output[h] = sigmoid(weighted_sum);
        }

        double final_output;
        double output_weighted_sum = 0;
        for (int h = 0; h < 2; h++) {
            output_weighted_sum += hidden_layer_output[h] * weights_output_hidden[h][0];
        }
        final_output = sigmoid(output_weighted_sum);
        // Depois de obter a 'final_output'
        int final_prediction;
        if (final_output > 0.98) {
            final_prediction = 1;
        } else {
            final_prediction = 0;
        }
        printf("Entrada: {%.0f, %.0f} -> Saida Esperada: %.0f, Previsao da Rede: %f, Precisao de 98 ou mais: %d\n",
               current_input[0], current_input[1], xor_expected[i][0], final_output, final_prediction);
    }

    return 0;
}