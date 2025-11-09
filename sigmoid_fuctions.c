// Simple sigmoid utilities. Do NOT include config_nn.c here — keep this module standalone.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Função de ativação Sigmoid
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// A derivada da sigmoid, usada na backpropagation
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}