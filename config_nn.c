// Configuration header-style file (included by modules). Defines macros and externs.
#ifndef CONFIG_NN_C_INCLUDED
#define CONFIG_NN_C_INCLUDED

#include <stddef.h>

// Network dimensions
#define NUM_INPUT_NODES   2
#define NUM_HIDDEN_NODES  2
#define NUM_OUTPUT_NODES  1
#define NUM_TRAINING_SETS 4

// Globals (defined in one C file - nn_basic.c)
extern double weights_input_hidden[NUM_INPUT_NODES][NUM_HIDDEN_NODES];
extern double weights_output_hidden[NUM_HIDDEN_NODES][NUM_OUTPUT_NODES];
extern double xor_inputs[NUM_TRAINING_SETS][NUM_INPUT_NODES];
extern double xor_expected[NUM_TRAINING_SETS][NUM_OUTPUT_NODES];
extern double hidden_layer_output[NUM_HIDDEN_NODES];
extern double final_output;
extern double learning_rate;
extern long epochs;

// Utility prototypes (implemented in sigmoid_fuctions.c)
double sigmoid(double x);
double sigmoid_derivative(double x);

// Initialization function to be implemented in one translation unit
void nn_config(void);

#endif // CONFIG_NN_C_INCLUDED