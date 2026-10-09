# XOR AI

A small neural network implemented in C++20 that models the XOR function:
the result is 1 when two binary inputs differ, and 0 otherwise.

## Architecture and Training

- Architecture: 2 inputs, 3 hidden neurons, and 1 output neuron.
- Activation function: sigmoid in both the hidden and output layers.
- Training: backpropagation with a gradient descent update after each sample.
- Loss function: mean squared error (MSE) across all four XOR cases.
- Configured training parameters in `main.cpp`: learning rate 0.1, up to
  10,000 epochs, and a target MSE of 0.0025.

The trained weights and biases are stored in `neural_network.cpp`.
The training call is currently commented out, so the program evaluates the
existing model. The original training duration and loss history are not
documented.

## Trained Model Results

Verified using a current build. Outputs of 0.5 or higher are classified as 1.

| Input A | Input B | Expected | Network Output | 
| --- | --- | --- | --- |
| 0 | 0 | 0 | 0.0509 |
| 0 | 1 | 1 | 0.9408 |
| 1 | 0 | 1 | 0.9550 |
| 1 | 1 | 0 | 0.0432 |

**Result: 4 out of 4 cases correct (100% on the XOR truth table).**
g.
