#ifndef NEURAL_NETWORK_NEURAL_NETWORK_H
#define NEURAL_NETWORK_NEURAL_NETWORK_H

#include <vector>
#include "neuron/neuron.h"

namespace neural_network {
    extern std::vector<std::vector<neuron>> layers;
    extern std::vector<double> last_input;

    void initialize();
    double forward(const std::vector<double>& input);
    void backward(double output_delta);
    void update_weights(double learning_rate);
}

#endif
