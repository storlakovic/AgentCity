#pragma once
#include "../layer/layer.hpp"

namespace neural_network {
    inline std::vector<layer> layers;
    inline std::vector<double> last_input;
    extern void initialize(const std::vector<std::size_t> &architecture);
    extern void update_weights(double learning_rate);
    extern double random_number();
    extern std::vector<double> forward(const std::vector<double>& input);
    extern void backward(std::vector<double> output_deltas);
    extern double sigmoid(double x);
    std::vector<double> softmax( const std::vector<double>& values);
};

