#pragma once
#include <string>
#include "../layer/layer.hpp"

namespace neural_network {
    // Writes architecture, weights and biases as JSON; throws on failure.
    void save(const std::string& path);
    // Replaces the network after validation; throws without changing it on failure.
    void load(const std::string& path);
    inline std::vector<layer> layers;
    inline std::vector<double> last_input;
    extern void initialize(const std::vector<std::size_t> &architecture);
    extern void update_weights(double learning_rate);
    extern double random_number();
    extern double random_number(double min, double max);
    extern std::vector<double> forward(const std::vector<double>& input);
    extern void backward(std::vector<double> output_deltas);
    extern double sigmoid(double x);
    std::vector<double> softmax( const std::vector<double>& values);
};
