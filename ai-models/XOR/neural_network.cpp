#include "neural_network.h"

#include <stdexcept>
#include <utility>

std::vector<std::vector<neuron>> neural_network::layers;
std::vector<double> neural_network::last_input;

void neural_network::initialize() {
    layers.clear();
    last_input.clear();

    layers.push_back({neuron(-1.60705), neuron(2.88909), neuron(-2.28006)});
    layers.push_back({neuron(3.15755)});

    layers[0][0].add_weight(2.54206);
    layers[0][0].add_weight(-2.76984);
    layers[0][1].add_weight(5.62301);
    layers[0][1].add_weight(-5.66794);
    layers[0][2].add_weight(3.95944);
    layers[0][2].add_weight(-4.14377);
    layers[1][0].add_weight(3.57717);
    layers[1][0].add_weight(-7.62879);
    layers[1][0].add_weight(5.88081);
}

double neural_network::forward(const std::vector<double>& input) {
    if (layers.empty()) {
        throw std::logic_error("Network is not initialized");
    }
    if (input.size() != 2) {
        throw std::invalid_argument("XOR requires exactly two inputs");
    }

    last_input = input;
    std::vector<double> activations = input;
    for (auto& layer : layers) {
        std::vector<double> next_activations;
        for (auto& current_neuron : layer) {
            next_activations.push_back(
                current_neuron.calculate_activation(activations)
            );
        }
        activations = std::move(next_activations);
    }
    return activations.at(0);
}

void neural_network::backward(const double output_delta) {
    if (layers.empty() || last_input.empty()) {
        throw std::logic_error("Call forward before backward");
    }

    for (std::size_t i = layers.size(); i-- > 0;) {
        for (std::size_t j = 0; j < layers[i].size(); ++j) {
            double delta;
            if (i == layers.size() - 1) {
                delta = output_delta;
            } else {
                double sum = 0.0;
                for (const auto& next_neuron : layers[i + 1]) {
                    sum += next_neuron.get_weight(j) * next_neuron.get_delta();
                }
                const double activation = layers[i][j].get_activation();
                delta = sum * activation * (1.0 - activation);
            }
            layers[i][j].set_delta(delta);

            for (std::size_t k = 0; k < layers[i][j].get_weight_count(); ++k) {
                const double previous_activation = i == 0
                    ? last_input[k]
                    : layers[i - 1][k].get_activation();
                layers[i][j].set_gradient(k, previous_activation * delta);
            }
        }
    }
}

void neural_network::update_weights(const double learning_rate) {
    for (auto& layer : layers) {
        for (auto& current_neuron : layer) {
            current_neuron.update_weights(learning_rate);
        }
    }
}
