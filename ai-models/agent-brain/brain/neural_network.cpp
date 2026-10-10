#include "neural_network.hpp"

#include <algorithm>
#include <random>
#include <cmath>

void neural_network::initialize(const std::vector<std::size_t> &architecture) {
    for (std::size_t i = 1; i < architecture.size(); i++) {
        layer currentLayer(architecture[i - 1], architecture[i]);

        currentLayer.init_activations(architecture[i]);
        currentLayer.init_biases(architecture[i]);
        currentLayer.init_weights(architecture[i] * architecture[i - 1]);
        currentLayer.init_gradients(architecture[i] * architecture[i - 1]);
        currentLayer.init_deltas(architecture[i]);

        layers.push_back(std::move(currentLayer));
    }
}

std::vector<double> neural_network::forward(const std::vector<double>& input) {
    std::vector<double> previous_activations = input;
    last_input = input;
    double sum;
    for (auto& layer : layers) {
        bool is_output_layer = &layer == &layers.back();
        std::vector<double> output_values;
        for (std::size_t i = 0; i < layer.get_neuron_count(); i++) {
            sum = layer.get_bias_at(i);
            for (std::size_t j = 0; j < previous_activations.size(); j++) {
                const std::size_t weight_index = j + (i * layer.get_input_count());
                sum += previous_activations.at(j) * layer.get_weight_at(weight_index);
            }
            if (is_output_layer) {
                output_values.push_back(sum);
            } else {
                layer.set_activation(i,sigmoid(sum));
            }
        }
        if (is_output_layer) {
            std::vector<double> probabilities = softmax(output_values);
            for (std::size_t i = 0; i < probabilities.size(); i++) {
                layer.set_activation(i, probabilities[i]);
            }
        }
        previous_activations = layer.get_activations();
    }

    return layers.back().get_activations();
}

void neural_network::backward(std::vector<double> output_deltas) {
    layer* output_layer = &layers.back();
    output_layer->set_deltas(output_deltas);

    for (int x = 0; x < output_layer->get_neuron_count(); x++) {
        for (int y = 0; y < output_layer->get_input_count(); y++) {
            double gradient = layers[layers.size()-2].get_activation_at(y) * output_deltas.at(x);
            output_layer->set_gradient_at(y + x*output_layer->get_input_count(), gradient);
        }
    }

    for (int layer_index = layers.size() - 2; layer_index >= 0; layer_index --) {
        layer& current_layer = layers[layer_index];
        layer& next_layer = layers[layer_index + 1];
        for (std::size_t neuron = 0; neuron < current_layer.get_neuron_count(); neuron++) {
            double sum = 0.0;
            for (std::size_t next_neuron = 0; next_neuron < next_layer.get_neuron_count(); next_neuron++) {
                std::size_t weight_index = neuron + next_neuron * next_layer.get_input_count();
                sum += next_layer.get_weight_at(weight_index) * next_layer.get_delta_at(next_neuron);
            }
            double activation = current_layer.get_activation_at(neuron);
            double delta = sum * activation * (1.0 - activation);
            current_layer.set_delta_at(neuron, delta);

            for (std::size_t input_index = 0; input_index < current_layer.get_input_count(); input_index++) {
                double previous_activation;

                if (layer_index == 0) {
                    previous_activation = last_input.at(input_index);
                } else {
                    previous_activation = layers[layer_index - 1].get_activation_at(input_index);
                }

                double gradient = previous_activation * delta;

                std::size_t gradient_index = input_index + neuron * current_layer.get_input_count();

                current_layer.set_gradient_at(gradient_index, gradient);
           }
        }
    }
}

void neural_network::update_weights(double learning_rate) {
    for (auto& current_layer : layers) {
        for (std::size_t neuron = 0; neuron < current_layer.get_neuron_count(); neuron++) {
            for (std::size_t input = 0; input < current_layer.get_input_count(); input++) {
                std::size_t index = input + neuron * current_layer.get_input_count();
                double new_weight = current_layer.get_weight_at(index) - learning_rate * current_layer.get_gradient_at(index);
                current_layer.set_weight_at(index, new_weight);
            }

            double new_bias = current_layer.get_bias_at(neuron) - learning_rate * current_layer.get_delta_at(neuron);
            current_layer.set_bias_at(neuron, new_bias);
        }
    }
}

double neural_network::random_number() {
    static std::mt19937 generator(
        std::random_device{}()
    );

    static std::uniform_real_distribution<double> distribution(
        -1.0,
        1.0
    );

    return distribution(generator);
}

std::vector<double> neural_network::softmax( const std::vector<double>& values) {
    std::vector<double> probabilities(values.size());

    double max_value = *std::max_element(values.begin(), values.end());
    double sum = 0.0;

    for (std::size_t i = 0; i < values.size(); i++) {
        probabilities[i] = std::exp(values[i] - max_value);
        sum += probabilities[i];
    }

    for (double& probability : probabilities) {
        probability /= sum;
    }

    return probabilities;
}

double neural_network::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}
