#include "neural_network.hpp"

#include <algorithm>
#include <random>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

void neural_network::save(const std::string& path) {
    if (layers.empty()) {
        throw std::runtime_error("Cannot save an uninitialized neural network");
    }
    nlohmann::json model = {
        {"format_version", 1},
        {"architecture", {layers.front().get_input_count()}},
        {"hidden_activation", "sigmoid"},
        {"output_activation", "softmax"},
        {"layers", nlohmann::json::array()}
    };
    for (auto& current_layer : layers) {
        auto weights = current_layer.get_weights();
        std::vector<double> biases;
        for (std::size_t i = 0; i < current_layer.get_neuron_count(); ++i) {
            biases.push_back(current_layer.get_bias_at(i));
        }

        model["architecture"].push_back(current_layer.get_neuron_count());
        model["layers"].push_back({{"weights", weights}, {"biases", biases}});
    }
    const std::string json = model.dump(2);

    const std::filesystem::path temporary_path = path + ".tmp";
    try {
        std::ofstream file;
        file.exceptions(std::ios::failbit | std::ios::badbit);
        file.open(temporary_path, std::ios::binary | std::ios::trunc);
        file << json;
        file.close();
        std::filesystem::rename(temporary_path, path);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary_path, ignored);
        throw;
    }
}

void neural_network::load(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open model: " + path);
    const auto model = nlohmann::json::parse(file);

    const auto& architecture = model.at("architecture");
    const auto& saved_layers = model.at("layers");
    std::vector<std::size_t> sizes;
    for (const auto& size : architecture) {
        sizes.push_back(size.get<std::size_t>());
    }

    std::vector<layer> loaded_layers;
    for (std::size_t i = 0; i < saved_layers.size(); ++i) {
        const auto inputs = sizes[i], neurons = sizes[i + 1];
        const auto& weights = saved_layers[i].at("weights");
        const auto& biases = saved_layers[i].at("biases");

        auto parameter = [](const nlohmann::json& value) {
            return value.get<double>();
        };

        layer current_layer(inputs, neurons);

        current_layer.init_activations(neurons);
        current_layer.init_biases(neurons);
        current_layer.init_weights(inputs * neurons);
        current_layer.init_gradients(inputs * neurons);
        current_layer.init_deltas(neurons);

        for (std::size_t j = 0; j < weights.size(); ++j) {
            current_layer.set_weight_at(j, parameter(weights[j]));
        }
        for (std::size_t j = 0; j < biases.size(); ++j) {
            current_layer.set_bias_at(j, parameter(biases[j]));
        }

        loaded_layers.push_back(std::move(current_layer));
    }

    layers = std::move(loaded_layers);
    last_input.clear();
}

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
                layer.set_activation(i,sum);
            } else {
                layer.set_activation(i,sigmoid(sum));
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

double neural_network::random_number(double min, double max) {
    static std::mt19937 generator(
        std::random_device{}()
    );

    std::uniform_real_distribution<double> distribution(
        min,
        max
    );

    return distribution(generator);
}

std::vector<double> neural_network::softmax(const std::vector<double>& values) {
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
