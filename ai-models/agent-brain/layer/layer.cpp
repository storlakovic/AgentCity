#include "layer.hpp"
#include <cmath>

#include "../brain/neural_network.hpp"


layer::layer(const size_t input_count, const size_t neuron_count) {
    m_input_count = input_count;
    m_neuron_count = neuron_count;
}

void layer::init_activations(const unsigned activation_count) {
    m_activations.resize(activation_count, 0.0);
}

void layer::init_weights(const unsigned weight_count) {
    m_weights.resize(weight_count);
    m_weight_mean.assign(weight_count, 0.0);
    m_weight_square.assign(weight_count, 0.0);
    m_adam_step = 0;

    for (double& weight : m_weights) {
        weight = neural_network::random_number();
    }
}

void layer::init_biases(const unsigned bias_count) {
    m_biases.assign(bias_count, 0.0);
    m_bias_mean.assign(bias_count, 0.0);
    m_bias_square.assign(bias_count, 0.0);
    m_adam_step = 0;
}

void layer::init_gradients(const unsigned gradient_count) {
    m_gradients.resize(gradient_count, 0.0);
}

void layer::init_deltas(unsigned deltas_count) {
    m_deltas.resize(deltas_count, 0.0);
}

std::vector<double> layer::get_weights() {
    return m_weights;
}

std::vector<double> layer::get_activations() {
    return m_activations;
}

void layer::set_gradients(const std::vector<double>& gradients) {
    m_gradients = gradients;
}

void layer::set_deltas(const std::vector<double> &deltas) {
    m_deltas = deltas;
}

size_t layer::get_input_count() const {
    return m_input_count;
}

void layer::set_activation(const unsigned index, const double activation) {
    m_activations[index] = activation;
}

void layer::set_gradient_at(unsigned index, double gradient) {
    m_gradients[index] = gradient;
}

double layer::get_delta_at(const unsigned index) const {
    return m_deltas.at(index);
}

void layer::set_delta_at(const unsigned index, const double delta) {
    m_deltas[index] = delta;
}

void layer::set_weight_at(unsigned index, double weight) {
    m_weights[index] = weight;
}

void layer::set_bias_at(unsigned index, double bias) {
    m_biases[index] = bias;
}

double layer::get_gradient_at(unsigned index) const {
    return m_gradients.at(index);
}

size_t layer::get_neuron_count() const {
    return m_neuron_count;
}

double layer::get_weight_at(const unsigned index) const {
    return m_weights.at(index);
}

double layer::get_bias_at(const unsigned index) const {
    return m_biases.at(index);
}

double layer::get_activation_at(const unsigned index) const {
    return m_activations.at(index);
}

void layer::apply_gradients(double learning_rate) {
    constexpr double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
    ++m_adam_step;
    const double correction1 = 1.0 - std::pow(beta1, m_adam_step);
    const double correction2 = 1.0 - std::pow(beta2, m_adam_step);
    const auto update = [&](double& parameter, double gradient,
                            double& mean, double& square) {
        mean = beta1 * mean + (1.0 - beta1) * gradient;
        square = beta2 * square + (1.0 - beta2) * gradient * gradient;
        parameter -= learning_rate * (mean / correction1)
            / (std::sqrt(square / correction2) + epsilon);
    };
    for (size_t i = 0; i < m_weights.size(); ++i) {
        update(m_weights[i], m_gradients[i], m_weight_mean[i], m_weight_square[i]);
    }
    for (size_t i = 0; i < m_biases.size(); ++i) {
        update(m_biases[i], m_deltas[i], m_bias_mean[i], m_bias_square[i]);
    }
}