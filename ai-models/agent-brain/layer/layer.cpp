#include "layer.hpp"

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

    for (double& weight : m_weights) {
        weight = neural_network::random_number();
    }
}

void layer::init_biases(const unsigned bias_count) {
    m_biases.resize(bias_count, 0.0);
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
