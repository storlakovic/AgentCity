#include "neuron.h"

#include <cmath>
#include <stdexcept>

double neuron::sigmoid(const double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

neuron::neuron(const double bias) : m_bias(bias) {}

double neuron::calculate_activation(const std::vector<double>& previous_activations) {
    if (previous_activations.size() != m_weights.size()) {
        throw std::invalid_argument("Input and weight counts do not match");
    }

    m_activation = m_bias;
    for (std::size_t i = 0; i < m_weights.size(); ++i) {
        m_activation += previous_activations[i] * m_weights[i];
    }
    m_activation = sigmoid(m_activation);
    return m_activation;
}

void neuron::add_weight(const double weight) {
    m_weights.push_back(weight);
    m_gradients.push_back(0.0);
}

void neuron::update_weights(const double learning_rate) {
    for (std::size_t i = 0; i < m_weights.size(); ++i) {
        m_weights[i] -= learning_rate * m_gradients[i];
    }
    m_bias -= learning_rate * m_delta;
}

void neuron::set_gradient(const std::size_t index, const double gradient) {
    m_gradients[index] = gradient;
}

double neuron::get_gradient(const std::size_t index) const {
    return m_gradients[index];
}

void neuron::set_weight(const std::size_t index, const double weight) {
    m_weights[index] = weight;
}

double neuron::get_weight(const std::size_t index) const {
    return m_weights[index];
}

void neuron::set_delta(const double delta) {
    m_delta = delta;
}

double neuron::get_delta() const {
    return m_delta;
}

double neuron::get_activation() const {
    return m_activation;
}

std::size_t neuron::get_weight_count() const {
    return m_weights.size();
}
