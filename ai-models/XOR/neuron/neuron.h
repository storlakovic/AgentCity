#ifndef NEURAL_NETWORK_NODE_H
#define NEURAL_NETWORK_NODE_H

#include <cstddef>
#include <vector>

class neuron {
    double m_bias = 0.0;
    double m_delta = 0.0;
    std::vector<double> m_gradients;
    std::vector<double> m_weights;
    double m_activation = 0.0;

    static double sigmoid(double x);

public:
    explicit neuron(double bias);

    [[nodiscard]] double calculate_activation(const std::vector<double>& previous_activations);
    void add_weight(double weight);
    void update_weights(double learning_rate);
    void set_gradient(std::size_t index, double gradient);
    [[nodiscard]] double get_gradient(std::size_t index) const;
    void set_weight(std::size_t index, double weight);
    [[nodiscard]] double get_weight(std::size_t index) const;
    void set_delta(double delta);
    [[nodiscard]] double get_delta() const;
    [[nodiscard]] double get_activation() const;
    [[nodiscard]] std::size_t get_weight_count() const;
};

#endif
