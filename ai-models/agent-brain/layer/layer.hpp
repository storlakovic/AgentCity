#pragma once
#include <vector>

class layer {
    std::vector<double> m_weights;
    std::vector<double> m_activations;
    std::vector<double> m_gradients;
    std::vector<double> m_biases;
    std::vector<double> m_deltas;
    size_t m_input_count;
    size_t m_neuron_count;

    public:
    explicit layer(size_t input_count, size_t neuron_count);
    void init_activations(unsigned activation_count);
    void init_weights(unsigned weight_count);
    void init_biases(unsigned bias_count);
    void init_gradients(unsigned gradient_count);
    void init_deltas(unsigned deltas_count);
    std::vector<double> get_weights();
    std::vector<double> get_activations();
    void set_gradients(const std::vector<double>& gradients);
    void set_deltas(const std::vector<double>& deltas);
    [[nodiscard]] size_t get_input_count() const;
    void set_activation(unsigned index, double activation);
    void set_gradient_at(unsigned index, double gradient);
    double get_delta_at(unsigned index) const;
    void set_delta_at(unsigned index, double delta);
    void set_weight_at(unsigned index, double weight);
    void set_bias_at(unsigned index, double bias);
    double get_gradient_at(unsigned index) const;
    [[nodiscard]] size_t get_neuron_count() const;
    [[nodiscard]] double get_weight_at(unsigned index) const;
    [[nodiscard]] double get_bias_at(unsigned index) const;
    double get_activation_at(unsigned index) const;
};

