#include "train.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "../neural_network.h"

void train::train_model(
    const std::vector<std::vector<double>>& inputs,
    const std::vector<double>& expected,
    const double learning_rate,
    const std::size_t max_epochs,
    const double target_loss
) {
    if (inputs.empty() || inputs.size() != expected.size()) {
        throw std::invalid_argument("Inputs and targets must match");
    }
    if (!std::isfinite(learning_rate) || learning_rate <= 0.0 ||
        !std::isfinite(target_loss) || target_loss < 0.0 ||
        max_epochs == 0) {
        throw std::invalid_argument("Invalid training configuration");
    }

    for (const auto& input : inputs) {
        if (input.size() != 2) {
            throw std::invalid_argument("XOR requires two inputs per sample");
        }
        for (const double value : input) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("Inputs must be finite");
            }
        }
    }
    for (const double target : expected) {
        if (!std::isfinite(target) || target < 0.0 || target > 1.0) {
            throw std::invalid_argument("Targets must be between zero and one");
        }
    }

    for (std::size_t epoch = 0; epoch < max_epochs; ++epoch) {
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            const double prediction = neural_network::forward(inputs[i]);
            const double error = prediction - expected[i];
            const double delta =
                2.0 * error * prediction * (1.0 - prediction);

            neural_network::backward(delta);
            neural_network::update_weights(learning_rate);
        }

        double mean_loss = 0.0;
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            const double error =
                neural_network::forward(inputs[i]) - expected[i];
            mean_loss += error * error;
        }
        mean_loss /= static_cast<double>(inputs.size());

        if (epoch % 100 == 0 || mean_loss <= target_loss ||
            epoch + 1 == max_epochs) {
            std::cout << "Epoch " << epoch + 1
                      << ", MSE: " << mean_loss << '\n';
        }
        if (mean_loss <= target_loss) {
            return;
        }
    }
}
