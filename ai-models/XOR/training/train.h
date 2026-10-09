#ifndef NEURAL_NETWORK_TRAIN_H
#define NEURAL_NETWORK_TRAIN_H

#include <cstddef>
#include <vector>

class train {
public:
    static void train_model(
        const std::vector<std::vector<double>>& inputs,
        const std::vector<double>& expected,
        double learning_rate,
        std::size_t max_epochs = 10000,
        double target_loss = 0.0025
    );
};

#endif
