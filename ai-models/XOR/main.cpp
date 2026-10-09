#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <vector>

#include "neural_network.h"
#include "training/train.h"

int main() try {
    constexpr double learning_rate = 0.1;
    constexpr std::size_t max_epochs = 10000;
    constexpr double target_loss = 0.0025;
    constexpr double classification_threshold = 0.5;

    const std::vector<std::vector<double>> inputs = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    const std::vector<double> expected = {0, 1, 1, 0};

    neural_network::initialize();
    //train::train_model(inputs, expected, learning_rate, max_epochs, target_loss);

    std::cout << "\nXOR predictions\n" << std::fixed << std::setprecision(4);

    std::size_t correct_predictions = 0;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const double output = neural_network::forward(inputs[i]);
        const int predicted = output >= classification_threshold ? 1 : 0;
        const int target = static_cast<int>(expected[i]);
        correct_predictions += predicted == target ? 1 : 0;
        std::cout << static_cast<int>(inputs[i][0]) << " XOR "
                  << static_cast<int>(inputs[i][1]) << " = " << predicted
                  << " (output: " << output << ", expected: " << target << ")\n";
    }

    std::cout << "Correct: " << correct_predictions << '/' << inputs.size() << '\n';
    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    std::cerr << "XOR example failed: " << error.what() << '\n';
    return EXIT_FAILURE;
}
