#include <cstdlib>

#include "brain/neural_network.hpp"
#include "training/train.hpp"

int main() {
    std::vector<double> input = {0,1,2};
    std::vector<double> deltas = {0,1,2,3,4};
    const std::vector<std::size_t> architecture = {
        3,
        32,
        16,
        5
    };
    neural_network::initialize(architecture);
    train::train_agent(5000, 0.1);
    std::vector<double> output = neural_network::forward(input);

    return EXIT_SUCCESS;
}
