#pragma once
#include <string>
#include <vector>
#include "../environment/action.hpp"

namespace train {
    extern void train_agent(int epochs, double learning_rate);
    action sample_action(std::vector<double>& probabilities);
    std::string action_to_string(std::size_t action_index);
};

