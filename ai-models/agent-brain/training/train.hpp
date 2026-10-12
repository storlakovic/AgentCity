#pragma once
#include <cstddef>
#include <string>
#include <vector>
#include "../environment/action.hpp"

class environment;
class experience;

namespace train {

    struct episode_result {
        double reward = 0.0;
        bool survived = false;
    };

    void train_agent();
    double evaluate_network(environment& env);
    action sample_action(const std::vector<double>& probabilities);
    std::string action_to_string(std::size_t action_index);
    episode_result collect_experiences(std::vector<experience>& trajectory, environment& env);
    void update_network(std::vector<experience>& trajectory);
};
