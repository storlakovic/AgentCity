#pragma once
#include <cstddef>
#include <string>
#include <vector>
#include "../environment/action.hpp"
#include "../environment/state.hpp"

class environment;
class experience;

namespace train {
    struct wellbeing {
        double health;
        double happiness;
        double financial_stability;
        double social_wellbeing;
    };

    struct episode_result {
        double reward = 0.0;
        bool survived = false;
        std::vector<wellbeing> wellbeings;
        std::vector<action> actions;
        std::vector<state> states;
    };

    void train_agent();
    double evaluate_network(environment& env);
    action sample_action(const std::vector<double>& probabilities);
    wellbeing calculate_wellbeing(const state& s);
    std::string action_to_string(std::size_t action_index);
    episode_result collect_experiences(std::vector<experience>& trajectory, environment& env);
    void update_network(std::vector<experience>& trajectory);
    double calculate_wellbeing_score(const wellbeing& w);

};
