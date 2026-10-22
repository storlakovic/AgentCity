#include "train.hpp"

#include <algorithm>
#include <random>
#include <iostream>

#include "../brain/neural_network.hpp"

#include <vector>

#include "config.hpp"
#include "../environment/environment.hpp"
#include "experience.hpp"
#include "nlohmann/json.hpp"

void train::train_agent() {
    environment env(state(0.5, 0.5, 1.0, 0.5, 0.2, 1.0, location::HOME));
    double best_avg_reward = evaluate_network(env);
    std::cout << "Initial evaluation | Avg reward: " << best_avg_reward << '\n';

    for (int epoch = 0; epoch < config::epochs; epoch++) {
        std::vector<experience> trajectory;
        auto result = collect_experiences(trajectory, env);
        update_network(trajectory);
        if ((epoch + 1) % config::log_interval == 0) {
            std::cout << "Episode " << epoch + 1
                      << " | Reward: " << result.reward
                      << " | Survived: " << (result.survived ? "yes" : "no")
                      << '\n';

            for (std::size_t i = 0; i < result.states.size(); i++) {
                std::cout << "State " << i << ": ";

                for (double value : result.states[i].to_vector()) {
                    std::cout << value << ' ';
                }

                if (i < result.actions.size()) {
                    std::cout << "| Action: "
                              << action_to_string(
                                     static_cast<std::size_t>(result.actions[i]));
                }

                std::cout << '\n';
            }
        }
        if ((epoch + 1) % config::evaluation_interval == 0) {
            const double avg_reward = evaluate_network(env);
            std::cout << "Evaluation " << epoch + 1
                      << " | Avg reward: " << avg_reward << '\n';
            if (avg_reward > best_avg_reward) {
                neural_network::save(config::checkpoint_path);
                best_avg_reward = avg_reward;
                std::cout << "Saved best model: " << config::checkpoint_path
                          << " | Avg reward: " << best_avg_reward << '\n';
            }
        }
    }
}

train::episode_result train::collect_experiences(std::vector<experience>& trajectory, environment& env) {
    state current_state = env.reset();
    bool done = false;
    double total_reward = 0.0;
    std::vector<state> states;
    std::vector<action> selected_actions;
    std::vector<wellbeing> wellbeings;
    for (int step = 0; step < config::max_steps && !done; step++) {
        auto mask = env.get_action_mask(current_state);
        std::vector<double> output_values = neural_network::forward(current_state.to_vector());
        for (std::size_t i = 0; i < output_values.size(); i++) {
            if (!mask[i]) {
                output_values[i] = -1e9;
            }
        }
        std::vector<double> probabilities = neural_network::softmax(output_values);

        action selected_action = sample_action(probabilities);

        states.push_back(current_state);
        selected_actions.push_back(selected_action);
        step_result result = env.step(selected_action);

        wellbeing wellbeing = result.wellbeing;
        wellbeings.push_back(wellbeing);

        trajectory.push_back({
            current_state.to_vector(), static_cast<std::size_t>(selected_action), wellbeing, 0.0, mask, result.reward
        });

        total_reward += result.reward;

        current_state = result.new_state;
        done = result.done;
    }
    return {total_reward, !done, wellbeings, selected_actions, states};
}

void train::update_network(std::vector<experience>& trajectory) {
    double return_value = 0.0;

    for (int i = static_cast<int>(trajectory.size()) - 1; i >= 0; i--) {
        return_value = trajectory[i].get_reward() + config::gamma * return_value;
        trajectory[i].set_return_value(return_value);
    }

    for (auto& exp : trajectory) {
        std::vector<double> output_values = neural_network::forward(exp.get_state());
        const auto mask = exp.get_mask();
        for (std::size_t i = 0; i < output_values.size(); i++) {
            if (!mask[i]) {
                output_values[i] = -1e9;
            }
        }
        std::vector<double> probabilities = neural_network::softmax(output_values);
        std::vector<double> output_deltas(probabilities.size());

        for (std::size_t output = 0; output < probabilities.size(); output++) {
            if (!mask[output]) {
                continue;
            }
            double target = output == exp.get_action() ? 1.0 : 0.0;
            output_deltas[output] = (probabilities[output] - target) * exp.get_return_value();
        }
        neural_network::backward(output_deltas);
        neural_network::update_weights(config::learning_rate);
    }
}

double train::evaluate_network(environment& env) {
    double total_reward = 0.0;
    for (int i = 0; i < config::evaluation_episodes; i++) {
        state test_state = env.reset();
        for (int step = 0; step < config::max_steps; step++) {
            auto mask = env.get_action_mask(test_state);
            std::vector<double> output_values = neural_network::forward(test_state.to_vector());
            for (std::size_t i = 0; i < output_values.size(); i++) {
                if (!mask[i]) {
                    output_values[i] = -1e9;
                }
            }
            std::vector<double> probabilities = neural_network::softmax(output_values);
            step_result result = env.step(sample_action(probabilities));
            total_reward += result.reward;
            test_state = result.new_state;
            if (result.done) {
                break;
            }
        }
    }
    return total_reward / config::evaluation_episodes;
}

std::string train::action_to_string(std::size_t action_index) {
    switch (static_cast<action>(action_index)) {
        case action::EAT:
            return "EAT";
        case action::WORK:
            return "WORK";
        case action::SLEEP:
            return "SLEEP";
        case action::STEAL:
            return "STEAL";
        case action::SPEND_MONEY:
            return "SPEND_MONEY";
        case action::BUY_FOOD:
            return "BUY_FOOD";
        case action::GO_HOME:
            return "GO_HOME";
        case action::GO_TO_WORKPLACE:
            return "GO_TO_WORKPLACE";
        case action::GO_TO_SHOP:
            return "GO_TO_SHOP";
    }

    return "UNKNOWN";
}

train::wellbeing train::calculate_wellbeing(const state& s) {
    wellbeing w;

    w.health =
        0.5 * s.get_energy()
        + 0.5 * (1.0 - s.get_hunger());

    w.financial_stability =
        std::clamp(s.get_money(), 0.0, 1.0);

    w.happiness =
        0.5 * w.health
        + 0.1 * w.financial_stability
        + 0.4 * s.get_moral();

    w.social_wellbeing = 0.7 * w.health + 0.3 * w.financial_stability;

    return w;
}

double train::calculate_wellbeing_score(const wellbeing& w) {
    constexpr double HEALTH_WEIGHT = 0.50;
    constexpr double HAPPINESS_WEIGHT = 0.35;
    constexpr double FINANCIAL_WEIGHT = 0.15;

    return
        HEALTH_WEIGHT * w.health +
        HAPPINESS_WEIGHT * w.happiness +
        FINANCIAL_WEIGHT * w.financial_stability;
}

action train::sample_action(const std::vector<double>& probabilities) {
    static std::mt19937 generator(std::random_device{}());

    std::discrete_distribution<std::size_t> distribution(probabilities.begin(), probabilities.end());

    std::size_t index = distribution(generator);

    return static_cast<action>(index);
}
