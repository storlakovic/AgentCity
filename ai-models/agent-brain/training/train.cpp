#include "train.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <iostream>
#include <utility>

#include "../brain/neural_network.hpp"

#include <vector>

#include "config.hpp"
#include "../environment/environment.hpp"
#include "experience.hpp"
#include "nlohmann/json.hpp"

void train::train_agent() {
    environment env(state(0.5, 0.5, 1.0, 0.5, 0.2, 1.0, location::HOME));
    std::vector<std::vector<experience>> batch;
    batch.reserve(config::batch_size);
    double best_avg_reward = evaluate_network(env);
    std::cout << "Initial evaluation | Avg reward: " << best_avg_reward << '\n';

    for (int epoch = 0; epoch < config::epochs; epoch++) {
        std::vector<experience> trajectory;
        auto result = collect_experiences(trajectory, env);
        batch.push_back(std::move(trajectory));
        if (batch.size() == config::batch_size || epoch + 1 == config::epochs) {
            update_network(batch);
            batch.clear();
        }
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
        const auto network_output = neural_network::forward(current_state.to_vector());
        std::vector<double> output_values = network_output;
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

void train::update_network(std::vector<std::vector<experience>>& batch) {
    if (batch.empty()) {
        return;
    }

    for (auto& trajectory : batch) {
        double return_value = 0.0;
        for (int i = static_cast<int>(trajectory.size()) - 1; i >= 0; i--) {
            return_value = trajectory[i].get_reward() + config::gamma * return_value;
            trajectory[i].set_return_value(return_value);
        }
    }

    static std::vector<double> return_baselines(config::max_steps, 0.0);
    static std::vector<bool> baseline_initialized(config::max_steps, false);
    constexpr double baseline_update_rate = 0.01;

    std::vector<experience*> experiences;
    std::vector<double> advantages;
    std::vector<double> return_sums(config::max_steps, 0.0);
    std::vector<std::size_t> return_counts(config::max_steps, 0);

    double mean_squared_advantage = 0.0;
    for (auto& trajectory : batch) {
        for (std::size_t i = 0; i < trajectory.size(); i++) {
            const double current_return = trajectory[i].get_return_value();
            const double advantage = baseline_initialized[i]
                ? current_return - return_baselines[i]
                : 0.0;

            experiences.push_back(&trajectory[i]);
            advantages.push_back(advantage);
            mean_squared_advantage += advantage * advantage;

            return_sums[i] += current_return;
            return_counts[i]++;
        }
    }
    if (experiences.empty()) {
        return;
    }
    mean_squared_advantage /= static_cast<double>(experiences.size());
    const double advantage_scale =
        std::sqrt(mean_squared_advantage) + 1e-8;

    std::vector<std::vector<double>> accumulated_gradients;
    std::vector<std::vector<double>> accumulated_deltas;

    for (const auto& current_layer : neural_network::layers) {
        accumulated_gradients.emplace_back(current_layer.get_input_count() * current_layer.get_neuron_count(), 0.0);
        accumulated_deltas.emplace_back(current_layer.get_neuron_count(), 0.0);
    }

    for (std::size_t experience_index = 0;
         experience_index < experiences.size();
         experience_index++) {
        auto& exp = *experiences[experience_index];
        const auto network_output = neural_network::forward(exp.get_state());
        std::vector<double> output_values = network_output;
        const auto mask = exp.get_mask();
        for (std::size_t i = 0; i < output_values.size(); i++) {
            if (!mask[i]) {
                output_values[i] = -1e9;
            }
        }
        std::vector<double> probabilities = neural_network::softmax(output_values);
        std::vector<double> output_deltas(network_output.size(), 0.0);
        const double normalized_advantage =
            advantages[experience_index] / advantage_scale;
        for (std::size_t output = 0; output < probabilities.size(); output++) {
            if (!mask[output]) {
                continue;
            }
            double target = output == exp.get_action() ? 1.0 : 0.0;
            output_deltas[output] = (probabilities[output] - target) * normalized_advantage;
        }
        neural_network::backward(output_deltas);

        for (std::size_t layer_index = 0; layer_index < neural_network::layers.size(); layer_index++) {
            auto& current_layer = neural_network::layers[layer_index];

            for (std::size_t gradient_index = 0; gradient_index < accumulated_gradients[layer_index].size(); gradient_index++) {
                accumulated_gradients[layer_index][gradient_index] +=
                    current_layer.get_gradient_at(gradient_index);
            }

            for (std::size_t neuron = 0; neuron < current_layer.get_neuron_count(); neuron++) {
                accumulated_deltas[layer_index][neuron] +=
                    current_layer.get_delta_at(neuron);
            }
        }
    }

    const double episode_scale = 1.0 / static_cast<double>(batch.size());

    for (std::size_t layer_index = 0; layer_index < neural_network::layers.size(); layer_index++) {
        auto& current_layer = neural_network::layers[layer_index];

        for (double& gradient : accumulated_gradients[layer_index]) {
            gradient *= episode_scale;
        }
        current_layer.set_gradients(accumulated_gradients[layer_index]);

        for (std::size_t neuron = 0; neuron < current_layer.get_neuron_count(); neuron++) {
            current_layer.set_delta_at(
                neuron,
                accumulated_deltas[layer_index][neuron] * episode_scale);
        }
    }

    neural_network::update_weights(config::learning_rate);
    for (std::size_t i = 0; i < return_counts.size(); i++) {
        if (return_counts[i] == 0) {
            continue;
        }
        const double mean_return = return_sums[i] / static_cast<double>(return_counts[i]);
        return_baselines[i] = baseline_initialized[i]
            ? (1.0 - baseline_update_rate) * return_baselines[i] +
              baseline_update_rate * mean_return
            : mean_return;
        baseline_initialized[i] = true;
    }
}

double train::evaluate_network(environment& env) {
    double total_reward = 0.0;
    for (int i = 0; i < config::evaluation_episodes; i++) {
        state test_state = env.reset();
        for (int step = 0; step < config::max_steps; step++) {
            auto mask = env.get_action_mask(test_state);
            const auto network_output = neural_network::forward(test_state.to_vector());
            std::vector<double> output_values = network_output;
            for (std::size_t i = 0; i < output_values.size(); i++) {
                if (!mask[i]) {
                    output_values[i] = -1e9;
                }
            }
            std::vector<double> probabilities = neural_network::softmax(output_values);
            const auto best_action = std::max_element(
                probabilities.begin(),
                probabilities.end()
            );

            const action selected_action = static_cast<action>(
                std::distance(probabilities.begin(), best_action)
            );

            if (i == 0) {
                std::cout << "\n[Evaluation] Step: " << step + 1
                          << " | Hunger: " << test_state.get_hunger()
                          << " | Energy: " << test_state.get_energy()
                          << " | Money: " << test_state.get_money()
                          << " | Food: " << test_state.get_food()
                          << '\n';

                for (std::size_t a = 0; a < probabilities.size(); ++a) {
                    if (mask[a]) {
                        std::cout << action_to_string(a)
                                  << ": " << probabilities[a] * 100.0 << "%\n";
                    }
                }
                std::cout << "Selected: "
                          << action_to_string(static_cast<std::size_t>(selected_action))
                          << '\n';
            }

            step_result result = env.step(selected_action);
            if (i == 0) {
                std::cout << "Reward: " << result.reward
                          << " | Done: " << (result.done ? "yes" : "no") << '\n';
            }
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
        + 0.2 * w.financial_stability
        + 0.2 * s.get_moral()
        + 0.1 * s.get_food();

    w.social_wellbeing = 0.7 * w.health + 0.3 * w.financial_stability;

    return w;
}

double train::calculate_wellbeing_score(const wellbeing& w) {
    constexpr double HEALTH_WEIGHT = 0.75;
    constexpr double HAPPINESS_WEIGHT = 0.25;

    return
        HEALTH_WEIGHT * w.health +
        HAPPINESS_WEIGHT * w.happiness;
}

action train::sample_action(const std::vector<double>& probabilities) {
    static std::mt19937 generator(std::random_device{}());

    std::discrete_distribution<std::size_t> distribution(probabilities.begin(), probabilities.end());

    std::size_t index = distribution(generator);

    return static_cast<action>(index);
}
