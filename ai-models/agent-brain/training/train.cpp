#include "train.hpp"

#include <iostream>
#include <random>

#include "../brain/neural_network.hpp"

#include <vector>
#include "../environment/environment.hpp"
#include "experience.hpp"


void train::train_agent(int epochs, double learning_rate) {
    environment env( state(0.5, 0.5, 1.0));
    for (int epoch = 0; epoch < epochs; epoch++) {
        std::vector<experience> trajectory;
        state current_state = env.reset();
        bool done = false;
        double total_reward = 0.0;

        for (int step = 0; step < 100 && !done; step++) {
            std::vector<double> probabilities = neural_network::forward(current_state.to_vector());

            action selected_action = sample_action(probabilities);

            step_result result = env.step(selected_action);

            trajectory.push_back({current_state.to_vector(), static_cast<std::size_t>(selected_action), result.reward});

            total_reward += result.reward;

            current_state = result.new_state;
            done = result.done;
        }
        for (auto& exp : trajectory) {
            std::vector<double> probabilities = neural_network::forward(exp.get_state());

            std::vector<double> output_deltas(probabilities.size());

            for (std::size_t output = 0; output < probabilities.size(); output++) {
                double target = output == exp.get_action() ? 1.0 : 0.0;
                output_deltas[output] = (probabilities[output] - target) * exp.get_reward();
            }
            const auto& current_state = exp.get_state();
            if (epoch % 100 == 0) {
                std::cout
                    << "{ "
                    << "\"hunger\": " << current_state[0] << ", "
                    << "\"money\": " << current_state[1] << ", "
                    << "\"moral\": " << current_state[2] << ", "
                    << "\"action\": \""
                    << action_to_string(exp.get_action()) << "\", "
                    << "\"reward\": " << exp.get_reward()
                    << " }\n";
            }
            neural_network::backward(output_deltas);

            neural_network::update_weights(learning_rate);
        }

        if (epoch % 100 == 0) {
            std::cout
                << "Episode: " << epoch
                << " Reward: " << total_reward
                << '\n';
        }
    }
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
    }

    return "UNKNOWN";
}

action train::sample_action(std::vector<double>& probabilities) {
    static std::mt19937 generator(std::random_device{}());

    std::discrete_distribution<std::size_t> distribution(probabilities.begin(),probabilities.end());

    std::size_t index = distribution(generator);

    return static_cast<action>(index);
}

