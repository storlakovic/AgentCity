#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

#include "brain/neural_network.hpp"
#include "environment/environment.hpp"
#include "training/config.hpp"
#include "training/train.hpp"

namespace {
    void run_model() {
        environment env(state(0.5, 0.5, 1.0, 0.5, 0.2, 1.0, location::HOME));
        state current_state = env.reset();
        for (int step = 0; step < train::config::max_steps; step++) {
            auto output_values = neural_network::forward(current_state.to_vector());
            auto mask = env.get_action_mask(current_state);
            for (std::size_t i = 0; i < output_values.size(); i++) {
                if (!mask[i]) {
                    output_values[i] = -1e9;
                }
            }
            const auto probabilities = neural_network::softmax(output_values);
            const action selected_action = train::sample_action(probabilities);
            const step_result result = env.step(selected_action);
            std::cout << "Step " << step + 1
                    << " | Money: " << current_state.get_money()
                    << " | Hunger: " << current_state.get_hunger()
                    << " | Energy: " << current_state.get_energy()
                    << " | Food: " << current_state.get_food()
                    << " | Economy: " << current_state.get_economy()
                    << " | Action: " << train::action_to_string(static_cast<std::size_t>(selected_action))
                    << " | Reward: " << result.reward << '\n';
            current_state = result.new_state;
            if (result.done) break;
        }
    }
}

int main() {
    std::cout << "run - Gespeichertes Modell ausführen\n" << "train - Neues Modell trainieren\n";
    std::string mode;
    while (true) {
        std::cout << "Modus eingeben: " << std::flush;
        if (!(std::cin >> mode)) {
            return EXIT_FAILURE;
        }
        if (mode == "run" || mode == "train") {
            break;
        }
        std::cout << "Bitte run oder train eingeben.\n";
    }
    if (mode == "train") {
        neural_network::initialize({9, 32, 16, 9});
        //neural_network::load(train::config::checkpoint_path);
    } else {
        neural_network::load(train::config::checkpoint_path);
    }

    if (mode == "run") {
        run_model();
    } else {
        train::train_agent();
    }

    return EXIT_SUCCESS;
}
