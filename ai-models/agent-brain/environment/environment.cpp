#include "environment.hpp"
#include "../brain/neural_network.hpp"
#include <algorithm>
#include <random>

environment::environment(state state) : m_state(state) {
}

step_result environment::step(action selected_action) {
    double reward = 0.0;
    bool done = false;
    m_state.set_energy(std::max(0.0, m_state.get_energy() - 0.05));
    m_state.set_hunger(std::min(1.0, m_state.get_hunger() + 0.05));
    switch (selected_action) {
        case action::GO_HOME:
        case action::GO_TO_WORKPLACE:
        case action::GO_TO_SHOP:
            reward += move(selected_action);
            break;
        case action::EAT: {
            if (m_state.get_money() <= 0.1 || m_state.get_hunger() <= 0.1) {
                reward -= 0.3;
                break;
            }
            double old_hunger = m_state.get_hunger();
            m_state.set_hunger(std::max(0.0, old_hunger - 0.4));

            m_state.set_money(std::max(0.0, m_state.get_money() - 0.1));
            reward += 0.5;
            break;
        }

        case action::WORK:
            if (m_state.get_energy() < 0.1) {
                reward -= 0.2;
                break;
            }
            m_state.set_money(m_state.get_money() + 0.3);
            m_state.set_energy(std::max(0.0, m_state.get_energy() - 0.1));
            m_state.set_hunger(std::min(1.0, m_state.get_hunger() + 0.1));
            reward += 0.5;
            break;

        case action::SLEEP:
            if (m_state.get_energy() >= 0.9) {
                reward -= 0.3;
                break;
            }
            m_state.set_hunger(std::min(1.0, m_state.get_hunger() + 0.05));
            m_state.set_energy(std::min(1.0, m_state.get_energy() + 0.4));
            reward += 0.5;
            break;

        case action::SPEND_MONEY:
            if (m_state.get_money() >= 0.2) {
                m_state.set_money(m_state.get_money() - 0.2);
                reward += 0.3;
            } else {
                reward -= 0.3;
            }
            break;

        case action::STEAL:
            m_state.set_money(m_state.get_money() + 0.1);
            m_state.set_moral(std::max(0.0, m_state.get_moral() - 0.2));
            reward -= 0.3;
            break;
    }
    if (m_state.get_hunger() >= 0.9 || m_state.get_energy() <= 0.1) {
        reward -= 1.0;
    }
    if (m_state.get_hunger() >= 1.0) {
        done = true;
        reward -= 2.0;
    }
    return {m_state, reward, done};
}

double environment::move(action selected_action) {
    double reward = 0.0;
    switch (selected_action) {
        case action::GO_HOME:
            m_state.set_location(location::HOME);
            reward += 0.05;
            break;
        case action::GO_TO_WORKPLACE:
            m_state.set_location(location::WORKPLACE);
            reward += 0.05;
            break;
        case action::GO_TO_SHOP:
            m_state.set_location(location::SHOP);
            reward += 0.05;
            break;
        default:
            break;
    }
    return reward;
}

std::vector<bool> environment::get_action_mask(state& current_state) {
    std::vector mask(8, true);

    switch (current_state.get_location()) {

        case location::HOME:
            mask[static_cast<std::size_t>(action::WORK)] = false;
            mask[static_cast<std::size_t>(action::STEAL)] = false;
            mask[static_cast<std::size_t>(action::GO_HOME)] = false;
            break;

        case location::WORKPLACE:
            mask[static_cast<std::size_t>(action::SLEEP)] = false;
            mask[static_cast<std::size_t>(action::EAT)] = false;
            mask[static_cast<std::size_t>(action::STEAL)] = false;
            mask[static_cast<std::size_t>(action::GO_TO_WORKPLACE)] = false;
            break;

        case location::SHOP:
            mask[static_cast<std::size_t>(action::WORK)] = false;
            mask[static_cast<std::size_t>(action::SLEEP)] = false;
            mask[static_cast<std::size_t>(action::GO_TO_SHOP)] = false;
            break;
    }

    return mask;
}

state environment::reset() {
    static std::mt19937 generator(std::random_device{}());
    std::uniform_int_distribution<int> distribution(0, 2);
    const location locations[] = {location::HOME, location::WORKPLACE, location::SHOP};
    const location start_location = locations[distribution(generator)];
    m_state = state(neural_network::random_number(0.1, 0.4), neural_network::random_number(0.0, 0.1), 1.0, neural_network::random_number(0.5, 1.0), start_location);
    return m_state;
}
