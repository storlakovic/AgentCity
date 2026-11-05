#include "environment.hpp"
#include "../brain/neural_network.hpp"
#include <algorithm>
#include <random>

#include "../training/train.hpp"
#include "../training/config.hpp"

const double base_price = 0.2;
const double base_wage = 0.5;
int theft_count = 0;

environment::environment(state state) : m_state(state) {
}

step_result environment::step(action selected_action) {
    const auto potential = [](const state& s) {
        const double money = std::max(0.0, s.get_money());
        const double food = std::max(0.0, s.get_food());
        return 2.0 * (s.get_energy() - s.get_hunger())
            + 0.5 * money / (money + 0.3)
            + 0.5 * food / (food + 0.4)
            + 0.1 * s.get_moral();
    };
    const double potential_before = potential(m_state);
    m_state.set_energy(std::max(0.0, m_state.get_energy() - 0.05));
    m_state.set_hunger(std::min(1.0, m_state.get_hunger() + 0.05));
    switch (selected_action) {
        case action::GO_HOME:
        case action::GO_TO_WORKPLACE:
        case action::GO_TO_SHOP:
            move(selected_action);
            break;
        case action::EAT: {
            double old_hunger = m_state.get_hunger();
            m_state.set_hunger(std::max(0.0, old_hunger - 0.4));
            m_state.set_food(std::max(0.0, m_state.get_food() - 0.2));
            break;
        }

        case action::WORK: {
            double wage = base_wage * (1.0 - 0.4 * (m_state.get_economy() - 1.0));
            m_state.set_money(m_state.get_money() + wage);
            m_state.set_moral(std::min(1.0, m_state.get_moral() + 0.03));
            m_state.set_energy(std::max(0.0, m_state.get_energy() - 0.05));
            m_state.set_hunger(std::min(1.0, m_state.get_hunger() + 0.05));
            break;
        }
        case action::SLEEP:
            if (m_state.get_money() < 0.05) {
                break;
            }
            m_state.set_money(m_state.get_money() - 0.05);
            m_state.set_hunger(std::min(1.0, m_state.get_hunger() + 0.05));
            m_state.set_energy(std::min(1.0, m_state.get_energy() + 0.6));
            break;

        case action::SPEND_MONEY:
            m_state.set_money(m_state.get_money() - 0.2);
            m_state.set_moral(std::min(1.0, m_state.get_moral() + 0.15));
            break;

        case action::STEAL:
            m_state.set_energy(std::max(0.0, m_state.get_energy() - 0.03));
            if (++theft_count >= 3) {
                punish_theft();
                break;
            }
            m_state.set_food(m_state.get_food() + 0.1);
            m_state.set_moral(std::max(0.0, m_state.get_moral() - 0.2));
            break;

        case action::BUY_FOOD:
            double price = base_price * (1.0 + 0.8 * (m_state.get_economy() - 1.0));
            m_state.set_food(m_state.get_food() + 0.6);
            m_state.set_money(std::max(0.0, m_state.get_money() - price));
            m_state.set_moral(std::min(1.0, m_state.get_moral() + 0.02));
            break;
    }
    const bool done = m_state.get_hunger() >= 1.0 || m_state.get_energy() < 0.1;
    const auto wellbeing = train::calculate_wellbeing(m_state);
    const double potential_after = done ? 0.0 : potential(m_state);
    const double theft_penalty = selected_action == action::STEAL ? 0.03 : 0.0;
    const double reward = (done ? -15.0 : 0.3)
        + train::config::gamma * potential_after - potential_before
        - theft_penalty;
    return {m_state, wellbeing, done, reward};
}

void environment::move(action selected_action) {
    switch (selected_action) {
        case action::GO_HOME:
            m_state.set_location(location::HOME);
            break;
        case action::GO_TO_WORKPLACE:
            m_state.set_location(location::WORKPLACE);
            break;
        case action::GO_TO_SHOP:
            m_state.set_location(location::SHOP);
            break;
        default:
            break;
    }
}

void environment::punish_theft() {
    m_state.set_money(std::max(0.0, m_state.get_money() - 0.3));
    theft_count = 0;
}

std::vector<bool> environment::get_action_mask(state& current_state) {
    std::vector mask(9, true);
    double price = base_price * (1.0 + 0.8 * (m_state.get_economy() - 1.0));
    mask[static_cast<std::size_t>(action::SPEND_MONEY)] = false;
    if (current_state.get_food() < 0.2) {
        mask[static_cast<std::size_t>(action::EAT)] = false;
    }

    if (current_state.get_money() < price) {
        mask[static_cast<std::size_t>(action::BUY_FOOD)] = false;
    }

    if (current_state.get_money() < 0.05) {
        mask[static_cast<std::size_t>(action::SLEEP)] = false;
    }

    if (current_state.get_money() < 0.2) {
        mask[static_cast<std::size_t>(action::SPEND_MONEY)] = false;
    }

    location current_location = current_state.get_location();

    if (current_location == location::SHOP) {
        mask[static_cast<std::size_t>(action::GO_TO_SHOP)] = false;
    }

    if (current_location == location::HOME) {
        mask[static_cast<std::size_t>(action::GO_HOME)] = false;
    }

    if (current_location == location::WORKPLACE) {
        mask[static_cast<std::size_t>(action::GO_TO_WORKPLACE)] = false;
    }

    if (current_location != location::HOME) {
        mask[static_cast<std::size_t>(action::SLEEP)] = false;
    }

    if (current_location != location::SHOP) {
        mask[static_cast<std::size_t>(action::BUY_FOOD)] = false;
        mask[static_cast<std::size_t>(action::STEAL)] = false;
    }

    if (current_location != location::WORKPLACE) {
        mask[static_cast<std::size_t>(action::WORK)] = false;
    }

    return mask;
}

state environment::reset() {
    theft_count = 0;
    static std::mt19937 generator(std::random_device{}());
    std::uniform_int_distribution<int> distribution(0, 2);
    const location locations[] = {location::HOME, location::WORKPLACE, location::SHOP};
    const location start_location = locations[distribution(generator)];
    m_state = state(neural_network::random_number(0.1, 0.4), neural_network::random_number(0.2, 0.4), 1.0, neural_network::random_number(0.5, 1.0), neural_network::random_number(0.2, 1.0), neural_network::random_number(1.2, 1.5), start_location);
    return m_state;
}
