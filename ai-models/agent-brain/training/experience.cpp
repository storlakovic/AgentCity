#include "experience.hpp"

experience::experience(std::vector<double> state, std::size_t action, double reward) {
    m_state = state;
    m_action = action;
    m_reward = reward;
}

std::vector<double> experience::get_state() {
    return m_state;
}

double experience::get_reward() {
    return m_reward;
}

size_t experience::get_action() {
    return m_action;
}
