#include "experience.hpp"

experience::experience(std::vector<double> state, std::size_t action, double reward, double return_value) {
    m_state = state;
    m_action = action;
    m_reward = reward;
    m_return_value = return_value;
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

void experience::set_return_value(double return_value) {
    m_return_value = return_value;
}

double experience::get_return_value() {
    return m_return_value;
}

