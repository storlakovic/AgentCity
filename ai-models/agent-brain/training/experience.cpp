#include "experience.hpp"

experience::experience(std::vector<double> state, std::size_t action, train::wellbeing wellbeing, double return_value, std::vector<bool> mask) {
    m_state = state;
    m_action = action;
    m_wellbeing = wellbeing;
    m_return_value = return_value;
    m_mask = mask;
}

std::vector<double> experience::get_state() {
    return m_state;
}

size_t experience::get_action() {
    return m_action;
}

void experience::set_return_value(double return_value) {
    m_return_value = return_value;
}

void experience::set_wellbeing(train::wellbeing wellbeing) {
    m_wellbeing = wellbeing;
}

double experience::get_return_value() {
    return m_return_value;
}

train::wellbeing experience::get_wellbeing() {
    return m_wellbeing;
}

std::vector<bool> experience::get_mask() {
    return m_mask;
}


