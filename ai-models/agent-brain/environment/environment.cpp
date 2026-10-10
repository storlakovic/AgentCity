#include "environment.hpp"
#include <algorithm>

environment::environment(state state) : m_state(state) {
}

step_result environment::step(action selected_action) {
    double reward = 0.0;
    bool done = false;

    switch (selected_action) {
        case action::EAT: {
            if (m_state.get_money() < 0.1) {
                reward -= 0.3;
                break;
            }

            double old_hunger = m_state.get_hunger();

            m_state.set_hunger(std::max(0.0, old_hunger - 0.4));

            m_state.set_money(std::max(0.0, m_state.get_money() - 0.1));
            reward += old_hunger - m_state.get_hunger();
            break;
        }

        case action::WORK:
            m_state.set_money(m_state.get_money() + 0.3);
            m_state.set_hunger(std::min(1.0, m_state.get_hunger() + 0.1));
            reward += 0.2;
            break;

        case action::SLEEP:
            m_state.set_hunger(std::min(1.0, m_state.get_hunger() + 0.05));
            reward += 0.1;
            break;

        case action::SPEND_MONEY:
            if (m_state.get_money() >= 0.2) {
                m_state.set_money(m_state.get_money() - 0.2);
                reward += 0.1;
            } else {
                reward -= 0.3;
            }
            break;

        case action::STEAL:
            m_state.set_money(m_state.get_money() + 0.5);
            m_state.set_moral(std::max(0.0, m_state.get_moral() - 0.2));
            reward -= 0.1;
            break;
    }
    if (m_state.get_hunger() >= 0.9) {
        reward -= 1.0;
    }

    if (m_state.get_hunger() >= 1.0) {
        done = true;
        reward -= 2.0;
    }

    return {m_state, reward, done};
}

state environment::reset() {
    m_state = state(0.5,0.5,1.0);
    return m_state;
}
