#pragma once

#include "state.hpp"
#include "action.hpp"
#include "step_result.hpp"

class environment {
    state m_state;

public:
    environment(state state);
    double move(action selected_action);
    state reset();
    step_result step(action selected_action);
};
