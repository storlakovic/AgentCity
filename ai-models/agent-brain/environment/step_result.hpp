#pragma once
#include "state.hpp"

struct step_result {
    state new_state;
    double reward;
    bool done;
};
