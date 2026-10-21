#pragma once
#include "state.hpp"
#include "../training/train.hpp"

struct step_result {
    state new_state;
    train::wellbeing wellbeing;
    bool done;
};
