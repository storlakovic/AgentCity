#pragma once
#include "state.hpp"

class step_result {
    public:

    step_result( state state, double reward, bool done);
    state new_state;
    double reward;
    bool done;
};



