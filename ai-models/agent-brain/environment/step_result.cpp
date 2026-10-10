#include "step_result.hpp"

step_result::step_result( state state, double reward, bool done) : new_state(state) {
    this->reward = reward;
    this->done = done;
}
