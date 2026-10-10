#pragma once
#include <vector>


class experience {
    std::vector<double> m_state;
    std::size_t m_action;
    double m_reward;

    public:
    experience(std::vector<double> state, std::size_t action, double reward);
    double get_reward();
    size_t get_action();
    std::vector<double> get_state();
};



