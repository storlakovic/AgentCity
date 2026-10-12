#pragma once
#include <vector>


class experience {
    std::vector<double> m_state;
    std::size_t m_action;
    double m_reward;
    double m_return_value;

    public:
    experience(std::vector<double> state, std::size_t action, double reward, double return_value);
    double get_reward();
    size_t get_action();
    void set_return_value(double return_value);
    double get_return_value();
    std::vector<double> get_state();
};



