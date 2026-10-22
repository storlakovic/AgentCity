#pragma once
#include <vector>

#include "train.hpp"


class experience {
    std::vector<double> m_state;
    std::size_t m_action;
    train::wellbeing m_wellbeing;
    double m_return_value;
    double m_reward;
    std::vector<bool> m_mask;

    public:
    experience(std::vector<double> state, std::size_t action, train::wellbeing wellbeing, double return_value, std::vector<bool> mask, double reward);
    size_t get_action();
    void set_return_value(double return_value);
    void set_wellbeing(train::wellbeing wellbeing);
    double get_return_value();
    double get_reward() const;
    train::wellbeing get_wellbeing();
    std::vector<bool> get_mask();
    std::vector<double> get_state();
};



