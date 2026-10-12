#include "state.hpp"

#include <vector>

state::state(double hunger, double money, double moral, double energy){
    m_hunger = hunger;
    m_money = money;
    m_moral = moral;
    m_energy = energy;
}

double state::get_hunger() const {
    return m_hunger;
}

void state::set_hunger(double hunger) {
    m_hunger = hunger;
}

void state::set_moral(double moral) {
    m_moral = moral;
}

void state::set_money(double money) {
    m_money = money;
}

void state::set_energy(double energy) {
    m_energy = energy;
}

double state::get_energy() const {
    return m_energy;
}

double state::get_money() const{
    return m_money;
}

double state::get_moral() const{
    return m_moral;
}

std::vector<double> state::to_vector() {
    return {
        m_hunger,
        m_money,
        m_moral,
        m_energy,
    };
}


