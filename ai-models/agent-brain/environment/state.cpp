#include "state.hpp"

#include <vector>

#include "location.hpp"

state::state(double hunger, double money, double moral, double energy, double food, location location){
    m_hunger = hunger;
    m_money = money;
    m_moral = moral;
    m_energy = energy;
    m_food = food;
    m_location = location;
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

void state::set_food(double food) {
    m_food = food;
}

void state::set_location(location location) {
    m_location = location;
}

location state::get_location() {
    return m_location;
}

double state::get_food() const {
    return m_food;
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
        m_food,
        m_location == location::HOME ? 1.0 : 0.0,
        m_location == location::WORKPLACE ? 1.0 : 0.0,
        m_location == location::SHOP ? 1.0 : 0.0,
    };
}


