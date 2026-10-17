#pragma once
#include <vector>

#include "location.hpp"

class state {
    double m_hunger;
    double m_money;
    double m_moral;
    double m_energy;
    double m_food;
    double m_economy;
    location m_location;

    public:
    state(double hunger, double money, double moral, double energy, double food, double economy, location location);
    double get_hunger() const;

    void set_hunger(double hunger);
    void set_moral(double moral);
    void set_money(double money);
    void set_energy(double energy);
    void set_food(double food);
    void set_economy(double economy);
    void set_location(location location);
    location get_location();
    double get_food() const;
    double get_energy() const;
    double get_money() const;
    double get_moral() const;
    double get_economy() const;
    std::vector<double> to_vector();
};


