#pragma once
#include <vector>

class state {
    double m_hunger;
    double m_money;
    double m_moral;

    public:
    state(double hunger, double money, double moral);
    double get_hunger() const;

    void set_hunger(double hunger);
    void set_moral(double moral);
    void set_money(double money);
    double get_money() const;
    double get_moral() const;
    std::vector<double> to_vector();
};


