#include <cmath>
#include <vector>
#include <stdexcept> 

#include "mass_spring.h"

// Constructor: Reuse your setters so validation happens upon creation too
MassSpringSystem::MassSpringSystem(double m, double k, double c)
{
    set_m(m);
    set_k(k);
    set_c(c);
}

// --- Getters ---
double MassSpringSystem::m() const 
{
    return m_;
}

double MassSpringSystem::k() const
{
    return k_;
} 

double MassSpringSystem::c() const 
{
    return c_;
}

// --- Setters with Validation ---
void MassSpringSystem::set_m(double new_m)
{
    if (new_m <= 0)
    {
        throw std::invalid_argument("mass must be a positive value");
    }

    m_ = new_m;
}

void MassSpringSystem::set_k(double new_k)
{
    if (new_k <= 0)
    {
        throw std::invalid_argument("stiffness must be a positive value");
    }

    k_ = new_k;
}

void MassSpringSystem::set_c(double new_c)
{
    if (new_c < 0)
    {
        throw std::invalid_argument("damping must be a non-negative value");
    }

    c_ = new_c;
}

// --- Helper Methods ---

// Return natural frequency
double MassSpringSystem::omega_n() const 
{
    return std::sqrt(k_ / m_);
}
// Return damping ration
double MassSpringSystem::zeta() const 
{
    return c_ / (2 * std::sqrt(k_ * m_));  
}
// Return displacement at a given moment in time
double MassSpringSystem::undamped_x(double t, double x0, double v0) const 
{
    double omega { omega_n() };
    return x0 * std::cos(omega * t) + (v0 / omega) * std::sin(omega * t);
}
// Return vectorized array of displacements for the undamped free response over a time interval
std::vector<double> MassSpringSystem::undamped_trajectory(
    const std::vector<double>& t,
    double x0,
    double v0
) const
{
    std::vector<double> x_out(t.size());
    double omega { omega_n() };
    for (std::size_t i = 0; i < t.size(); ++i)
    {
        x_out[i] = x0 * std::cos(omega * t[i]) + (v0 / omega) * std::sin(omega * t[i]);
    }

    return x_out;
}




///////////// reserve ///////////

// double natural_frequency(double m, double k)
// {
//    return std::sqrt(k / m);
// }

// // At a given moment in time
// double undamped_free_response_displacement(double x0, double v0, double m, double k, double t)
// {
//     double omega_n { natural_frequency(m, k)};
//     return x0 * std::cos(omega_n * t) + (v0 / omega_n) * std::sin(omega_n * t);
// }

// Over an array of a time interval - many moments 
// std::vector<double> undamped_free_response(
//     const std::vector<double>& t,
//     double x0, 
//     double v0,
//     double m, 
//     double k 
// )
// {
//     std::vector<double> x_out(t.size());
//     double omega_n { natural_frequency(m, k)};
//     for (std::size_t i = 0; i < t.size(); ++i)
//     {
//         x_out[i] = x0 * std::cos(omega_n * t[i]) + (v0 / omega_n) * std::sin(omega_n * t[i]);
//     }
//     return x_out;
// }
