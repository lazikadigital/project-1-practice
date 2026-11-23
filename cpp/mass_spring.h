#pragma once

#include <vector>

class MassSpringSystem {
private:
    double m_;
    double k_;
    double c_;

    // void validate() const;

public:
    // Constructor
    MassSpringSystem(double m, double k, double c = 0.0);

    // Getters
    double m() const;
    double k() const;
    double c() const;

    // Setters
    void set_m(double new_m);
    void set_k(double new_k);
    void set_c(double new_c = 0.0);
    
    double omega_n() const;   // √(k/m)
    double zeta() const;      // c / (2√(km))

    // Undamped analytical displacement at one time
    double undamped_x(double t, double x0, double v0) const;

    // Full trajectory on a time grid (uses undamped formula; ignores c for now
    // or document that this path is undamped-only)
    std::vector<double> undamped_trajectory(
        const std::vector<double>& t,
        double x0,
        double v0) const;
};


/* 
    Return the undamped natural frequency ωₙ in rad/s.

    Parameters
    ----------
    m : float
        Mass (> 0)
    k : float
        Spring stiffness (> 0)

    Returns
    -------
    float
        ωₙ = √(k / m)
    */
// double natural_frequency(double m, double k); // restore


/*
    Return displacement for the undamped free response 

    Parameters
    ----------
     m : float
        Mass (> 0)
    k : float
        Spring stiffness (> 0)
    t: float
        Time
    x0: float
        Initial displacement
    v0: float
        Initial velocity
    ωₙ: float
        Natural frequency (> 0)   
        
    Returns
    -------
    float
        X(t) = x0 * cos(ωₙt) + (v0 / ωₙ) * sin(ωₙt)    

  */
// double undamped_free_response_displacement(double x0, double v0, double m, double k, double t);



/*
    Return vectorized displacement for the undamped free response 
*/
// std::vector<double> undamped_free_response(
//     const std::vector<double>& t,
//     double x0, 
//     double v0,
//     double m, 
//     double k 
//   );
