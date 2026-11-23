#include <iostream>
#include <vector>
#include "mass_spring.h"
#include "utilities.h"


int main()
{

  MassSpringSystem sys(1.0, 4.0, 0.5);
  std::cout << "omega_n: " << sys.omega_n() << '\n';
  std::cout << "zeta: " << sys.zeta() << '\n';

  auto t = utilities::linspace(0.0, 15.0, 16);
  auto x = sys.undamped_trajectory(t, 1.0, 0.0);
  std::cout << "x(0): " << x.front() << '\n';



  // std::cout << "displacement at every second (new)" << '\n';
  // std::cout << "**************" << '\n';
  // for (std::size_t i = 0; i < x.size(); ++i)
  // {
  //   std::cout << "At t=" << t[i] << "sec, x(t)=" << x[i] << '\n';
  // }
  
  return 0;
}
