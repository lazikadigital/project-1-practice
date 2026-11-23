#include <vector>
#include <stdexcept>
#include "utilities.h"

namespace utilities 
{
  std::vector<double> linspace(double start, double stop, std::size_t num)
{
    if (stop < start)
    {
        throw std::invalid_argument("Time series linspace requires stop >= start.");
    }

    std::vector<double> result;
    if (num == 0)
    {
        return result;
    }

    result.reserve(num);
    if (num == 1)
    {
        result.push_back(start);
        return result;
    }

    const double step = (stop - start) / static_cast<double>(num - 1);

    for (std::size_t i = 0; i < num - 1; ++i)
    {
        result.push_back(start + static_cast<double>(i) * step);
    }

    // Explicitly push stop to guarantee exactness despite floating-point rounding
    result.push_back(stop);

    return result;
}
}
