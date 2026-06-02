#ifndef EPIC_TRAJ_OPT_QUAD_INV_TIME_MAP_HPP
#define EPIC_TRAJ_OPT_QUAD_INV_TIME_MAP_HPP

#include <cmath>

namespace fast_planner {
namespace traj_opt {
namespace temporal_map {

struct QuadInvTimeMap {
  double toTime(double tau) const {
    return tau > 0.0 ? ((0.5 * tau + 1.0) * tau + 1.0)
                     : (1.0 / ((0.5 * tau - 1.0) * tau + 1.0));
  }

  double toTau(double T) const {
    return T > 1.0 ? (std::sqrt(2.0 * T - 1.0) - 1.0)
                   : (1.0 - std::sqrt(2.0 / T - 1.0));
  }

  double backward(double tau, double T, double gradT) const {
    (void)T;
    if (tau > 0.0) {
      return gradT * (tau + 1.0);
    }
    const double den = (0.5 * tau - 1.0) * tau + 1.0;
    return gradT * (1.0 - tau) / (den * den);
  }
};

}  // namespace temporal_map
}  // namespace traj_opt
}  // namespace fast_planner

#endif
