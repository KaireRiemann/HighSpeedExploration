#ifndef EPIC_TRAJ_OPT_TARGET_TIME_COST_HPP
#define EPIC_TRAJ_OPT_TARGET_TIME_COST_HPP

#include <algorithm>
#include <vector>

#include <Eigen/Eigen>

namespace fast_planner {
namespace traj_opt {
namespace cost_functional {

struct TargetTimeCost {
  double target_duration{0.0};
  double weight{0.0};
  double smooth_eps{1.5e-1};

  double operator()(const std::vector<double> &times, Eigen::VectorXd &grad) const {
    if (weight <= 0.0 || target_duration <= 0.0) {
      return 0.0;
    }

    double total = 0.0;
    for (const double time : times) {
      total += time;
    }

    double value = 0.0;
    double deriv = 0.0;
    double cost = 0.0;
    const double mu = std::max(1.0e-6, smooth_eps);
    if (smoothedL1(total - target_duration, mu, value, deriv)) {
      cost += weight * value;
      grad.array() += weight * deriv;
    } else if (smoothedL1(target_duration - total, mu, value, deriv)) {
      cost += weight * value;
      grad.array() -= weight * deriv;
    }
    return cost;
  }

 private:
  static bool smoothedL1(double x, double mu, double &f, double &df) {
    if (x < 0.0) {
      return false;
    }
    if (x > mu) {
      f = x - 0.5 * mu;
      df = 1.0;
      return true;
    }
    const double xdmu = x / mu;
    const double sqrxdmu = xdmu * xdmu;
    const double mumxd2 = mu - 0.5 * x;
    f = mumxd2 * sqrxdmu * xdmu;
    df = sqrxdmu * (-0.5 * xdmu + 3.0 * mumxd2 / mu);
    return true;
  }
};

}  // namespace cost_functional
}  // namespace traj_opt
}  // namespace fast_planner

#endif
