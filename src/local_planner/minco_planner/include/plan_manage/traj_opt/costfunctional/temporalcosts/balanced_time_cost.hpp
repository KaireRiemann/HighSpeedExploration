#ifndef EPIC_TRAJ_OPT_BALANCED_TIME_COST_HPP
#define EPIC_TRAJ_OPT_BALANCED_TIME_COST_HPP

#include <algorithm>
#include <vector>

#include <Eigen/Eigen>

namespace fast_planner {
namespace traj_opt {
namespace cost_functional {

struct BalancedTimeCost {
  double linear_weight{0.0};
  double lower_bound{0.0};
  double lower_bound_weight{0.0};
  double smooth_eps{1.5e-1};

  double operator()(const std::vector<double> &times, Eigen::VectorXd &grad) const {
    double total = 0.0;
    double cost = 0.0;
    for (std::size_t i = 0; i < times.size(); ++i) {
      total += times[i];
      cost += linear_weight * times[i];
      grad(static_cast<Eigen::Index>(i)) += linear_weight;
    }

    double penalty = 0.0;
    double d_penalty = 0.0;
    if (lower_bound_weight > 0.0 &&
        smoothedL1(lower_bound - total, std::max(1.0e-6, smooth_eps), penalty, d_penalty)) {
      cost += lower_bound_weight * penalty;
      grad.array() -= lower_bound_weight * d_penalty;
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
