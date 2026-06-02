#ifndef EPIC_TRAJ_OPT_YAW_INTEGRAL_COST_MANAGER_HPP
#define EPIC_TRAJ_OPT_YAW_INTEGRAL_COST_MANAGER_HPP

#include <algorithm>
#include <cmath>

#include <Eigen/Eigen>

namespace fast_planner {
namespace traj_opt {
namespace cost_functional_manager {

class YawIntegralCostManager {
 public:
  void reset(const Eigen::Matrix3Xd *guide_points,
             const Eigen::Vector3d &head_yaw_state,
             const Eigen::Vector3d &tail_yaw_state,
             int samples_per_piece,
             double smooth_eps,
             double guide_weight) {
    guide_points_ = guide_points;
    head_yaw_state_ = head_yaw_state;
    tail_yaw_state_ = tail_yaw_state;
    samples_per_piece_ = std::max(1, samples_per_piece);
    smooth_eps_ = std::max(1.0e-6, smooth_eps);
    guide_weight_ = std::max(0.0, guide_weight);
  }

  double evaluateIntegral(int logical_idx,
                          double t_local,
                          double t_global,
                          int seg_idx,
                          int step_in_seg,
                          const Eigen::Vector3d &position,
                          const Eigen::Vector3d &velocity,
                          const Eigen::Vector3d &acceleration,
                          const Eigen::Vector3d &jerk,
                          Eigen::Vector3d &grad_position,
                          Eigen::Vector3d &grad_velocity,
                          Eigen::Vector3d &grad_acceleration,
                          Eigen::Vector3d &grad_jerk,
                          double &grad_time) {
    (void)logical_idx;
    (void)t_local;
    (void)t_global;
    (void)velocity;
    (void)acceleration;
    (void)jerk;

    grad_position.setZero();
    grad_velocity.setZero();
    grad_acceleration.setZero();
    grad_jerk.setZero();
    grad_time = 0.0;

    if (!ready() || guide_weight_ <= 0.0) {
      return 0.0;
    }

    const double alpha = std::min(1.0, std::max(0.0,
        static_cast<double>(step_in_seg) / static_cast<double>(samples_per_piece_)));
    const double guide_yaw = interpolateGuideYaw(seg_idx, alpha);
    const double diff = position.x() - guide_yaw;
    const double abs_diff = std::abs(diff);

    double value = 0.0;
    double deriv = 0.0;
    if (!smoothedL1(abs_diff, smooth_eps_, value, deriv)) {
      return 0.0;
    }

    const double sign = diff > 0.0 ? 1.0 : (diff < 0.0 ? -1.0 : 0.0);
    grad_position.x() += guide_weight_ * deriv * sign;
    return guide_weight_ * value;
  }

  template <typename SampleBuffer, typename GradPositionMat, typename GradTimeVec>
  double evaluateSample(const SampleBuffer &samples,
                        GradPositionMat &grad_position,
                        GradTimeVec &grad_time) {
    (void)samples;
    grad_position.setZero();
    grad_time.setZero();
    return 0.0;
  }

 private:
  bool ready() const {
    return guide_points_;
  }

  double interpolateGuideYaw(int seg_idx, double alpha) const {
    const int inner_cols = guide_points_ ? static_cast<int>(guide_points_->cols()) : 0;
    const double start =
        seg_idx == 0 ? head_yaw_state_(0) : (*guide_points_)(0, seg_idx - 1);
    const double goal =
        seg_idx >= inner_cols ? tail_yaw_state_(0) : (*guide_points_)(0, seg_idx);
    return (1.0 - alpha) * start + alpha * goal;
  }

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

  const Eigen::Matrix3Xd *guide_points_{nullptr};
  Eigen::Vector3d head_yaw_state_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d tail_yaw_state_{Eigen::Vector3d::Zero()};
  int samples_per_piece_{1};
  double smooth_eps_{1.0e-3};
  double guide_weight_{0.0};
};

}  // namespace cost_functional_manager
}  // namespace traj_opt
}  // namespace fast_planner

#endif
