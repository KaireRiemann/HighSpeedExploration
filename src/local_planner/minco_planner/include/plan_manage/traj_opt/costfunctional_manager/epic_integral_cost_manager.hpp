#ifndef EPIC_TRAJ_OPT_INTEGRAL_COST_MANAGER_HPP
#define EPIC_TRAJ_OPT_INTEGRAL_COST_MANAGER_HPP

#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Eigen>
#include <gcopter/flatness.hpp>

namespace fast_planner {
namespace traj_opt {
namespace cost_functional_manager {

class EpicIntegralCostManager {
 public:
  using PolyhedronH = Eigen::MatrixX4d;
  using PolyhedraH = std::vector<PolyhedronH>;

  void reset(const PolyhedraH *h_polys,
             const Eigen::VectorXi *h_poly_idx,
             const Eigen::VectorXd *piece_velocity_bounds,
             double smooth_eps,
             const Eigen::VectorXd &magnitude_bounds,
             const Eigen::VectorXd &penalty_weights,
             flatness::FlatnessMap *flat_map) {
    h_polys_ = h_polys;
    h_poly_idx_ = h_poly_idx;
    piece_velocity_bounds_ = piece_velocity_bounds;
    smooth_eps_ = std::max(1.0e-6, smooth_eps);
    magnitude_bounds_ = magnitude_bounds;
    penalty_weights_ = penalty_weights;
    flat_map_ = flat_map;
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
    (void)step_in_seg;

    grad_position.setZero();
    grad_velocity.setZero();
    grad_acceleration.setZero();
    grad_jerk.setZero();
    grad_time = 0.0;

    if (!ready() || seg_idx < 0 || seg_idx >= h_poly_idx_->size()) {
      return 0.0;
    }

    const double vel_max = segmentVelocityBound(seg_idx);
    const double vel_sqr_max = sqr(vel_max);
    const double acc_sqr_max = sqr(magnitude_bounds_(1));
    const double omg_sqr_max = sqr(magnitude_bounds_(2));
    const double theta_max = magnitude_bounds_(3);
    const double thrust_mean = 0.5 * (magnitude_bounds_(4) + magnitude_bounds_(5));
    const double thrust_radius = 0.5 * std::abs(magnitude_bounds_(5) - magnitude_bounds_(4));
    const double thrust_sqr_radius = thrust_radius * thrust_radius;

    const double weight_pos = penalty_weights_(0);
    const double weight_vel = penalty_weights_(1);
    const double weight_acc = penalty_weights_(2);
    const double weight_omg = penalty_weights_(3);
    const double weight_theta = penalty_weights_(4);
    const double weight_thrust = penalty_weights_(5);

    double thrust = 0.0;
    Eigen::Vector4d quat = Eigen::Vector4d::Zero();
    Eigen::Vector3d omega = Eigen::Vector3d::Zero();
    flat_map_->forward(velocity, acceleration, jerk, 0.0, 0.0, thrust, quat, omega);

    double penalty = 0.0;
    Eigen::Vector3d flat_grad_pos = Eigen::Vector3d::Zero();
    Eigen::Vector3d flat_grad_vel = Eigen::Vector3d::Zero();
    Eigen::Vector3d direct_grad_acc = Eigen::Vector3d::Zero();
    Eigen::Vector3d flat_grad_omega = Eigen::Vector3d::Zero();
    Eigen::Vector4d grad_quat = Eigen::Vector4d::Zero();
    double grad_thrust = 0.0;

    const int h_id = (*h_poly_idx_)(seg_idx);
    if (h_id >= 0 && h_id < static_cast<int>(h_polys_->size())) {
      const PolyhedronH &poly = (*h_polys_)[h_id];
      for (int i = 0; i < poly.rows(); ++i) {
        const Eigen::Vector3d normal = poly.block<1, 3>(i, 0).transpose();
        const double violation = normal.dot(position) + poly(i, 3);
        double value = 0.0;
        double deriv = 0.0;
        if (smoothedL1(violation, smooth_eps_, value, deriv)) {
          flat_grad_pos += weight_pos * deriv * normal;
          penalty += weight_pos * value;
        }
      }
    }

    double value = 0.0;
    double deriv = 0.0;
    if (smoothedL1(velocity.squaredNorm() - vel_sqr_max, smooth_eps_, value, deriv)) {
      flat_grad_vel += weight_vel * deriv * 2.0 * velocity;
      penalty += weight_vel * value;
    }

    if (smoothedL1(acceleration.squaredNorm() - acc_sqr_max, smooth_eps_, value, deriv)) {
      direct_grad_acc += weight_acc * deriv * 2.0 * acceleration;
      penalty += weight_acc * value;
    }

    if (smoothedL1(omega.squaredNorm() - omg_sqr_max, smooth_eps_, value, deriv)) {
      flat_grad_omega += weight_omg * deriv * 2.0 * omega;
      penalty += weight_omg * value;
    }

    const double cos_theta_raw = 1.0 - 2.0 * (quat(1) * quat(1) + quat(2) * quat(2));
    const double cos_theta = std::clamp(cos_theta_raw, -1.0, 1.0);
    if (smoothedL1(std::acos(cos_theta) - theta_max, smooth_eps_, value, deriv)) {
      const double denom = std::sqrt(std::max(1.0e-9, 1.0 - cos_theta * cos_theta));
      grad_quat += weight_theta * deriv / denom *
                   4.0 * Eigen::Vector4d(0.0, quat(1), quat(2), 0.0);
      penalty += weight_theta * value;
    }

    const double thrust_violation = (thrust - thrust_mean) * (thrust - thrust_mean) -
                                    thrust_sqr_radius;
    if (smoothedL1(thrust_violation, smooth_eps_, value, deriv)) {
      grad_thrust += weight_thrust * deriv * 2.0 * (thrust - thrust_mean);
      penalty += weight_thrust * value;
    }

    double grad_psi = 0.0;
    double grad_dpsi = 0.0;
    flat_map_->backward(flat_grad_pos, flat_grad_vel, grad_thrust, grad_quat,
                        flat_grad_omega, grad_position, grad_velocity,
                        grad_acceleration, grad_jerk, grad_psi, grad_dpsi);
    grad_acceleration += direct_grad_acc;

    return penalty;
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
  static double sqr(double value) { return value * value; }

  bool ready() const {
    return h_polys_ && h_poly_idx_ && flat_map_ &&
           magnitude_bounds_.size() >= 6 && penalty_weights_.size() >= 6;
  }

  double segmentVelocityBound(int seg_idx) const {
    if (piece_velocity_bounds_ &&
        seg_idx >= 0 &&
        seg_idx < piece_velocity_bounds_->size() &&
        std::isfinite((*piece_velocity_bounds_)(seg_idx)) &&
        (*piece_velocity_bounds_)(seg_idx) > 1.0e-3) {
      return (*piece_velocity_bounds_)(seg_idx);
    }
    return magnitude_bounds_.size() > 0 ? magnitude_bounds_(0) : 1.0;
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

  const PolyhedraH *h_polys_{nullptr};
  const Eigen::VectorXi *h_poly_idx_{nullptr};
  const Eigen::VectorXd *piece_velocity_bounds_{nullptr};
  double smooth_eps_{1.0e-3};
  Eigen::VectorXd magnitude_bounds_;
  Eigen::VectorXd penalty_weights_;
  flatness::FlatnessMap *flat_map_{nullptr};
};

}  // namespace cost_functional_manager
}  // namespace traj_opt
}  // namespace fast_planner

#endif
