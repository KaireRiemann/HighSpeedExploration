#ifndef EPIC_TRAJ_OPT_POLYTOPE_SPATIAL_MAP_HPP
#define EPIC_TRAJ_OPT_POLYTOPE_SPATIAL_MAP_HPP

#include <cfloat>
#include <cmath>
#include <vector>

#include <Eigen/Eigen>
#include <gcopter/lbfgs.hpp>

namespace fast_planner {
namespace traj_opt {
namespace spatial_map {

class PolytopeSpatialMap {
 public:
  using VectorType = Eigen::Vector3d;
  using PolyhedronV = Eigen::Matrix3Xd;
  using PolyhedraV = std::vector<PolyhedronV>;

  void reset(const PolyhedraV *polys,
             const Eigen::VectorXi *indices,
             int segments,
             bool identity = false) {
    v_polys_ = polys;
    v_poly_idx_ = indices;
    num_segments_ = segments;
    identity_mode_ = identity;
  }

  int getUnconstrainedDim(int index) const {
    if (identityMode(index)) {
      return 3;
    }
    return (*v_polys_)[(*v_poly_idx_)(index - 1)].cols();
  }

  VectorType toPhysical(const Eigen::VectorXd &xi, int index) const {
    if (identityMode(index)) {
      return xi.head<3>();
    }

    const int poly_id = (*v_poly_idx_)(index - 1);
    const auto &poly = (*v_polys_)[poly_id];
    const int k = poly.cols();
    const double norm = xi.norm();
    if (norm < 1.0e-12) {
      return poly.col(0);
    }

    const Eigen::VectorXd r = (xi / norm).head(k - 1);
    return poly.rightCols(k - 1) * r.cwiseProduct(r) + poly.col(0);
  }

  Eigen::VectorXd toUnconstrained(const VectorType &point, int index) const {
    if (identityMode(index)) {
      return point;
    }

    const int poly_id = (*v_poly_idx_)(index - 1);
    const auto &poly = (*v_polys_)[poly_id];
    const int k = poly.cols();

    Eigen::Matrix3Xd ov_poly(3, k + 1);
    ov_poly.col(0) = point;
    ov_poly.rightCols(k) = poly;

    Eigen::VectorXd xi(k);
    xi.setConstant(std::sqrt(1.0 / static_cast<double>(k)));

    double min_sqr_d = 0.0;
    lbfgs::lbfgs_parameter_t params;
    params.past = 0;
    params.delta = 1.0e-5;
    params.g_epsilon = FLT_EPSILON;
    params.max_iterations = 128;
    lbfgs::lbfgs_optimize(xi, min_sqr_d, &PolytopeSpatialMap::costTinyNLS,
                          nullptr, nullptr, &ov_poly, params);
    return xi;
  }

  Eigen::VectorXd backwardGrad(const Eigen::VectorXd &xi,
                               const VectorType &grad_p,
                               int index) const {
    if (identityMode(index)) {
      return grad_p;
    }

    const int poly_id = (*v_poly_idx_)(index - 1);
    const auto &poly = (*v_polys_)[poly_id];
    const int k = poly.cols();
    Eigen::VectorXd grad_xi = Eigen::VectorXd::Zero(k);
    const double norm = xi.norm();
    if (norm < 1.0e-12) {
      return grad_xi;
    }

    const double norm_inv = 1.0 / norm;
    const Eigen::VectorXd unit_q = xi * norm_inv;
    Eigen::VectorXd grad_q(k);
    grad_q.head(k - 1) =
        (poly.rightCols(k - 1).transpose() * grad_p).array() *
        unit_q.head(k - 1).array() * 2.0;
    grad_q(k - 1) = 0.0;
    grad_xi = (grad_q - unit_q * unit_q.dot(grad_q)) * norm_inv;
    return grad_xi;
  }

  void addNormPenalty(const Eigen::VectorXd &xi,
                      double &cost,
                      Eigen::VectorXd &grad_xi) const {
    if (identity_mode_ || !v_polys_ || !v_poly_idx_ || xi.size() <= 0) {
      return;
    }

    const double sqr_norm_q = xi.squaredNorm();
    const double sqr_norm_violation = sqr_norm_q - 1.0;
    if (sqr_norm_violation > 0.0) {
      double c = sqr_norm_violation * sqr_norm_violation;
      const double dc = 3.0 * c;
      c *= sqr_norm_violation;
      cost += c;
      grad_xi += dc * 2.0 * xi;
    }
  }

 private:
  bool identityMode(int index) const {
    return identity_mode_ || !v_polys_ || !v_poly_idx_ ||
           index <= 0 || index > num_segments_;
  }

  static double costTinyNLS(void *ptr, const Eigen::VectorXd &xi, Eigen::VectorXd &grad_xi) {
    const int n = xi.size();
    const Eigen::Matrix3Xd &ov_poly = *(Eigen::Matrix3Xd *)ptr;
    const double sqr_norm_xi = xi.squaredNorm();
    if (sqr_norm_xi < 1.0e-12) {
      grad_xi.setZero();
      return 0.0;
    }

    const double inv_norm_xi = 1.0 / std::sqrt(sqr_norm_xi);
    const Eigen::VectorXd unit_xi = xi * inv_norm_xi;
    const Eigen::VectorXd r = unit_xi.head(n - 1);
    const Eigen::Vector3d delta =
        ov_poly.rightCols(n - 1) * r.cwiseProduct(r) + ov_poly.col(1) - ov_poly.col(0);

    double cost = delta.squaredNorm();
    grad_xi.head(n - 1) =
        (ov_poly.rightCols(n - 1).transpose() * (2.0 * delta)).array() *
        r.array() * 2.0;
    grad_xi(n - 1) = 0.0;
    grad_xi = (grad_xi - unit_xi.dot(grad_xi) * unit_xi).eval() * inv_norm_xi;

    const double sqr_norm_violation = sqr_norm_xi - 1.0;
    if (sqr_norm_violation > 0.0) {
      double c = sqr_norm_violation * sqr_norm_violation;
      const double dc = 3.0 * c;
      c *= sqr_norm_violation;
      cost += c;
      grad_xi += dc * 2.0 * xi;
    }
    return cost;
  }

  const PolyhedraV *v_polys_{nullptr};
  const Eigen::VectorXi *v_poly_idx_{nullptr};
  int num_segments_{0};
  bool identity_mode_{false};
};

}  // namespace spatial_map
}  // namespace traj_opt
}  // namespace fast_planner

#endif
