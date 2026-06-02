#include <plan_manage/traj_opt/traj_manager.h>

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <iostream>

#include <gcopter/geo_utils.hpp>
#include <gcopter/lbfgs.hpp>

namespace fast_planner {
namespace traj_opt {
namespace {

struct ShortestPathContext {
  double smooth_eps{1.0e-3};
  const Eigen::Vector3d *start{nullptr};
  const Eigen::Vector3d *goal{nullptr};
  const TrajManager::PolyhedraV *v_polys{nullptr};
};

double costDistance(void *ptr, const Eigen::VectorXd &xi, Eigen::VectorXd &grad_xi) {
  const auto &ctx = *static_cast<ShortestPathContext *>(ptr);
  const int overlaps = static_cast<int>(ctx.v_polys->size()) / 2;

  double cost = 0.0;
  Eigen::Matrix3Xd grad_p = Eigen::Matrix3Xd::Zero(3, overlaps);
  Eigen::Vector3d a;
  Eigen::Vector3d b;
  Eigen::Vector3d d;
  Eigen::VectorXd r;

  for (int i = 0, j = 0, k = 0; i <= overlaps; ++i, j += k) {
    a = i == 0 ? *ctx.start : b;
    if (i < overlaps) {
      k = (*ctx.v_polys)[2 * i + 1].cols();
      Eigen::Map<const Eigen::VectorXd> q(xi.data() + j, k);
      r = q.normalized().head(k - 1);
      b = (*ctx.v_polys)[2 * i + 1].rightCols(k - 1) * r.cwiseProduct(r) +
          (*ctx.v_polys)[2 * i + 1].col(0);
    } else {
      b = *ctx.goal;
    }

    d = b - a;
    const double smoothed_dist = std::sqrt(d.squaredNorm() + ctx.smooth_eps);
    cost += smoothed_dist;

    if (i < overlaps) {
      grad_p.col(i) += d / smoothed_dist;
    }
    if (i > 0) {
      grad_p.col(i - 1) -= d / smoothed_dist;
    }
  }

  grad_xi.setZero(xi.size());
  for (int i = 0, j = 0, k = 0; i < overlaps; ++i, j += k) {
    k = (*ctx.v_polys)[2 * i + 1].cols();
    Eigen::Map<const Eigen::VectorXd> q(xi.data() + j, k);
    Eigen::Map<Eigen::VectorXd> grad_q(grad_xi.data() + j, k);
    const double norm = q.norm();
    if (norm < 1.0e-12) {
      continue;
    }

    const double norm_inv = 1.0 / norm;
    const Eigen::VectorXd unit_q = q * norm_inv;
    Eigen::VectorXd raw_grad(k);
    raw_grad.head(k - 1) =
        ((*ctx.v_polys)[2 * i + 1].rightCols(k - 1).transpose() * grad_p.col(i)).array() *
        unit_q.head(k - 1).array() * 2.0;
    raw_grad(k - 1) = 0.0;
    grad_q = (raw_grad - unit_q * unit_q.dot(raw_grad)) * norm_inv;

    const double sqr_norm_violation = q.squaredNorm() - 1.0;
    if (sqr_norm_violation > 0.0) {
      double c = sqr_norm_violation * sqr_norm_violation;
      const double dc = 3.0 * c;
      c *= sqr_norm_violation;
      cost += c;
      grad_q += dc * 2.0 * q;
    }
  }

  return cost;
}

bool isRecoverableLbfgsReturn(int ret) {
  return ret == lbfgs::LBFGSERR_MAXIMUMITERATION ||
         ret == lbfgs::LBFGSERR_MAXIMUMLINESEARCH ||
         ret == lbfgs::LBFGSERR_MINIMUMSTEP ||
         ret == lbfgs::LBFGSERR_WIDTHTOOSMALL;
}

}  // namespace

TrajManager::TrajManager() {
  config_.magnitude_bounds.resize(6);
  config_.magnitude_bounds << 3.0, 6.0, 3.0, 0.7, 5.0, 20.0;
  config_.penalty_weights.resize(6);
  config_.penalty_weights << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
  config_.physical_params.resize(6);
  config_.physical_params << 1.0, 9.81, 0.0, 0.0, 0.0, 1.0e-3;
}

void TrajManager::setConfig(const Config &config) {
  config_ = config;
  config_.integral_resolution = std::max(1, config_.integral_resolution);
  if (config_.magnitude_bounds.size() != 6) {
    config_.magnitude_bounds.resize(6);
    config_.magnitude_bounds << config_.max_vel, 6.0, 3.0, 0.7, 5.0, 20.0;
  }
  if (config_.penalty_weights.size() != 6) {
    config_.penalty_weights.resize(6);
    config_.penalty_weights.setOnes();
  }
  if (config_.physical_params.size() != 6) {
    config_.physical_params.resize(6);
    config_.physical_params << 1.0, 9.81, 0.0, 0.0, 0.0, 1.0e-3;
  }
}

bool TrajManager::optimize(const Request &request, Trajectory<7> &trajectory, double *final_cost) {
  trajectory.clear();
  if (!setupProblem(request)) {
    return false;
  }

  typename SnapOptimizer::WaypointsType waypoints;
  buildWaypoints(waypoints);

  std::vector<double> time_segments(static_cast<std::size_t>(times_.size()));
  for (int i = 0; i < times_.size(); ++i) {
    time_segments[static_cast<std::size_t>(i)] = times_(i);
  }

  optimizer_.setTimeMap(&time_map_);
  optimizer_.setSpatialMap(&spatial_map_);
  optimizer_.setEnergyWeight(config_.energy_weight);
  optimizer_.setSamplesPerPiece(config_.integral_resolution);
  if (!optimizer_.setInitState(time_segments, waypoints,
                               toBoundaryState(active_request_.initial_state),
                               toBoundaryState(active_request_.terminal_state))) {
    return false;
  }

  Eigen::VectorXd x = optimizer_.generateInitialGuess();
  if (x.size() <= 0 || !x.allFinite()) {
    return false;
  }

  flat_map_.reset(config_.physical_params(0), config_.physical_params(1),
                  config_.physical_params(2), config_.physical_params(3),
                  config_.physical_params(4), config_.physical_params(5));

  time_cost_.linear_weight = std::max(0.0, active_request_.time_weight);
  time_cost_.lower_bound = std::max(0.0, active_request_.min_total_time);
  time_cost_.lower_bound_weight = std::max(0.0, active_request_.time_weight);
  time_cost_.smooth_eps = std::max(1.0e-6, config_.smoothing_eps);
  cost_manager_.reset(&h_polytopes_, &h_poly_idx_, &piece_velocity_limits_,
                      config_.smoothing_eps,
                      config_.magnitude_bounds, active_request_.penalty_weights,
                      &flat_map_);

  lbfgs::lbfgs_parameter_t params;
  params.mem_size = 256;
  params.past = 3;
  params.min_step = 1.0e-20;
  params.g_epsilon = FLT_EPSILON;
  params.delta = std::max(1.0e-7, config_.rel_cost_tol);

  double min_cost = 0.0;
  const int ret = lbfgs::lbfgs_optimize(x, min_cost, &TrajManager::costFunctional,
                                        nullptr, nullptr, this, params);
  if (ret < 0 && !isRecoverableLbfgsReturn(ret)) {
    std::cout << "Optimization Failed: " << lbfgs::lbfgs_strerror(ret) << std::endl;
    return false;
  }

  Eigen::VectorXd grad(x.size());
  min_cost = evaluate(x, grad);
  if (!std::isfinite(min_cost)) {
    return false;
  }

  trajectory = toEpicTrajectory(optimizer_.getTrajectory());
  if (final_cost) {
    *final_cost = min_cost;
  }
  return trajectory.getPieceNum() > 0;
}

bool TrajManager::optimizeYaw(const YawRequest &request,
                              Trajectory<5> &trajectory,
                              double *final_cost) {
  trajectory.clear();

  const int piece_num = static_cast<int>(request.time_allocation.size());
  if (piece_num <= 0 ||
      request.guide_points.cols() != std::max(0, piece_num - 1) ||
      (request.time_allocation.array() <= 1.0e-6).any() ||
      !request.time_allocation.allFinite()) {
    return false;
  }

  typename YawOptimizer::WaypointsType waypoints(piece_num + 1, 3);
  waypoints.row(0) = request.initial_state.col(0).transpose();
  for (int i = 0; i < request.guide_points.cols(); ++i) {
    waypoints.row(i + 1) = request.guide_points.col(i).transpose();
  }
  waypoints.row(piece_num) = request.terminal_state.col(0).transpose();

  std::vector<double> time_segments(static_cast<std::size_t>(piece_num));
  for (int i = 0; i < piece_num; ++i) {
    time_segments[static_cast<std::size_t>(i)] = request.time_allocation(i);
  }

  yaw_spatial_map_.reset(nullptr, nullptr, piece_num, true);
  yaw_optimizer_.setTimeMap(&time_map_);
  yaw_optimizer_.setSpatialMap(&yaw_spatial_map_);
  yaw_optimizer_.setEnergyWeight(1.0);
  yaw_optimizer_.setSamplesPerPiece(config_.integral_resolution);
  if (!yaw_optimizer_.setInitState(time_segments, waypoints,
                                   toYawBoundaryState(request.initial_state),
                                   toYawBoundaryState(request.terminal_state))) {
    return false;
  }

  Eigen::VectorXd x = yaw_optimizer_.generateInitialGuess();
  if (x.size() <= 0 || !x.allFinite()) {
    return false;
  }

  yaw_time_cost_.target_duration = request.target_duration > 0.0
                                       ? request.target_duration
                                       : request.time_allocation.sum();
  yaw_time_cost_.weight = std::max(0.0, request.time_weight);
  yaw_time_cost_.smooth_eps = std::max(1.0e-6, config_.smoothing_eps);
  yaw_cost_manager_.reset(&request.guide_points, request.initial_state.col(0),
                          request.terminal_state.col(0), config_.integral_resolution,
                          config_.smoothing_eps, request.guide_weight);

  lbfgs::lbfgs_parameter_t params;
  params.mem_size = 256;
  params.past = 3;
  params.min_step = 1.0e-20;
  params.g_epsilon = 1.0e-5;
  params.delta = std::max(1.0e-7, config_.rel_cost_tol);

  double min_cost = 0.0;
  const int ret = lbfgs::lbfgs_optimize(x, min_cost, &TrajManager::yawCostFunctional,
                                        nullptr, nullptr, this, params);
  if (ret < 0 && !isRecoverableLbfgsReturn(ret)) {
    std::cout << "Yaw Optimization Failed: " << lbfgs::lbfgs_strerror(ret) << std::endl;
    return false;
  }

  Eigen::VectorXd grad(x.size());
  min_cost = evaluateYaw(x, grad);
  if (!std::isfinite(min_cost)) {
    return false;
  }

  trajectory = toEpicYawTrajectory(yaw_optimizer_.getTrajectory());
  if (final_cost) {
    *final_cost = min_cost;
  }
  return trajectory.getPieceNum() > 0;
}

double TrajManager::costFunctional(void *ptr, const Eigen::VectorXd &x, Eigen::VectorXd &grad) {
  return static_cast<TrajManager *>(ptr)->evaluate(x, grad);
}

double TrajManager::yawCostFunctional(void *ptr, const Eigen::VectorXd &x, Eigen::VectorXd &grad) {
  return static_cast<TrajManager *>(ptr)->evaluateYaw(x, grad);
}

double TrajManager::evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &grad) {
  return optimizer_.evaluate(x, grad, time_cost_, cost_manager_);
}

double TrajManager::evaluateYaw(const Eigen::VectorXd &x, Eigen::VectorXd &grad) {
  return yaw_optimizer_.evaluate(x, grad, yaw_time_cost_, yaw_cost_manager_);
}

bool TrajManager::setupProblem(const Request &request) {
  active_request_ = request;
  if (active_request_.safe_corridor.empty()) {
    return false;
  }
  if (active_request_.penalty_weights.size() != 6) {
    active_request_.penalty_weights = config_.penalty_weights;
  }

  h_polytopes_ = active_request_.safe_corridor;
  for (auto &poly : h_polytopes_) {
    if (poly.cols() != 4 || poly.rows() <= 0) {
      return false;
    }
    const Eigen::ArrayXd norms = poly.leftCols<3>().rowwise().norm();
    for (int r = 0; r < poly.rows(); ++r) {
      const double norm = std::max(1.0e-9, norms(r));
      poly.row(r) /= norm;
    }
  }

  if (!processCorridor(h_polytopes_, v_polytopes_)) {
    return false;
  }

  getShortestPath(active_request_.initial_state.col(0),
                  active_request_.terminal_state.col(0),
                  v_polytopes_, short_path_);
  if (short_path_.cols() < 2) {
    return false;
  }

  const int poly_num = static_cast<int>(h_polytopes_.size());
  const Eigen::Matrix3Xd deltas = short_path_.rightCols(poly_num) -
                                  short_path_.leftCols(poly_num);
  piece_idx_.resize(poly_num);
  const double length_per_piece =
      std::isfinite(config_.length_per_piece) && config_.length_per_piece > 1.0e-6
          ? config_.length_per_piece
          : std::numeric_limits<double>::infinity();
  for (int i = 0; i < poly_num; ++i) {
    piece_idx_(i) = static_cast<int>(deltas.col(i).norm() / length_per_piece) + 1;
    piece_idx_(i) = std::max(1, piece_idx_(i));
  }
  piece_num_ = piece_idx_.sum();
  if (piece_num_ <= 0) {
    return false;
  }

  v_poly_idx_.resize(piece_num_ - 1);
  h_poly_idx_.resize(piece_num_);
  int waypoint_cursor = 0;
  for (int poly_i = 0, piece_cursor = 0; poly_i < poly_num; ++poly_i) {
    const int pieces_in_poly = piece_idx_(poly_i);
    for (int local_i = 0; local_i < pieces_in_poly; ++local_i, ++piece_cursor) {
      if (local_i < pieces_in_poly - 1) {
        v_poly_idx_(waypoint_cursor++) = 2 * poly_i;
      } else if (poly_i < poly_num - 1) {
        v_poly_idx_(waypoint_cursor++) = 2 * poly_i + 1;
      }
      h_poly_idx_(piece_cursor) = poly_i;
    }
  }

  piece_velocity_limits_.resize(piece_num_);
  if (active_request_.corridor_velocity_limits.size() == poly_num) {
    for (int i = 0; i < piece_num_; ++i) {
      const int h_idx = std::clamp(h_poly_idx_(i), 0, poly_num - 1);
      piece_velocity_limits_(i) = active_request_.corridor_velocity_limits(h_idx);
    }
  } else if (active_request_.corridor_velocity_limits.size() == piece_num_) {
    piece_velocity_limits_ = active_request_.corridor_velocity_limits;
  } else {
    piece_velocity_limits_.setConstant(config_.max_vel);
  }
  for (int i = 0; i < piece_velocity_limits_.size(); ++i) {
    if (!std::isfinite(piece_velocity_limits_(i)) || piece_velocity_limits_(i) <= 1.0e-3) {
      piece_velocity_limits_(i) = config_.max_vel;
    }
  }

  setInitial(short_path_, piece_idx_, inner_points_, times_);
  if (times_.size() != piece_num_ ||
      (times_.array() <= 1.0e-6).any() ||
      !times_.allFinite()) {
    return false;
  }

  spatial_map_.reset(&v_polytopes_, &v_poly_idx_, piece_num_);
  return true;
}

bool TrajManager::processCorridor(const PolyhedraH &h_polys, PolyhedraV &v_polys) const {
  const int size_corridor = static_cast<int>(h_polys.size()) - 1;
  if (size_corridor < 0) {
    return false;
  }

  v_polys.clear();
  v_polys.reserve(2 * size_corridor + 1);

  PolyhedronH overlap_h;
  PolyhedronV vertices;
  PolyhedronV local_vertices;
  for (int i = 0; i < size_corridor; ++i) {
    if (!geo_utils::enumerateVs(h_polys[i], vertices)) {
      return false;
    }
    const int nv = vertices.cols();
    local_vertices.resize(3, nv);
    local_vertices.col(0) = vertices.col(0);
    local_vertices.rightCols(nv - 1) = vertices.rightCols(nv - 1).colwise() - vertices.col(0);
    v_polys.push_back(local_vertices);

    overlap_h.resize(h_polys[i].rows() + h_polys[i + 1].rows(), 4);
    overlap_h.topRows(h_polys[i].rows()) = h_polys[i];
    overlap_h.bottomRows(h_polys[i + 1].rows()) = h_polys[i + 1];
    if (!geo_utils::enumerateVs(overlap_h, vertices)) {
      return false;
    }
    const int overlap_nv = vertices.cols();
    local_vertices.resize(3, overlap_nv);
    local_vertices.col(0) = vertices.col(0);
    local_vertices.rightCols(overlap_nv - 1) =
        vertices.rightCols(overlap_nv - 1).colwise() - vertices.col(0);
    v_polys.push_back(local_vertices);
  }

  if (!geo_utils::enumerateVs(h_polys.back(), vertices)) {
    return false;
  }
  const int nv = vertices.cols();
  local_vertices.resize(3, nv);
  local_vertices.col(0) = vertices.col(0);
  local_vertices.rightCols(nv - 1) = vertices.rightCols(nv - 1).colwise() - vertices.col(0);
  v_polys.push_back(local_vertices);
  return true;
}

void TrajManager::getShortestPath(const Eigen::Vector3d &start,
                                  const Eigen::Vector3d &goal,
                                  const PolyhedraV &v_polys,
                                  Eigen::Matrix3Xd &path) const {
  const int overlaps = static_cast<int>(v_polys.size()) / 2;
  Eigen::VectorXi v_sizes(overlaps);
  for (int i = 0; i < overlaps; ++i) {
    v_sizes(i) = v_polys[2 * i + 1].cols();
  }

  Eigen::VectorXd xi(v_sizes.sum());
  for (int i = 0, j = 0; i < overlaps; ++i) {
    xi.segment(j, v_sizes(i)).setConstant(std::sqrt(1.0 / v_sizes(i)));
    j += v_sizes(i);
  }

  ShortestPathContext ctx;
  ctx.smooth_eps = std::max(1.0e-6, config_.smoothing_eps);
  ctx.start = &start;
  ctx.goal = &goal;
  ctx.v_polys = &v_polys;

  double min_distance = 0.0;
  lbfgs::lbfgs_parameter_t params;
  params.past = 3;
  params.delta = 1.0e-3;
  params.g_epsilon = 1.0e-5;
  lbfgs::lbfgs_optimize(xi, min_distance, &costDistance, nullptr, nullptr, &ctx, params);

  path.resize(3, overlaps + 2);
  path.leftCols<1>() = start;
  path.rightCols<1>() = goal;
  for (int i = 0, j = 0, k = 0; i < overlaps; ++i, j += k) {
    k = v_polys[2 * i + 1].cols();
    Eigen::Map<const Eigen::VectorXd> q(xi.data() + j, k);
    const Eigen::VectorXd r = q.normalized().head(k - 1);
    path.col(i + 1) =
        v_polys[2 * i + 1].rightCols(k - 1) * r.cwiseProduct(r) +
        v_polys[2 * i + 1].col(0);
  }
}

void TrajManager::setInitial(const Eigen::Matrix3Xd &path,
                             const Eigen::VectorXi &interval_counts,
                             Eigen::Matrix3Xd &inner_points,
                             Eigen::VectorXd &time_alloc) const {
  const int segment_count = interval_counts.size();
  const int piece_count = interval_counts.sum();
  inner_points.resize(3, std::max(0, piece_count - 1));
  time_alloc.resize(piece_count);

  const double speed = std::max(1.0e-2, config_.max_vel * 3.0);
  for (int i = 0, time_cursor = 0, point_cursor = 0; i < segment_count; ++i) {
    const int pieces_in_segment = std::max(1, interval_counts(i));
    const Eigen::Vector3d start = path.col(i);
    const Eigen::Vector3d goal = path.col(i + 1);
    const Eigen::Vector3d step = (goal - start) / static_cast<double>(pieces_in_segment);
    time_alloc.segment(time_cursor, pieces_in_segment)
        .setConstant(std::max(0.01, step.norm() / speed));
    time_cursor += pieces_in_segment;

    for (int j = 0; j < pieces_in_segment; ++j) {
      if (i > 0 || j > 0) {
        inner_points.col(point_cursor++) = start + step * static_cast<double>(j);
      }
    }
  }
}

void TrajManager::buildWaypoints(typename SnapOptimizer::WaypointsType &waypoints) const {
  waypoints.resize(piece_num_ + 1, 3);
  waypoints.row(0) = active_request_.initial_state.col(0).transpose();
  for (int i = 0; i < inner_points_.cols(); ++i) {
    waypoints.row(i + 1) = inner_points_.col(i).transpose();
  }
  waypoints.row(piece_num_) = active_request_.terminal_state.col(0).transpose();
}

TrajManager::BoundaryState TrajManager::toBoundaryState(
    const Eigen::Matrix<double, 3, 4> &state) {
  BoundaryState boundary;
  boundary.col(0) = state.col(0);
  boundary.col(1) = state.col(1);
  boundary.col(2) = state.col(2);
  boundary.col(3) = state.col(3);
  return boundary;
}

TrajManager::YawBoundaryState TrajManager::toYawBoundaryState(const Eigen::Matrix3d &state) {
  YawBoundaryState boundary;
  boundary.col(0) = state.col(0);
  boundary.col(1) = state.col(1);
  boundary.col(2) = state.col(2);
  return boundary;
}

Trajectory<7> TrajManager::toEpicTrajectory(const SnapTraj &traj) {
  Trajectory<7> out;
  const Eigen::VectorXd &durations = traj.getDurations();
  out.reserve(static_cast<int>(durations.size()));
  for (int i = 0; i < durations.size(); ++i) {
    out.emplace_back(durations(i), traj.getPieceCoeffMat(i));
  }
  return out;
}

Trajectory<5> TrajManager::toEpicYawTrajectory(const YawTraj &traj) {
  Trajectory<5> out;
  const Eigen::VectorXd &durations = traj.getDurations();
  out.reserve(static_cast<int>(durations.size()));
  for (int i = 0; i < durations.size(); ++i) {
    out.emplace_back(durations(i), traj.getPieceCoeffMat(i));
  }
  return out;
}

}  // namespace traj_opt
}  // namespace fast_planner
