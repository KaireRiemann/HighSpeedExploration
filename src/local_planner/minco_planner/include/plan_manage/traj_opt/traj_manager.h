#ifndef EPIC_TRAJ_OPT_TRAJ_MANAGER_H
#define EPIC_TRAJ_OPT_TRAJ_MANAGER_H

#include <limits>
#include <vector>

#include <Eigen/Eigen>
#include <gcopter/flatness.hpp>
#include <gcopter/trajectory.hpp>
#include <plan_manage/traj_opt/costfunctional/spatialmap/polytope_spatial_map.hpp>
#include <plan_manage/traj_opt/costfunctional/temporalmap/quad_inv_time_map.hpp>
#include <plan_manage/traj_opt/costfunctional/temporalcosts/balanced_time_cost.hpp>
#include <plan_manage/traj_opt/costfunctional/temporalcosts/target_time_cost.hpp>
#include <plan_manage/traj_opt/costfunctional_manager/epic_integral_cost_manager.hpp>
#include <plan_manage/traj_opt/costfunctional_manager/yaw_integral_cost_manager.hpp>
#include <plan_manage/traj_opt/minco/minco_optimizer.hpp>

namespace fast_planner {
namespace traj_opt {

class TrajManager {
 public:
  using PolyhedronH = Eigen::MatrixX4d;
  using PolyhedraH = std::vector<PolyhedronH>;
  using PolyhedronV = Eigen::Matrix3Xd;
  using PolyhedraV = std::vector<PolyhedronV>;
  using SnapOptimizer =
      epic_minco::MINCOOptimizer<3, 4, temporal_map::QuadInvTimeMap,
                                 spatial_map::PolytopeSpatialMap>;
  using YawOptimizer =
      epic_minco::MINCOOptimizer<3, 3, temporal_map::QuadInvTimeMap,
                                 spatial_map::PolytopeSpatialMap>;
  using SnapTraj = typename SnapOptimizer::TrajType;
  using YawTraj = typename YawOptimizer::TrajType;
  using BoundaryState = typename SnapOptimizer::BoundaryState;
  using YawBoundaryState = typename YawOptimizer::BoundaryState;

  struct Config {
    double max_vel{3.0};
    double length_per_piece{std::numeric_limits<double>::infinity()};
    double smoothing_eps{1.0e-3};
    int integral_resolution{8};
    double rel_cost_tol{1.0e-4};
    double energy_weight{1.0};
    Eigen::VectorXd magnitude_bounds;
    Eigen::VectorXd penalty_weights;
    Eigen::VectorXd physical_params;
  };

  struct Request {
    Eigen::Matrix<double, 3, 4> initial_state;
    Eigen::Matrix<double, 3, 4> terminal_state;
    PolyhedraH safe_corridor;
    Eigen::VectorXd penalty_weights;
    Eigen::VectorXd corridor_velocity_limits;
    double time_weight{1.0};
    double min_total_time{0.0};
  };

  struct YawRequest {
    Eigen::Matrix3d initial_state;
    Eigen::Matrix3d terminal_state;
    Eigen::Matrix3Xd guide_points;
    Eigen::VectorXd time_allocation;
    double target_duration{0.0};
    double guide_weight{0.0};
    double time_weight{0.0};
  };

  TrajManager();

  void setConfig(const Config &config);
  const Config &config() const { return config_; }

  bool optimize(const Request &request, Trajectory<7> &trajectory, double *final_cost = nullptr);
  bool optimizeYaw(const YawRequest &request, Trajectory<5> &trajectory, double *final_cost = nullptr);

 private:
  static double costFunctional(void *ptr, const Eigen::VectorXd &x, Eigen::VectorXd &grad);
  static double yawCostFunctional(void *ptr, const Eigen::VectorXd &x, Eigen::VectorXd &grad);
  double evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &grad);
  double evaluateYaw(const Eigen::VectorXd &x, Eigen::VectorXd &grad);

  bool setupProblem(const Request &request);
  bool processCorridor(const PolyhedraH &h_polys, PolyhedraV &v_polys) const;
  void getShortestPath(const Eigen::Vector3d &start,
                       const Eigen::Vector3d &goal,
                       const PolyhedraV &v_polys,
                       Eigen::Matrix3Xd &path) const;
  void setInitial(const Eigen::Matrix3Xd &path,
                  const Eigen::VectorXi &interval_counts,
                  Eigen::Matrix3Xd &inner_points,
                  Eigen::VectorXd &time_alloc) const;
  void buildWaypoints(typename SnapOptimizer::WaypointsType &waypoints) const;
  static BoundaryState toBoundaryState(const Eigen::Matrix<double, 3, 4> &state);
  static YawBoundaryState toYawBoundaryState(const Eigen::Matrix3d &state);
  static Trajectory<7> toEpicTrajectory(const SnapTraj &traj);
  static Trajectory<5> toEpicYawTrajectory(const YawTraj &traj);

  Config config_;
  Request active_request_;

  PolyhedraH h_polytopes_;
  PolyhedraV v_polytopes_;
  Eigen::VectorXi piece_idx_;
  Eigen::VectorXi v_poly_idx_;
  Eigen::VectorXi h_poly_idx_;
  Eigen::VectorXd piece_velocity_limits_;
  Eigen::Matrix3Xd short_path_;
  Eigen::Matrix3Xd inner_points_;
  Eigen::VectorXd times_;
  int piece_num_{0};

  temporal_map::QuadInvTimeMap time_map_;
  spatial_map::PolytopeSpatialMap spatial_map_;
  cost_functional::BalancedTimeCost time_cost_;
  cost_functional::TargetTimeCost yaw_time_cost_;
  cost_functional_manager::EpicIntegralCostManager cost_manager_;
  cost_functional_manager::YawIntegralCostManager yaw_cost_manager_;
  flatness::FlatnessMap flat_map_;
  SnapOptimizer optimizer_;
  spatial_map::PolytopeSpatialMap yaw_spatial_map_;
  YawOptimizer yaw_optimizer_;
};

}  // namespace traj_opt
}  // namespace fast_planner

#endif
