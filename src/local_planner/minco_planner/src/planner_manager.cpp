/***
 * @Author: ning-zelin && zl.ning@qq.com
 * @Date: 2023-12-28 14:48:50
 * @LastEditTime: 2023-12-30 15:02:15
 * @Description:
 * @
 * @Copyright (c) 2023 by ning-zelin, All Rights Reserved.
 */
// #include <fstream>
#include <algorithm>
#include <cmath>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <plan_manage/planner_manager.h>
#include <plan_manage/traj_opt/traj_manager.h>
#include <std_msgs/Int32.h>
#include <limits>
#include <thread>
#include <visualization_msgs/Marker.h>

namespace fast_planner {
namespace {
Eigen::VectorXd makeMagnitudeBounds(const GcopterConfig &cfg) {
  Eigen::VectorXd bounds(6);
  bounds << cfg.maxVelMag, cfg.maxAccMag, cfg.maxBdrMag, cfg.maxTiltAngle, cfg.minThrust,
      cfg.maxThrust;
  return bounds;
}

Eigen::VectorXd makePenaltyWeights(const GcopterConfig &cfg, bool safe_mode) {
  Eigen::VectorXd weights(6);
  if (cfg.chiVec.size() >= 6) {
    for (int i = 0; i < 6; ++i) {
      weights(i) = cfg.chiVec[i];
    }
  } else {
    weights(0) = cfg.chiVec.size() > 0 ? cfg.chiVec[0] : 1.0;  // position
    weights(1) = cfg.chiVec.size() > 1 ? cfg.chiVec[1] : 1.0;  // velocity
    weights(2) = cfg.chiVec.size() > 1 ? cfg.chiVec[1] : 1.0;  // acceleration
    weights(3) = cfg.chiVec.size() > 2 ? cfg.chiVec[2] : 1.0;  // angular rate
    weights(4) = cfg.chiVec.size() > 3 ? cfg.chiVec[3] : 1.0;  // tilt
    weights(5) = cfg.chiVec.size() > 4 ? cfg.chiVec[4] : 1.0;  // thrust
  }
  if (safe_mode) {
    weights(0) *= 2.0;
    weights.tail<5>() *= 0.5;
  }
  return weights;
}

Eigen::VectorXd makePhysicalParams(const GcopterConfig &cfg) {
  Eigen::VectorXd params(6);
  params << cfg.vehicleMass, cfg.gravAcc, cfg.horizDrag, cfg.vertDrag,
      cfg.parasDrag, cfg.speedEps;
  return params;
}

traj_opt::TrajManager::Config makeTrajOptConfig(const GcopterConfig &cfg) {
  traj_opt::TrajManager::Config opt_cfg;
  opt_cfg.max_vel = cfg.maxVelMag;
  opt_cfg.length_per_piece = std::numeric_limits<double>::infinity();
  opt_cfg.smoothing_eps = cfg.smoothingEps;
  opt_cfg.integral_resolution = cfg.integralIntervs;
  opt_cfg.rel_cost_tol = cfg.relCostTol;
  opt_cfg.energy_weight = std::max(1.0e-4, cfg.energyWeight);
  opt_cfg.magnitude_bounds = makeMagnitudeBounds(cfg);
  opt_cfg.penalty_weights = makePenaltyWeights(cfg, false);
  opt_cfg.physical_params = makePhysicalParams(cfg);
  return opt_cfg;
}

template <int D>
Trajectory<D> extractPrefixTrajectory(const Trajectory<D> &traj, double end_t) {
  Trajectory<D> out;
  if (traj.getPieceNum() <= 0 || !std::isfinite(end_t) || end_t <= 1.0e-6) {
    return out;
  }

  double remaining = std::min(end_t, traj.getTotalDuration());
  out.reserve(traj.getPieceNum());
  for (const auto &piece : traj.pieces) {
    if (remaining <= 1.0e-6) {
      break;
    }
    const double duration = piece.getDuration();
    if (remaining >= duration - 1.0e-6) {
      out.emplace_back(piece);
      remaining -= duration;
    } else {
      out.emplace_back(std::max(1.0e-4, remaining), piece.getCoeffMat());
      break;
    }
  }
  return out;
}

Eigen::MatrixX4d makeBoxHPoly(const Eigen::Vector3d &min_bd,
                              const Eigen::Vector3d &max_bd) {
  Eigen::Matrix<double, 6, 4> hp = Eigen::Matrix<double, 6, 4>::Zero();
  hp(0, 0) = 1.0;
  hp(0, 3) = -max_bd.x();
  hp(1, 0) = -1.0;
  hp(1, 3) = min_bd.x();
  hp(2, 1) = 1.0;
  hp(2, 3) = -max_bd.y();
  hp(3, 1) = -1.0;
  hp(3, 3) = min_bd.y();
  hp(4, 2) = 1.0;
  hp(4, 3) = -max_bd.z();
  hp(5, 2) = -1.0;
  hp(5, 3) = min_bd.z();
  return hp;
}

double estimateHPolyClearance(const Eigen::MatrixX4d &poly) {
  if (poly.cols() != 4 || poly.rows() <= 0) {
    return 0.0;
  }
  Eigen::Vector3d inner;
  if (!geo_utils::findInterior(poly, inner)) {
    return 0.0;
  }

  double clearance = std::numeric_limits<double>::infinity();
  for (int i = 0; i < poly.rows(); ++i) {
    const double norm = std::max(1.0e-6, poly.block<1, 3>(i, 0).norm());
    const double signed_dist =
        (poly.block<1, 3>(i, 0).transpose().dot(inner) + poly(i, 3)) / norm;
    clearance = std::min(clearance, -signed_dist);
  }
  return std::isfinite(clearance) ? std::max(0.0, clearance) : 0.0;
}

Eigen::Vector3d pathHorizontalDirectionAt(const std::vector<Eigen::Vector3d> &path, int idx) {
  if (path.size() < 2) {
    return Eigen::Vector3d::UnitX();
  }
  const int seg = std::clamp(idx, 0, static_cast<int>(path.size()) - 2);
  Eigen::Vector3d dir = path[seg + 1] - path[seg];
  dir.z() = 0.0;
  if (dir.norm() < 1.0e-3) {
    return Eigen::Vector3d::UnitX();
  }
  return dir.normalized();
}

double estimateHorizontalSideClearance(const Eigen::MatrixX4d &poly,
                                       const Eigen::Vector3d &travel_dir,
                                       bool reject_front_back_planes) {
  if (poly.cols() != 4 || poly.rows() <= 0) {
    return std::numeric_limits<double>::infinity();
  }

  Eigen::Vector3d inner;
  if (!geo_utils::findInterior(poly, inner)) {
    return std::numeric_limits<double>::infinity();
  }

  Eigen::Vector3d dir = travel_dir;
  dir.z() = 0.0;
  if (dir.norm() < 1.0e-3) {
    dir = Eigen::Vector3d::UnitX();
  } else {
    dir.normalize();
  }

  double clearance = std::numeric_limits<double>::infinity();
  for (int i = 0; i < poly.rows(); ++i) {
    Eigen::Vector3d normal = poly.block<1, 3>(i, 0).transpose();
    const double normal_norm = normal.norm();
    if (normal_norm < 1.0e-6) {
      continue;
    }
    normal /= normal_norm;
    const double offset = poly(i, 3) / normal_norm;

    const Eigen::Vector2d normal_xy(normal.x(), normal.y());
    const double normal_xy_norm = normal_xy.norm();
    if (normal_xy_norm < 1.0e-4 || std::abs(normal.z()) > 0.75) {
      continue;
    }

    const Eigen::Vector3d normal_xy_dir(normal.x() / normal_xy_norm,
                                        normal.y() / normal_xy_norm, 0.0);
    const double forward_alignment = std::abs(normal_xy_dir.dot(dir));
    if (reject_front_back_planes && forward_alignment > 0.85) {
      continue;
    }

    const double signed_dist = normal.dot(inner) + offset;
    clearance = std::min(clearance, std::max(0.0, -signed_dist));
  }

  return clearance;
}

double estimateLateralHPolyClearance(const Eigen::MatrixX4d &poly,
                                     const Eigen::Vector3d &travel_dir) {
  double clearance = estimateHorizontalSideClearance(poly, travel_dir, true);
  if (std::isfinite(clearance)) {
    return clearance;
  }

  return std::numeric_limits<double>::infinity();
}

double turnAngleAt(const std::vector<Eigen::Vector3d> &path, int idx) {
  if (path.size() < 3) {
    return 0.0;
  }
  const int mid = std::clamp(idx + 1, 1, static_cast<int>(path.size()) - 2);
  Eigen::Vector3d a = path[mid] - path[mid - 1];
  Eigen::Vector3d b = path[mid + 1] - path[mid];
  if (a.norm() < 1.0e-3 || b.norm() < 1.0e-3) {
    return 0.0;
  }
  const double c = std::clamp(a.normalized().dot(b.normalized()), -1.0, 1.0);
  return std::acos(c);
}

double pathLength(const std::vector<Eigen::Vector3d> &path) {
  double len = 0.0;
  for (std::size_t i = 1; i < path.size(); ++i) {
    len += (path[i] - path[i - 1]).norm();
  }
  return len;
}

Eigen::Vector3d terminalDirection(const std::vector<Eigen::Vector3d> &path) {
  for (int i = static_cast<int>(path.size()) - 1; i > 0; --i) {
    Eigen::Vector3d dir = path[static_cast<std::size_t>(i)] -
                          path[static_cast<std::size_t>(i - 1)];
    if (dir.norm() > 1.0e-3) {
      return dir.normalized();
    }
  }
  return Eigen::Vector3d::UnitX();
}

template <int D>
double estimatePeakSpeed(const Trajectory<D> &traj, double dt) {
  if (traj.getPieceNum() <= 0) {
    return 0.0;
  }
  const double duration = traj.getTotalDuration();
  if (!std::isfinite(duration) || duration <= 1.0e-6) {
    return 0.0;
  }

  double peak = 0.0;
  const double step = std::max(0.02, dt);
  for (double t = 0.0; t <= duration + 1.0e-6; t += step) {
    peak = std::max(peak, traj.getVel(std::min(t, duration)).norm());
  }
  return peak;
}

struct PeakSpeedInfo {
  double speed{0.0};
  double time{0.0};
};

template <int D>
PeakSpeedInfo estimatePeakSpeedInfo(const Trajectory<D> &traj, double dt,
                                    double start_t = 0.0,
                                    double end_t = std::numeric_limits<double>::infinity()) {
  PeakSpeedInfo info;
  if (traj.getPieceNum() <= 0) {
    return info;
  }
  const double duration = traj.getTotalDuration();
  if (!std::isfinite(duration) || duration <= 1.0e-6) {
    return info;
  }

  double begin = std::clamp(start_t, 0.0, duration);
  double end = std::clamp(end_t, begin, duration);
  const double step = std::max(0.02, dt);
  for (double t = begin; t <= end + 1.0e-6; t += step) {
    const double sample_t = std::min(t, end);
    const double speed = traj.getVel(sample_t).norm();
    if (speed > info.speed) {
      info.speed = speed;
      info.time = sample_t;
    }
  }
  return info;
}
}  // namespace

// SECTION interfaces for setup and query

FastPlannerManager::FastPlannerManager() {}

FastPlannerManager::~FastPlannerManager() {
  lidar_map_interface_.reset();
  gcopter_viz_.reset();
  std::cout << "des manager" << std::endl;
}

void FastPlannerManager::printTimeCost(double time_threhold, double time_cost,
                                       string printInfo) {
  if (time_cost > time_threhold) {
    std::cout << "\033[31m " << printInfo << time_cost << " ms" << "\033[0m"
              << std::endl;
  } else {
    std::cout << "\033[32m " << printInfo << time_cost << " ms" << "\033[0m"
              << std::endl;
  }
}

void FastPlannerManager::initPlanModules(
    ros::NodeHandle &nh, ParallelBubbleAstar::Ptr &parallel_path_finder,
    TopoGraph::Ptr &graph) {

  local_data_.traj_id_ = 0;

  lidar_map_interface_ = graph->lidar_map_interface_;
  nh.getParam("max_traj_len", max_traj_len_);
  nh.getParam("lidar_perception/max_ray_length", max_ray_length);
  nh.getParam("lidar_perception/fov_up", fov_up);
  nh.getParam("lidar_perception/fov_down", fov_down);
  nh.getParam("lidar_perception/lidar_pitch", lidar_pitch);

  gcopter_viz_.reset(new Visualizer);
  gcopter_viz_->init(nh);
  gcopter_config_.reset(new GcopterConfig);
  gcopter_config_->init(nh);
  map_manager_.reset(new general_planner::MapManager);
  map_manager_->setEpicLioMap(lidar_map_interface_);
  if (gcopter_config_->rogMapEnable) {
    if (gcopter_config_->rogMapConfigPath.empty()) {
      ROS_WARN("RogMapEnable=true but RogMapConfigPath is empty; ROG map disabled.");
    } else {
      try {
        auto rog_map = std::make_shared<rog_map::ROGMapROS>(
            nh, gcopter_config_->rogMapConfigPath);
        map_manager_->setMap(rog_map);
        ROS_INFO_STREAM("ROG map enabled with config: "
                        << gcopter_config_->rogMapConfigPath);
      } catch (const std::exception &e) {
        ROS_ERROR_STREAM("Failed to initialize ROG map: " << e.what());
      }
    }
  }
  traj_manager_.reset(new traj_opt::TrajManager);
  traj_manager_->setConfig(makeTrajOptConfig(*gcopter_config_));

  graph_visualizer_.reset(new GraphVisualizer);
  graph_visualizer_->init(nh);
  bubble_path_finder_.reset(new BubbleAstar);
  bubble_path_finder_->init(nh, lidar_map_interface_);
  topo_graph_ = graph;

  parallel_path_finder_ = parallel_path_finder;
  fast_searcher_.reset(new FastSearcher);
  fast_searcher_->init(topo_graph_, bubble_path_finder_);

  pos_sub = nh.subscribe("/quad_0/lidar_slam/odom", 10,
                         &FastPlannerManager::posCallback, this);
  goal_sub = nh.subscribe("/move_base_simple/goal", 10,
                          &FastPlannerManager::goalCallback, this);
  yaw_state_pub = nh.advertise<std_msgs::Int32>("/quad_0/yaw_state", 10);
}

// test_gs
void FastPlannerManager::posCallback(const nav_msgs::OdometryConstPtr &msg) {

  // 提取四元数
  double roll, pitch;
  tf::Quaternion quat;
  tf::quaternionMsgToTF(msg->pose.pose.orientation, quat);

  // 将四元数转换为Euler角
  tf::Matrix3x3(quat).getRPY(roll, pitch, local_data_.curr_yaw_);
}

void FastPlannerManager::goalCallback(
    const geometry_msgs::PoseStampedConstPtr &msg) {
  // 提取四元数
  double roll, pitch;
  tf::Quaternion quat;
  tf::quaternionMsgToTF(msg->pose.orientation, quat);

  // 将四元数转换为Euler角
  tf::Matrix3x3(quat).getRPY(roll, pitch, local_data_.end_yaw_);
}

bool FastPlannerManager::checkTrajVelocity() {
  auto traj = local_data_.minco_traj_;
  if (traj.getPieceNum() <= 0 || !gcopter_config_) {
    return true;
  }
  double duration = local_data_.duration_;
  if (!std::isfinite(duration) || duration <= 1.0e-6) {
    return true;
  }
  double curr_time = (ros::Time::now() - local_data_.start_time_).toSec();
  if (!std::isfinite(curr_time)) {
    return true;
  }
  curr_time = std::max(0.0, curr_time);
  while (curr_time < duration) {
    Vector3d curr_vel = traj.getVel(curr_time);
    if (curr_vel.norm() > gcopter_config_->maxVelMag + 1.0) {
      return false;
    }
    Vector3d curr_acc = traj.getAcc(curr_time);
    if (curr_acc.norm() > gcopter_config_->maxAccMag + 3.0) {
      return false;
    }
    curr_time += 0.3;
  }
  return true;
}

bool FastPlannerManager::checkTrajCollision(double &collision_time) {
  collision_time = 0.0;
  if (!lidar_map_interface_ || !gcopter_config_) {
    return true;
  }

  PointType target;
  PointVector nearest_point;
  vector<float> PointDist;

  auto traj = local_data_.minco_traj_;
  if (traj.getPieceNum() <= 0) {
    return true;
  }

  double duration = local_data_.duration_;
  if (!std::isfinite(duration) || duration <= 1.0e-6) {
    return true;
  }

  double curr_time = (ros::Time::now() - local_data_.start_time_).toSec();
  if (!std::isfinite(curr_time)) {
    return true;
  }
  curr_time = std::max(0.0, curr_time);

  Vector3d last_sphere_cen_;
  if (curr_time > duration) {
    collision_time = duration;
    return true;
  }

  last_sphere_cen_ = traj.getPos(curr_time);
  double last_radius_ = lidar_map_interface_->getDisToOcc(last_sphere_cen_) -
                        gcopter_config_->dilateRadiusHard;
  while (curr_time < duration) {
    Vector3d curr_pos = traj.getPos(curr_time);
    if ((curr_pos - last_sphere_cen_).norm() < last_radius_) {
      curr_time += 0.05;
      continue;
    }
    // 超出了上一个球的范围, 更新一个球
    last_radius_ = lidar_map_interface_->getDisToOcc(curr_pos) -
                   gcopter_config_->dilateRadiusHard;
    last_sphere_cen_ = curr_pos;
    if (last_radius_ < 0) {
      collision_time = curr_time;
      return false;
    }
  }
  return true;
}

bool FastPlannerManager::hasCommittedBackup() const {
  return local_data_.backup_available_ &&
         local_data_.backup_traj_.getPieceNum() > 0 &&
         std::isfinite(local_data_.backup_start_t_) &&
         local_data_.backup_start_t_ < local_data_.duration_;
}

double FastPlannerManager::timeToCommittedBackup() const {
  if (!hasCommittedBackup()) {
    return std::numeric_limits<double>::infinity();
  }
  const double curr_time = (ros::Time::now() - local_data_.start_time_).toSec();
  if (!std::isfinite(curr_time)) {
    return std::numeric_limits<double>::infinity();
  }
  return local_data_.backup_start_t_ - curr_time;
}

double FastPlannerManager::committedTrajectoryRemainingTime() const {
  if (local_data_.minco_traj_.getPieceNum() <= 0) {
    return 0.0;
  }
  const double curr_time = (ros::Time::now() - local_data_.start_time_).toSec();
  if (!std::isfinite(curr_time)) {
    return 0.0;
  }
  return std::max(0.0, local_data_.duration_ - std::max(0.0, curr_time));
}

bool FastPlannerManager::isOnCommittedBackup() const {
  if (!hasCommittedBackup()) {
    return false;
  }
  const double curr_time = (ros::Time::now() - local_data_.start_time_).toSec();
  return std::isfinite(curr_time) && curr_time >= local_data_.backup_start_t_;
}

bool FastPlannerManager::updateRogMap(
    const sensor_msgs::PointCloud2ConstPtr &cloud_msg,
    const nav_msgs::Odometry::ConstPtr &odom_msg) {
  if (!gcopter_config_ || !gcopter_config_->rogMapEnable ||
      !map_manager_ || !map_manager_->rawRosMap()) {
    return false;
  }

  pcl::PointCloud<pcl::PointXYZ> input_cloud;
  pcl::fromROSMsg(*cloud_msg, input_cloud);
  if (input_cloud.empty() && lidar_map_interface_ &&
      lidar_map_interface_->ld_) {
    input_cloud = lidar_map_interface_->ld_->lidar_cloud_;
  }
  if (input_cloud.empty()) {
    return false;
  }

  rog_map::PointCloud rog_cloud;
  rog_cloud.reserve(input_cloud.size());
  for (const auto &p : input_cloud.points) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
      continue;
    }
    rog_map::PointType rp;
    rp.x = p.x;
    rp.y = p.y;
    rp.z = p.z;
    rp.intensity = 1.0F;
    rog_cloud.push_back(rp);
  }
  if (rog_cloud.empty()) {
    return false;
  }

  const auto &pose = odom_msg->pose.pose;
  super_utils::Pose rog_pose{
      super_utils::Vec3f(pose.position.x, pose.position.y, pose.position.z),
      super_utils::Quatf(pose.orientation.w, pose.orientation.x,
                         pose.orientation.y, pose.orientation.z)};
  map_manager_->rawRosMap()->updateMap(rog_cloud, rog_pose);
  return true;
}

bool FastPlannerManager::isRogReady() const {
  return gcopter_config_ && gcopter_config_->rogMapEnable &&
         map_manager_ && map_manager_->ready(general_planner::MapBackend::ROG);
}

bool FastPlannerManager::isLioStateSafe(const Eigen::Vector3d &pos,
                                        double safe_distance) const {
  return pos.allFinite() && lidar_map_interface_ &&
         lidar_map_interface_->getDisToOcc(pos) >= safe_distance;
}

bool FastPlannerManager::isLioSegmentSafe(const Eigen::Vector3d &start,
                                          const Eigen::Vector3d &end,
                                          double safe_distance,
                                          double step) const {
  if (!start.allFinite() || !end.allFinite()) {
    return false;
  }
  const double length = (end - start).norm();
  const int samples =
      std::max(1, static_cast<int>(std::ceil(length / std::max(1.0e-3, step))));
  for (int i = 0; i <= samples; ++i) {
    const double ratio = static_cast<double>(i) / static_cast<double>(samples);
    if (!isLioStateSafe(start + ratio * (end - start), safe_distance)) {
      return false;
    }
  }
  return true;
}

bool FastPlannerManager::isStateKnownFree(const Eigen::Vector3d &pos,
                                          double safe_distance) const {
  if (!pos.allFinite()) {
    return false;
  }
  const double lio_dist =
      lidar_map_interface_ ? lidar_map_interface_->getDisToOcc(pos)
                           : -std::numeric_limits<double>::infinity();
  const bool lio_safe = lio_dist >= safe_distance;
  if (!isRogReady()) {
    return lio_safe;
  }
  const auto inf_state = map_manager_->getPolicyGridType(
      pos, general_planner::MapBackend::ROG, true);
  if (inf_state == rog_map::GridType::KNOWN_FREE) {
    return map_manager_->isStateSafe(pos, safe_distance,
                                     general_planner::MapBackend::ROG) ||
           (gcopter_config_->rogKnownFreeFallbackToLio && lio_safe);
  }

  const auto raw_state = map_manager_->getPolicyGridType(
      pos, general_planner::MapBackend::ROG, false);
  const bool rog_unknown_like =
      inf_state == rog_map::GridType::UNKNOWN ||
      inf_state == rog_map::GridType::UNDEFINED ||
      inf_state == rog_map::GridType::FRONTIER ||
      inf_state == rog_map::GridType::OUT_OF_MAP ||
      ((inf_state == rog_map::GridType::OCCUPIED) &&
       (raw_state == rog_map::GridType::UNKNOWN ||
        raw_state == rog_map::GridType::UNDEFINED ||
        raw_state == rog_map::GridType::FRONTIER ||
        raw_state == rog_map::GridType::OUT_OF_MAP));
  if (gcopter_config_->rogKnownFreeFallbackToLio && lio_safe && rog_unknown_like) {
    ROS_WARN_THROTTLE(2.0,
                      "ROG known-free fallback to LIO is active: inf=%d raw=%d "
                      "lio_dist=%.2f safe=%.2f pos=(%.2f, %.2f, %.2f)",
                      static_cast<int>(inf_state), static_cast<int>(raw_state),
                      lio_dist, safe_distance, pos.x(), pos.y(), pos.z());
    return true;
  }
  ROS_WARN_THROTTLE(2.0,
                    "Known-free reject: inf=%d raw=%d lio_dist=%.2f safe=%.2f "
                    "pos=(%.2f, %.2f, %.2f)",
                    static_cast<int>(inf_state), static_cast<int>(raw_state),
                    lio_dist, safe_distance, pos.x(), pos.y(), pos.z());
  return false;
}

bool FastPlannerManager::isSegmentKnownFree(const Eigen::Vector3d &start,
                                            const Eigen::Vector3d &end,
                                            double safe_distance,
                                            double step) const {
  if (!start.allFinite() || !end.allFinite()) {
    return false;
  }
  const double length = (end - start).norm();
  const int samples =
      std::max(1, static_cast<int>(std::ceil(length / std::max(1.0e-3, step))));
  for (int i = 0; i <= samples; ++i) {
    const double ratio = static_cast<double>(i) / static_cast<double>(samples);
    if (!isStateKnownFree(start + ratio * (end - start), safe_distance)) {
      return false;
    }
  }
  return true;
}

double FastPlannerManager::estimateKnownFreePathLength(
    const vector<Eigen::Vector3d> &path,
    double safe_distance,
    double step) const {
  if (path.size() < 2) {
    return 0.0;
  }
  double known_free_len = 0.0;
  Eigen::Vector3d last = path.front();
  if (!isStateKnownFree(last, safe_distance)) {
    return 0.0;
  }
  for (std::size_t i = 1; i < path.size(); ++i) {
    const Eigen::Vector3d next = path[i];
    const double seg_len = (next - last).norm();
    if (seg_len < 1.0e-6) {
      continue;
    }
    const int samples =
        std::max(1, static_cast<int>(std::ceil(seg_len / std::max(1.0e-3, step))));
    Eigen::Vector3d prev = last;
    for (int s = 1; s <= samples; ++s) {
      const double ratio = static_cast<double>(s) / static_cast<double>(samples);
      const Eigen::Vector3d p = last + ratio * (next - last);
      if (!isSegmentKnownFree(prev, p, safe_distance, step)) {
        return known_free_len;
      }
      known_free_len += (p - prev).norm();
      prev = p;
    }
    last = next;
  }
  return known_free_len;
}

double FastPlannerManager::estimateLioSafePathLength(
    const vector<Eigen::Vector3d> &path,
    double safe_distance,
    double step) const {
  if (path.size() < 2) {
    return 0.0;
  }
  double safe_len = 0.0;
  Eigen::Vector3d last = path.front();
  if (!isLioStateSafe(last, safe_distance)) {
    return 0.0;
  }
  for (std::size_t i = 1; i < path.size(); ++i) {
    const Eigen::Vector3d next = path[i];
    const double seg_len = (next - last).norm();
    if (seg_len < 1.0e-6) {
      continue;
    }
    const int samples =
        std::max(1, static_cast<int>(std::ceil(seg_len / std::max(1.0e-3, step))));
    Eigen::Vector3d prev = last;
    for (int s = 1; s <= samples; ++s) {
      const double ratio = static_cast<double>(s) / static_cast<double>(samples);
      const Eigen::Vector3d p = last + ratio * (next - last);
      if (!isLioSegmentSafe(prev, p, safe_distance, step)) {
        return safe_len;
      }
      safe_len += (p - prev).norm();
      prev = p;
    }
    last = next;
  }
  return safe_len;
}

double FastPlannerManager::knownFreeAdaptiveVelocity(
    double known_free_remaining) const {
  if (!gcopter_config_) {
    return 3.0;
  }
  const double v_min = std::max(0.5, gcopter_config_->minSegmentVel);
  const double v_short =
      std::clamp(gcopter_config_->velocityShortKnownFree, v_min,
                 gcopter_config_->maxVelMag);
  const double v_medium =
      std::clamp(gcopter_config_->velocityMediumKnownFree, v_short,
                 gcopter_config_->maxVelMag);
  const double v_long =
      std::clamp(gcopter_config_->velocityLongKnownFree, v_medium,
                 gcopter_config_->maxVelMag);
  if (known_free_remaining >= gcopter_config_->knownFreeLongLength) {
    return v_long;
  }
  if (known_free_remaining >= gcopter_config_->knownFreeMediumLength) {
    return v_medium;
  }
  if (known_free_remaining >= gcopter_config_->knownFreeShortLength) {
    return std::min(v_medium, std::max(v_short, 0.5 * (v_short + v_medium)));
  }
  return v_short;
}

double FastPlannerManager::estimateCommittedKnownFreeSpan(
    const Trajectory<7> &traj,
    double &known_free_length) const {
  known_free_length = 0.0;
  if (traj.getPieceNum() <= 0 || !gcopter_config_) {
    return 0.0;
  }
  const double duration = traj.getTotalDuration();
  if (!std::isfinite(duration) || duration <= 1.0e-6) {
    return 0.0;
  }

  const double safe_distance =
      std::max(0.05, gcopter_config_->commitKnownFreeSafeDistance);
  const double sample_dt = std::max(0.02, gcopter_config_->commitSampleDt);
  Eigen::Vector3d last = traj.getPos(0.0);
  if (!isStateKnownFree(last, safe_distance)) {
    return 0.0;
  }
  double last_t = 0.0;
  for (double t = sample_dt; t <= duration + 1.0e-6; t += sample_dt) {
    const double tt = std::min(t, duration);
    const Eigen::Vector3d p = traj.getPos(tt);
    if (!isSegmentKnownFree(last, p, safe_distance, 0.5 * sample_dt *
                                                std::max(1.0, traj.getVel(last_t).norm()))) {
      return last_t;
    }
    known_free_length += (p - last).norm();
    last = p;
    last_t = tt;
    if (tt >= duration) {
      break;
    }
  }
  return duration;
}

double FastPlannerManager::estimateCommittedLioSafeSpan(
    const Trajectory<7> &traj,
    double &safe_length) const {
  safe_length = 0.0;
  if (traj.getPieceNum() <= 0 || !gcopter_config_) {
    return 0.0;
  }
  const double duration = traj.getTotalDuration();
  if (!std::isfinite(duration) || duration <= 1.0e-6) {
    return 0.0;
  }

  const double safe_distance =
      std::max(0.05, gcopter_config_->commitKnownFreeSafeDistance);
  const double sample_dt = std::max(0.02, gcopter_config_->commitSampleDt);
  Eigen::Vector3d last = traj.getPos(0.0);
  if (!isLioStateSafe(last, safe_distance)) {
    return 0.0;
  }
  double last_t = 0.0;
  for (double t = sample_dt; t <= duration + 1.0e-6; t += sample_dt) {
    const double tt = std::min(t, duration);
    const Eigen::Vector3d p = traj.getPos(tt);
    const double step =
        std::max(0.05, 0.5 * sample_dt * std::max(1.0, traj.getVel(last_t).norm()));
    if (!isLioSegmentSafe(last, p, safe_distance, step)) {
      return last_t;
    }
    safe_length += (p - last).norm();
    last = p;
    last_t = tt;
    if (tt >= duration) {
      break;
    }
  }
  return duration;
}

Eigen::VectorXd FastPlannerManager::computeCorridorVelocityLimits(
    const vector<Eigen::MatrixX4d> &hPolys,
    const vector<Eigen::Vector3d> &path) const {
  Eigen::VectorXd limits(static_cast<int>(hPolys.size()));
  if (!gcopter_config_) {
    limits.setConstant(3.0);
    return limits;
  }

  const double v_min = std::max(0.5, gcopter_config_->minSegmentVel);
  const double v_open = std::min(gcopter_config_->maxVelMag,
                                 std::max(v_min, gcopter_config_->openSegmentVel));
  const bool use_known_free_velocity = isRogReady();
  if (!gcopter_config_->dynamicVelocityEnable && !use_known_free_velocity) {
    limits.setConstant(v_open);
    std::cout << "[vel diag] DynamicVelocityEnable=false, all segment limits="
              << v_open << std::endl;
    return limits;
  }

  int clearance_limited = 0;
  int turn_limited = 0;
  int visible_limited = 0;
  int knownfree_limited = 0;
  int open_limited = 0;
  int min_idx = -1;
  double min_limit = std::numeric_limits<double>::infinity();
  double bottleneck_clearance = std::numeric_limits<double>::quiet_NaN();
  double bottleneck_effective_clearance = std::numeric_limits<double>::quiet_NaN();
  double min_turn_angle = 0.0;
  double min_v_open = v_open;
  double min_v_clearance = v_open;
  double min_v_turn = v_open;
  double min_v_visible = v_open;
  double min_v_knownfree = v_open;
  double min_knownfree_remaining = std::numeric_limits<double>::infinity();
  const char *min_reason = "open";
  double known_free_path_len =
      use_known_free_velocity
          ? estimateKnownFreePathLength(
                path, std::max(0.05, gcopter_config_->commitKnownFreeSafeDistance),
                std::max(0.05, gcopter_config_->commitSampleDt *
                                    std::max(1.0, gcopter_config_->maxVelMag)))
          : std::numeric_limits<double>::infinity();
  if (use_known_free_velocity && gcopter_config_->rogKnownFreeFallbackToLio &&
      known_free_path_len <
          std::max(1.0, 0.5 * gcopter_config_->knownFreeShortLength)) {
    const double lio_safe_path_len =
        estimateLioSafePathLength(
            path, std::max(0.05, gcopter_config_->commitKnownFreeSafeDistance),
            std::max(0.05, gcopter_config_->commitSampleDt *
                                std::max(1.0, gcopter_config_->maxVelMag)));
    if (lio_safe_path_len > known_free_path_len + 1.0) {
      std::cout << "[vel diag] use LIO safe path length fallback: "
                << "rog_known_free_path_len=" << known_free_path_len
                << " lio_safe_path_len=" << lio_safe_path_len << std::endl;
      known_free_path_len = lio_safe_path_len;
    }
  }
  double path_prefix_len = 0.0;

  for (int i = 0; i < limits.size(); ++i) {
    const Eigen::Vector3d travel_dir = pathHorizontalDirectionAt(path, i);
    const double clearance = estimateLateralHPolyClearance(hPolys[i], travel_dir);
    const double clearance_margin =
        std::max(0.0, gcopter_config_->dynamicVelocityClearanceMargin);
    const double min_clearance =
        std::max(0.05, gcopter_config_->dynamicVelocityMinClearance);
    const double open_clearance =
        std::max(clearance_margin + min_clearance,
                 gcopter_config_->dynamicVelocityOpenClearance);
    const double effective_clearance =
        std::isfinite(clearance) ? std::max(min_clearance, clearance - clearance_margin)
                                 : open_clearance;
    double v_clearance = v_open;
    if (gcopter_config_->dynamicVelocityEnable &&
        std::isfinite(clearance) && clearance < open_clearance) {
      v_clearance =
          0.92 * std::sqrt(2.0 * gcopter_config_->maxAccMag * effective_clearance);
    }

    const double turn_angle = turnAngleAt(path, i);
    double v_turn = v_open;
    if (gcopter_config_->dynamicVelocityEnable && turn_angle > 0.15) {
      const double turn_radius =
          std::isfinite(clearance) ? std::max(0.6, clearance) : open_clearance;
      v_turn = std::sqrt(std::max(0.1, gcopter_config_->maxAccMag * turn_radius /
                                           std::max(0.15, turn_angle)));
      if (turn_angle > 1.05) {
        v_turn = std::min(v_turn, 4.5);
      } else if (turn_angle > 0.65) {
        v_turn = std::min(v_turn, 6.0);
      }
    }

    const double v_visible =
        gcopter_config_->dynamicVelocityEnable
            ? std::sqrt(2.0 * gcopter_config_->maxAccMag *
                        std::max(0.5, max_ray_length -
                                           gcopter_config_->dilateRadiusHard - 2.0))
            : v_open;
    const double knownfree_remaining =
        use_known_free_velocity
            ? std::max(0.0, known_free_path_len - path_prefix_len)
            : std::numeric_limits<double>::infinity();
    const double v_knownfree =
        use_known_free_velocity ? knownFreeAdaptiveVelocity(knownfree_remaining)
                                : v_open;
    const double raw_limit =
        std::min({v_open, v_clearance, v_turn, v_visible, v_knownfree});
    limits(i) = std::clamp(raw_limit, v_min, gcopter_config_->maxVelMag);

    const char *reason = "open";
    if (raw_limit == v_clearance) {
      reason = "clearance";
      ++clearance_limited;
    } else if (raw_limit == v_turn) {
      reason = "turn";
      ++turn_limited;
    } else if (raw_limit == v_visible) {
      reason = "visible";
      ++visible_limited;
    } else if (raw_limit == v_knownfree) {
      reason = "known_free";
      ++knownfree_limited;
    } else {
      ++open_limited;
    }

    if (limits(i) < min_limit) {
      min_idx = i;
      min_limit = limits(i);
      bottleneck_clearance = clearance;
      bottleneck_effective_clearance = effective_clearance;
      min_turn_angle = turn_angle;
      min_v_open = v_open;
      min_v_clearance = v_clearance;
      min_v_turn = v_turn;
      min_v_visible = v_visible;
      min_v_knownfree = v_knownfree;
      min_knownfree_remaining = knownfree_remaining;
      min_reason = reason;
    }
    if (i + 1 < static_cast<int>(path.size())) {
      path_prefix_len += (path[i + 1] - path[i]).norm();
    }
  }
  if (limits.size() > 0) {
    std::cout << "[vel diag] segments=" << limits.size()
              << " limited(open/clearance/turn/visible/knownfree)=" << open_limited
              << "/" << clearance_limited << "/" << turn_limited << "/"
              << visible_limited << "/" << knownfree_limited << " min_idx=" << min_idx
              << " reason=" << min_reason
              << " final=" << min_limit
              << " known_free_path_len=" << known_free_path_len
              << " known_free_remaining=" << min_knownfree_remaining
              << " clearance=" << bottleneck_clearance
              << " effective_clearance=" << bottleneck_effective_clearance
              << " turn_angle=" << min_turn_angle
              << " components(open/clearance/turn/visible/knownfree)=" << min_v_open
              << "/" << min_v_clearance << "/" << min_v_turn << "/"
              << min_v_visible << "/" << min_v_knownfree << std::endl;
  }
  return limits;
}

bool FastPlannerManager::generateBackupTrajectory(const Trajectory<7> &exp_traj,
                                                  const Trajectory<5> &exp_yaw_traj,
                                                  Trajectory<7> &backup_traj,
                                                  Trajectory<5> &backup_yaw_traj,
                                                  double &backup_start_t) {
  backup_traj.clear();
  backup_yaw_traj.clear();
  backup_start_t = std::numeric_limits<double>::infinity();

  if (!gcopter_config_ || !gcopter_config_->backupTrajEnable) {
    return true;
  }
  if (exp_traj.getPieceNum() <= 0) {
    return false;
  }

  const double exp_dur = exp_traj.getTotalDuration();
  if (!std::isfinite(exp_dur) || exp_dur <= gcopter_config_->backupMinStartTime + 0.2) {
    return true;
  }

  double known_free_length = 0.0;
  double known_free_end_t = estimateCommittedKnownFreeSpan(exp_traj, known_free_length);
  bool use_lio_safe_span = false;
  if (gcopter_config_->rogKnownFreeFallbackToLio &&
      (!std::isfinite(known_free_end_t) || known_free_end_t <= 1.0e-4 ||
       known_free_length <
           std::max(1.0, 0.5 * gcopter_config_->knownFreeShortLength))) {
    double lio_safe_length = 0.0;
    const double lio_safe_end_t =
        estimateCommittedLioSafeSpan(exp_traj, lio_safe_length);
    if (std::isfinite(lio_safe_end_t) &&
        lio_safe_end_t > std::max(1.0e-4, known_free_end_t + 0.15) &&
        lio_safe_length > known_free_length + 1.0) {
      std::cout << "[backup diag] use LIO safe span fallback: "
                << "rog_known_free_t=" << known_free_end_t
                << " lio_safe_t=" << lio_safe_end_t
                << " lio_safe_len=" << lio_safe_length << std::endl;
      known_free_end_t = lio_safe_end_t;
      known_free_length = lio_safe_length;
      use_lio_safe_span = true;
    }
  }
  if (!std::isfinite(known_free_end_t) || known_free_end_t <= 1.0e-4) {
    known_free_end_t = isRogReady() ? 0.0 : exp_dur;
  }

  const double adaptive_commit_max =
      std::clamp(known_free_length /
                     std::max(1.0, knownFreeAdaptiveVelocity(known_free_length)),
                 gcopter_config_->commitMinDuration,
                 gcopter_config_->commitMaxDuration);
  const double max_start =
      std::min({gcopter_config_->backupMaxStartTime,
                gcopter_config_->commitMaxDuration,
                adaptive_commit_max,
                known_free_end_t - gcopter_config_->commitBackupTimeBuffer,
                exp_dur - 0.1});
  const double hard_min_start =
      std::min(gcopter_config_->backupMinStartTime,
               std::max({0.25, 3.0 * gcopter_config_->commitSampleDt,
                         2.0 * gcopter_config_->backupSampleDt}));
  const double effective_min_start =
      std::min(gcopter_config_->backupMinStartTime,
               std::max(hard_min_start,
                        std::min(gcopter_config_->commitMinDuration, max_start)));
  if (max_start < hard_min_start) {
    std::cout << "[backup diag] known-free span too short for committed backup: "
              << "known_free_t=" << known_free_end_t
              << " known_free_len=" << known_free_length
              << " max_start=" << max_start
              << " adaptive_commit_max=" << adaptive_commit_max
              << " hard_min_start=" << hard_min_start
              << " configured_min_start=" << gcopter_config_->backupMinStartTime
              << std::endl;
    return false;
  }
  if (max_start < gcopter_config_->backupMinStartTime) {
    std::cout << "[backup diag] short known-free commit horizon: "
              << "known_free_t=" << known_free_end_t
              << " known_free_len=" << known_free_length
              << " max_start=" << max_start
              << " adaptive_commit_max=" << adaptive_commit_max
              << " effective_min_start=" << effective_min_start
              << " configured_min_start=" << gcopter_config_->backupMinStartTime
              << std::endl;
  }

  const double ratio_start = exp_dur * gcopter_config_->backupStartRatio;
  backup_start_t = std::clamp(ratio_start,
                              effective_min_start,
                              std::max(effective_min_start, max_start));
  backup_start_t = std::min(backup_start_t, known_free_end_t - 0.05);
  if (backup_start_t >= exp_dur - 0.05) {
    backup_start_t = exp_dur - 0.05;
  }

  const Eigen::Vector3d start_p = exp_traj.getPos(backup_start_t);
  const Eigen::Vector3d start_v = exp_traj.getVel(backup_start_t);
  const double brake_acc = std::max(1.0, gcopter_config_->backupMaxAcc);
  const double stop_dist =
      start_v.squaredNorm() / (2.0 * brake_acc) + gcopter_config_->dilateRadiusHard + 0.4;

  double seed_t = backup_start_t;
  double walked = 0.0;
  Eigen::Vector3d last_p = start_p;
  const double sample_dt = std::max(0.02, gcopter_config_->backupSampleDt);
  const double seed_search_end_t = std::min(exp_dur, known_free_end_t);
  auto backupStateSafe = [&](const Eigen::Vector3d &p) {
    return use_lio_safe_span
               ? isLioStateSafe(p, gcopter_config_->commitKnownFreeSafeDistance)
               : isStateKnownFree(p, gcopter_config_->commitKnownFreeSafeDistance);
  };
  auto backupSegmentSafe = [&](const Eigen::Vector3d &a, const Eigen::Vector3d &b,
                               double step) {
    return use_lio_safe_span
               ? isLioSegmentSafe(a, b, gcopter_config_->commitKnownFreeSafeDistance, step)
               : isSegmentKnownFree(a, b, gcopter_config_->commitKnownFreeSafeDistance,
                                    step);
  };
  for (double t = backup_start_t + sample_dt; t <= seed_search_end_t + 1.0e-6; t += sample_dt) {
    const double tt = std::min(t, seed_search_end_t);
    const Eigen::Vector3d p = exp_traj.getPos(tt);
    if (!backupStateSafe(p)) {
      break;
    }
    walked += (p - last_p).norm();
    last_p = p;
    seed_t = tt;
    if (walked >= stop_dist || tt >= seed_search_end_t - 1.0e-6) {
      break;
    }
  }
  if (seed_t <= backup_start_t + 0.15) {
    seed_t = std::min(seed_search_end_t, backup_start_t + 0.45);
  }
  if (seed_t <= backup_start_t + 0.05) {
    std::cout << "[backup diag] no known-free braking distance after backup start: "
              << "backup_start=" << backup_start_t
              << " known_free_t=" << known_free_end_t
              << " speed_at_start=" << start_v.norm()
              << " stop_dist=" << stop_dist << std::endl;
    return false;
  }

  Eigen::Vector3d seed_p = exp_traj.getPos(seed_t);
  if ((seed_p - start_p).norm() < 0.15 && start_v.norm() > 1.0e-3) {
    seed_p = start_p + start_v.normalized() * 0.4;
  }

  const double margin = std::max(1.0, gcopter_config_->backupSearchMargin);
  Eigen::Vector3d min_bd = start_p.cwiseMin(seed_p) - Eigen::Vector3d(margin, margin, 1.5);
  Eigen::Vector3d max_bd = start_p.cwiseMax(seed_p) + Eigen::Vector3d(margin, margin, 1.5);
  if (lidar_map_interface_ && lidar_map_interface_->lp_) {
    min_bd = min_bd.cwiseMax(lidar_map_interface_->lp_->global_map_min_boundary_.cast<double>());
    max_bd = max_bd.cwiseMin(lidar_map_interface_->lp_->global_map_max_boundary_.cast<double>());
  }

  PointVector searched_points;
  if (lidar_map_interface_) {
    lidar_map_interface_->boxSearch(min_bd.cast<float>(), max_bd.cast<float>(), searched_points);
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_origin(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tmp(new pcl::PointCloud<pcl::PointXYZ>);
  cloud_origin->points = searched_points;
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  sor.setInputCloud(cloud_origin);
  sor.setLeafSize(0.2, 0.2, 0.2);
  sor.filter(*cloud_tmp);

  std::vector<Eigen::Vector3d> surf_points;
  surf_points.reserve(cloud_tmp->points.size());
  for (const pcl::PointXYZ &point : cloud_tmp->points) {
    surf_points.emplace_back(point.x, point.y, point.z);
  }

  Eigen::MatrixX4d bd = makeBoxHPoly(min_bd, max_bd);
  Eigen::MatrixX4d hp = bd;
  if (!surf_points.empty()) {
    Eigen::Map<const Eigen::Matrix<double, 3, -1, Eigen::ColMajor>> pc(
        surf_points[0].data(), 3, surf_points.size());
    Eigen::MatrixX4d firi_hp;
    if (firi::firi(bd, pc, start_p, seed_p, firi_hp) &&
        firi_hp.rows() > 0 && firi_hp.cols() == 4 && std::isfinite(firi_hp.sum())) {
      hp = firi_hp;
    }
  }
  if (hp.rows() <= 0 || hp.cols() != 4 || !std::isfinite(hp.sum())) {
    hp = bd;
  }

  std::vector<Eigen::MatrixX4d> hPolys;
  const int backup_poly_num = std::max(2, gcopter_config_->backupPieceNum);
  hPolys.reserve(backup_poly_num);
  for (int i = 0; i < backup_poly_num; ++i) {
    hPolys.push_back(hp);
  }

  Eigen::Matrix<double, 3, 4> iniState;
  iniState << start_p, start_v, exp_traj.getAcc(backup_start_t),
      exp_traj.getJer(backup_start_t);
  Eigen::Matrix<double, 3, 4> finState;
  finState << seed_p, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero();

  traj_manager_->setConfig(makeTrajOptConfig(*gcopter_config_));
  traj_opt::TrajManager::Request request;
  request.initial_state = iniState;
  request.terminal_state = finState;
  request.safe_corridor = hPolys;
  request.penalty_weights = makePenaltyWeights(*gcopter_config_, true);
  request.time_weight = gcopter_config_->WeightSafeT;
  const double backup_vel_bound =
      std::min(gcopter_config_->maxVelMag,
               std::max(gcopter_config_->backupMaxVel, start_v.norm() + 0.5));
  request.min_total_time =
      std::max({0.35, start_v.norm() / brake_acc,
                (seed_p - start_p).norm() / std::max(0.5, backup_vel_bound)});
  request.corridor_velocity_limits =
      Eigen::VectorXd::Constant(static_cast<int>(hPolys.size()),
                                backup_vel_bound);

  if (!traj_manager_->optimize(request, backup_traj)) {
    std::cout << "backup optimize failed!" << std::endl;
    return false;
  }
  const double allowed_backup_peak =
      std::min(gcopter_config_->maxVelMag,
               std::max(gcopter_config_->backupMaxVel, start_v.norm() + 0.5));
  if (backup_traj.getPieceNum() <= 0 ||
      backup_traj.getMaxVelRate() > allowed_backup_peak + 1.0 ||
      backup_traj.getMaxAccRate() > gcopter_config_->backupMaxAcc + 5.0) {
    std::cout << "backup magnitude check failed! peak_v="
              << backup_traj.getMaxVelRate()
              << " allowed_v=" << allowed_backup_peak
              << " peak_a=" << backup_traj.getMaxAccRate()
              << " allowed_a=" << gcopter_config_->backupMaxAcc
              << " speed_at_start=" << start_v.norm()
              << " known_free_len=" << known_free_length
              << " known_free_t=" << known_free_end_t
              << std::endl;
    return false;
  }
  const double backup_dur_check = backup_traj.getTotalDuration();
  Eigen::Vector3d last_backup_p = start_p;
  for (double t = 0.0; t <= backup_dur_check + 1.0e-6; t += sample_dt) {
    const double tt = std::min(t, backup_dur_check);
    const Eigen::Vector3d p = backup_traj.getPos(tt);
    if (!backupSegmentSafe(
            last_backup_p, p,
            std::max(0.05, sample_dt * std::max(1.0, start_v.norm())))) {
      std::cout << "[backup diag] optimized backup leaves known-free space at t="
                << tt << std::endl;
      return false;
    }
    last_backup_p = p;
    if (tt >= backup_dur_check) {
      break;
    }
  }

  const double backup_dur = backup_traj.getTotalDuration();
  if (exp_yaw_traj.getPieceNum() > 0 && backup_dur > 1.0e-4) {
    const double yaw_t = std::min(backup_start_t, exp_yaw_traj.getTotalDuration());
    Eigen::Matrix3d iniStateYaw;
    iniStateYaw << exp_yaw_traj.getPos(yaw_t), exp_yaw_traj.getVel(yaw_t),
        exp_yaw_traj.getAcc(yaw_t);
    Eigen::Matrix3d finStateYaw;
    finStateYaw << exp_yaw_traj.getPos(yaw_t), Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero();
    traj_opt::TrajManager::YawRequest yaw_request;
    yaw_request.initial_state = iniStateYaw;
    yaw_request.terminal_state = finStateYaw;
    yaw_request.guide_points.resize(3, 0);
    yaw_request.time_allocation = Eigen::VectorXd::Constant(1, backup_dur);
    yaw_request.target_duration = backup_dur;
    yaw_request.guide_weight = 0.0;
    yaw_request.time_weight = gcopter_config_->yaw_rho_vis;
    if (!traj_manager_->optimizeYaw(yaw_request, backup_yaw_traj)) {
      backup_yaw_traj.clear();
    }
  }
  if (backup_yaw_traj.getPieceNum() <= 0) {
    Piece<5>::CoefficientMat coeff = Piece<5>::CoefficientMat::Zero();
    const double yaw = exp_yaw_traj.getPieceNum() > 0
                           ? exp_yaw_traj.getPos(std::min(backup_start_t,
                                                          exp_yaw_traj.getTotalDuration())).x()
                           : local_data_.curr_yaw_;
    coeff(0, 5) = yaw;
    backup_yaw_traj.emplace_back(std::max(1.0e-3, backup_dur), coeff);
  }
  return true;
}

bool FastPlannerManager::commitTrajectory(const Trajectory<7> &exp_traj,
                                          const Trajectory<5> &exp_yaw_traj,
                                          const Trajectory<7> &backup_traj,
                                          const Trajectory<5> &backup_yaw_traj,
                                          double backup_start_t,
                                          const ros::Time &start_time) {
  if (exp_traj.getPieceNum() <= 0 || exp_yaw_traj.getPieceNum() <= 0) {
    return false;
  }

  Trajectory<7> committed_pos;
  Trajectory<5> committed_yaw;
  const bool use_backup =
      backup_traj.getPieceNum() > 0 &&
      backup_yaw_traj.getPieceNum() > 0 &&
      std::isfinite(backup_start_t) &&
      backup_start_t > 1.0e-4 &&
      backup_start_t < exp_traj.getTotalDuration();

  if (use_backup) {
    committed_pos = extractPrefixTrajectory(exp_traj, backup_start_t);
    committed_yaw = extractPrefixTrajectory(exp_yaw_traj,
                                            std::min(backup_start_t,
                                                     exp_yaw_traj.getTotalDuration()));
    if (committed_pos.getPieceNum() <= 0 || committed_yaw.getPieceNum() <= 0) {
      return false;
    }
    committed_pos.append(backup_traj);
    committed_yaw.append(backup_yaw_traj);
  } else {
    committed_pos = exp_traj;
    committed_yaw = exp_yaw_traj;
  }

  local_data_.exp_traj_ = exp_traj;
  local_data_.exp_yaw_traj_ = exp_yaw_traj;
  local_data_.backup_traj_ = backup_traj;
  local_data_.backup_yaw_traj_ = backup_yaw_traj;
  local_data_.backup_available_ = use_backup;
  local_data_.backup_start_t_ =
      use_backup ? backup_start_t : std::numeric_limits<double>::infinity();
  local_data_.minco_traj_ = committed_pos;
  local_data_.minco_yaw_traj_ = committed_yaw;
  local_data_.start_time_ = start_time;
  local_data_.duration_ = local_data_.minco_traj_.getTotalDuration();
  return local_data_.duration_ > 1.0e-4;
}

bool FastPlannerManager::planExploreTraj(const vector<Eigen::Vector3f> &path,
                                         bool is_static) {

  ros::Time start = ros::Time::now();

  vector<Eigen::Vector3d> path_shorten;
  bool use_shorten_path = false;
  int i = 0;
  int j = 0;
  for (j = path.size() - 1; j > 0; j--) {
    if ((path[j] - path[0]).norm() <= max_traj_len_ / 2.0)
      break;
  }
  double len = 0.0;
  for (i = 1; i < path.size();) {
    len += (path[i] - path[i - 1]).norm();
    if (len > max_traj_len_ || i == path.size() - 1) {
      break;
    }
    i++;
  }
  int end_idx = max(i, j);
  if (end_idx < path.size() - 1) {
    use_shorten_path = true;
  } else {
    use_shorten_path = false;
  }
  for (int i = 0; i <= end_idx; i++) {
    path_shorten.emplace_back(path[i].cast<double>());
  }
  const double shortened_path_len = pathLength(path_shorten);

  if (use_shorten_path) {
    Eigen::Vector3f fwd_dir = path[end_idx] - path[end_idx - 1];
    if (fwd_dir.x() * fwd_dir.x() + fwd_dir.y() * fwd_dir.y() >
        fwd_dir.z() * fwd_dir.z()) {
      local_data_.end_yaw_ = atan2(fwd_dir.y(), fwd_dir.x());
    }
  }
  // 从小到大
  Eigen::Vector3f min_bd, max_bd;
  for (int i = 0; i < 3; i++) {
    min_bd[i] = path_shorten[0][i];
    max_bd[i] = path_shorten[0][i];
  }
  for (const Eigen::Vector3d &waypoint : path_shorten) {
    for (int i = 0; i < 3; i++) {
      if (waypoint[i] < min_bd[i]) {
        min_bd[i] = waypoint[i];
      }
      if (waypoint[i] > max_bd[i]) {
        max_bd[i] = waypoint[i];
      }
    }
  }
  for (int i = 0; i < 2; i++) {
    min_bd[i] = (min_bd[i] - 3.0); // range
    max_bd[i] = (max_bd[i] + 3.0);
  }
  min_bd[2] -= 1.0;
  max_bd[2] += 1.0;

  PointVector Searched_Points;
  bool use_lio_corridor_points = true;
  if (gcopter_config_->corridorUseRogOccPoints &&
      isRogReady() && map_manager_ && map_manager_->rawMap()) {
    rog_map::Vec3f rog_min = min_bd.cast<double>();
    rog_map::Vec3f rog_max = max_bd.cast<double>();
    rog_map::vec_E<rog_map::Vec3f> rog_occ_points;
    map_manager_->rawMap()->boundBoxByLocalMap(rog_min, rog_max);
    if ((rog_max - rog_min).minCoeff() > 0.0) {
      map_manager_->rawMap()->boxSearch(
          rog_min, rog_max, rog_map::GridType::OCCUPIED, rog_occ_points);
    }
    if (!rog_occ_points.empty()) {
      Searched_Points.reserve(rog_occ_points.size());
      for (const auto &p : rog_occ_points) {
        Searched_Points.emplace_back(
            static_cast<float>(p.x()), static_cast<float>(p.y()),
            static_cast<float>(p.z()));
      }
      use_lio_corridor_points = false;
      std::cout << "[corridor diag] use ROG raw occ points="
                << Searched_Points.size() << std::endl;
    }
  }
  if (use_lio_corridor_points && lidar_map_interface_) {
    lidar_map_interface_->boxSearch(min_bd, max_bd, Searched_Points);
    std::cout << "[corridor diag] use LIO occ points="
              << Searched_Points.size() << std::endl;
  }

  // 降采样
  std::vector<Eigen::Vector3d> surf_points;
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_origin(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tmp(
      new pcl::PointCloud<pcl::PointXYZ>);
  cloud_origin->points = Searched_Points;
  sor.setInputCloud(cloud_origin);
  sor.setLeafSize(0.2, 0.2, 0.2);
  sor.filter(*cloud_tmp);

  surf_points.reserve(cloud_tmp->points.size());
  for (const pcl::PointXYZ &point : cloud_tmp->points) {
    surf_points.emplace_back(point.x, point.y, point.z);
  }

  ros::Time point_process_end_stamp = ros::Time::now();

  std::vector<Eigen::MatrixX4d> hPolys; // 多面体飞行走廊

  sfc_gen::convexCover(gcopter_viz_, path_shorten, surf_points,
                       min_bd.cast<double>(), max_bd.cast<double>(), 7.0,
                       gcopter_config_->corridor_size, hPolys, 1e-6,
                       gcopter_config_->dilateRadiusSoft);
  Eigen::Matrix<double, 3, 4> iniState;
  Eigen::Matrix<double, 3, 4> finState;
  double time_now = (ros::Time::now() - local_data_.start_time_).toSec();
  if (is_static) {
    // TSP、更新地图等会阻塞里程计回调函数，导致这里的数据不准，所以只有static才用
    iniState << local_data_.curr_pos_, Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
  } else {
    time_now =
        time_now > local_data_.duration_ ? local_data_.duration_ : time_now;
    Eigen::Vector3d current_pose = local_data_.minco_traj_.getPos(time_now);
    Eigen::Vector3d curr_vel = local_data_.minco_traj_.getVel(time_now);
    Eigen::Vector3d curr_acc = local_data_.minco_traj_.getAcc(time_now);
    Eigen::Vector3d curr_jerk = local_data_.minco_traj_.getJer(time_now);
    iniState << current_pose, curr_vel, curr_acc, curr_jerk;
  }

  Eigen::Vector4d bh;
  bh << iniState.topLeftCorner<3, 1>(), 1.0;
  int start_idx = -1;
  for (int i = hPolys.size() - 1; i >= 0; i--) {
    Eigen::MatrixX4d hp = hPolys[i];
    if ((((hp * bh).array() > -1.0e-6).cast<int>().sum() <= 0)) {
      start_idx = i;
      break;
    }
  }
  if (start_idx == -1) {
    ROS_ERROR("current position not in corridor");
    double time;
    bool safe =
        local_data_.traj_id_ >= 1 && checkTrajCollision(time) && time > 2.0;
    if (!safe)
      return flyToSafeRegion(is_static);
    // return false;
  }
  if (start_idx != 0) {
    hPolys.erase(hPolys.begin(), hPolys.begin() + start_idx);
  }
  sfc_gen::shortCut(hPolys);

  // ros::Duration(1.0).sleep();

  ros::Time hpoly_gen_end = ros::Time::now();

  if (hPolys.size() < 2) {
    cout << "hPolys size < 2" << endl;
    return false;
  }
  int front = 0;
  int back = 1;
  while (back < hPolys.size() - 1) {
    bool overlap = geo_utils::overlap(hPolys[front], hPolys[back], 1e-2);
    if (overlap) {
      front += 1;
      back += 1;
    } else {
      break;
    }
  }
  if (front != hPolys.size() - 2) {
    ROS_ERROR("front != hPolys.size() - 2");
    Eigen::Vector3d inner;
    geo_utils::findInterior(hPolys[front], inner);
    finState << inner, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero();
    hPolys.resize(front + 1);
    gcopter_viz_->visualizePolytope(hPolys, true);
  } else {
    finState << path_shorten.back(), Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
    gcopter_viz_->visualizePolytope(hPolys);
  }
  gcopter_viz_->visualizeRoute(path);

  auto local_data_backup = local_data_;
  double time_lb;
  calculateTimelb(path_shorten, local_data_.curr_yaw_, local_data_.end_yaw_,
                  time_lb);
  cout << "lower_bd = " << time_lb << endl;
  const Eigen::VectorXd corridor_velocity_limits =
      computeCorridorVelocityLimits(hPolys, path_shorten);
  double terminal_speed = 0.0;
  if (gcopter_config_->nonstopTerminalVelocityEnable &&
      gcopter_config_->backupTrajEnable &&
      front == hPolys.size() - 2 &&
      shortened_path_len >= gcopter_config_->nonstopTerminalMinPathLength) {
    const Eigen::Vector3d dir = terminalDirection(path_shorten);
    const double last_limit =
        corridor_velocity_limits.size() > 0
            ? corridor_velocity_limits(corridor_velocity_limits.size() - 1)
            : gcopter_config_->maxVelMag;
    const double desired_speed =
        std::max(gcopter_config_->minSegmentVel,
                 gcopter_config_->openSegmentVel *
                     gcopter_config_->nonstopTerminalVelocityRatio);
    terminal_speed = std::clamp(desired_speed, 0.0,
                                std::min(gcopter_config_->maxVelMag, last_limit));
    finState.col(1) = dir * terminal_speed;
  }
  if (corridor_velocity_limits.size() > 0) {
    std::cout << "[traj opt] path_len=" << shortened_path_len
              << " hpolys=" << hPolys.size()
              << " vel_limit(min/max/avg)="
              << corridor_velocity_limits.minCoeff() << "/"
              << corridor_velocity_limits.maxCoeff() << "/"
              << corridor_velocity_limits.mean()
              << " terminal_v=" << terminal_speed << std::endl;
  }
  traj_manager_->setConfig(makeTrajOptConfig(*gcopter_config_));
  traj_opt::TrajManager::Request opt_request;
  opt_request.initial_state = iniState;
  opt_request.terminal_state = finState;
  opt_request.safe_corridor = hPolys;
  opt_request.penalty_weights = makePenaltyWeights(*gcopter_config_, false);
  opt_request.corridor_velocity_limits = corridor_velocity_limits;
  opt_request.time_weight = gcopter_config_->weightT;
  opt_request.min_total_time = time_lb;
  Trajectory<7> exp_traj;
  if (!traj_manager_->optimize(opt_request, exp_traj)) {
    std::cout << "optimize failed!" << std::endl;
    local_data_ = local_data_backup;
    return false;
  }

  local_data_.minco_traj_ = exp_traj;
  local_data_.duration_ = local_data_.minco_traj_.getTotalDuration();
  local_data_.start_time_ = hpoly_gen_end;

  double time = 10.0;
  if (!gcopter_config_->backupTrajEnable && !checkTrajCollision(time) && time < 1.0) {
    std::cout << "check traj collision failed" << std::endl;
    local_data_ = local_data_backup;
    return false;
  }
  if (!checkTrajVelocity()) {
    std::cout << "check traj velocity failed" << std::endl;
    local_data_ = local_data_backup;
    return false;
  }

  if (local_data_.minco_traj_.getPieceNum() <= 0) {
    local_data_ = local_data_backup;
    std::cout << "traj empty!" << std::endl;
    return false;
  }

  local_data_.start_time_ = local_data_backup.start_time_;
  if (!YawTrajOpt(local_data_.curr_yaw_, local_data_.end_yaw_, is_static,
                  use_shorten_path)) {
    cout << "yaw_traj_opt failed!" << endl;
    local_data_ = local_data_backup;
    return false;
  }
  local_data_.start_time_ = hpoly_gen_end;
  Trajectory<5> exp_yaw_traj = local_data_.minco_yaw_traj_;

  Trajectory<7> backup_traj;
  Trajectory<5> backup_yaw_traj;
  double backup_start_t = std::numeric_limits<double>::infinity();
  if (!generateBackupTrajectory(exp_traj, exp_yaw_traj, backup_traj,
                                backup_yaw_traj, backup_start_t)) {
    local_data_ = local_data_backup;
    return false;
  }
  if (!commitTrajectory(exp_traj, exp_yaw_traj, backup_traj, backup_yaw_traj,
                        backup_start_t, hpoly_gen_end)) {
    local_data_ = local_data_backup;
    return false;
  }
  if (gcopter_config_->backupTrajEnable &&
      exp_traj.getTotalDuration() > gcopter_config_->backupMinStartTime + 0.2 &&
      !local_data_.backup_available_) {
    std::cout << "backup required but committed trajectory has no backup!" << std::endl;
    local_data_ = local_data_backup;
    return false;
  }
  if (local_data_.backup_available_) {
    const double sample_dt = std::max(0.02, gcopter_config_->commitSampleDt);
    Eigen::Vector3d last_commit_p = local_data_.minco_traj_.getPos(0.0);
    for (double t = sample_dt; t <= local_data_.duration_ + 1.0e-6; t += sample_dt) {
      const double tt = std::min(t, local_data_.duration_);
      const Eigen::Vector3d p = local_data_.minco_traj_.getPos(tt);
      const double safety_step =
          std::max(0.05, sample_dt *
                             std::max(1.0, local_data_.minco_traj_.getVel(tt).norm()));
      bool commit_segment_safe =
          isSegmentKnownFree(last_commit_p, p,
                             gcopter_config_->commitKnownFreeSafeDistance,
                             safety_step);
      if (!commit_segment_safe && gcopter_config_->rogKnownFreeFallbackToLio &&
          isLioSegmentSafe(last_commit_p, p,
                           gcopter_config_->commitKnownFreeSafeDistance,
                           safety_step)) {
        ROS_WARN_THROTTLE(2.0,
                          "Committed trajectory check falls back to LIO safety.");
        commit_segment_safe = true;
      }
      if (!commit_segment_safe) {
        std::cout << "committed trajectory known-free check failed at t="
                  << tt << std::endl;
        local_data_ = local_data_backup;
        return false;
      }
      last_commit_p = p;
      if (tt >= local_data_.duration_) {
        break;
      }
    }
  }
  const auto exp_peak = estimatePeakSpeedInfo(exp_traj, 0.02);
  const auto commit_peak = estimatePeakSpeedInfo(local_data_.minco_traj_, 0.02);
  std::cout << "[traj opt] exp_duration=" << exp_traj.getTotalDuration()
            << " exp_peak_v=" << exp_peak.speed << "@" << exp_peak.time
            << " commit_duration=" << local_data_.duration_
            << " commit_peak_v=" << commit_peak.speed << "@" << commit_peak.time
            << " backup_start=" << local_data_.backup_start_t_ << std::endl;
  if (local_data_.backup_available_) {
    const auto exp_peak_before_backup =
        estimatePeakSpeedInfo(exp_traj, 0.02, 0.0, backup_start_t);
    const auto exp_peak_after_backup =
        estimatePeakSpeedInfo(exp_traj, 0.02, backup_start_t,
                              exp_traj.getTotalDuration());
    const auto backup_peak = estimatePeakSpeedInfo(backup_traj, 0.02);
    std::cout << "[backup diag] speed_at_start="
              << exp_traj.getVel(backup_start_t).norm()
              << " exp_peak_before=" << exp_peak_before_backup.speed << "@"
              << exp_peak_before_backup.time
              << " exp_peak_after=" << exp_peak_after_backup.speed << "@"
              << exp_peak_after_backup.time
              << " backup_duration=" << backup_traj.getTotalDuration()
              << " backup_peak_v=" << backup_peak.speed << "@"
              << backup_peak.time << std::endl;
  } else {
    std::cout << "[backup diag] backup disabled or unavailable; committed trajectory is exploration trajectory"
              << std::endl;
  }

  double commit_collision_time = 0.0;
  if (!checkTrajCollision(commit_collision_time)) {
    std::cout << "committed traj collision check failed: " << commit_collision_time << std::endl;
    local_data_ = local_data_backup;
    return false;
  }
  if (!checkTrajVelocity()) {
    std::cout << "committed traj velocity/acc check failed" << std::endl;
    local_data_ = local_data_backup;
    return false;
  }
  gcopter_viz_->visualize(local_data_.minco_traj_,
                          gcopter_config_->maxVelMag);

  ros::Time optimize_end_stamp = ros::Time::now();
  double trajOptimize_time =
      (optimize_end_stamp - hpoly_gen_end).toSec() * 1000;
  local_data_.traj_id_ += 1;
  local_data_.start_pos_ = path_shorten.front();
  return true;
}

void FastPlannerManager::angleLimite(double &angle) {
  while (angle > M_PI) {
    angle -= (M_PI * 2);
  }
  while (angle < -M_PI) {
    angle += (M_PI * 2);
  }
}

bool FastPlannerManager::YawTrajOpt(double &start_yaw, double &end_yaw,
                                    bool is_static, bool use_shorten_path) {
  Eigen::Matrix3d iniStateYaw, finStateYaw;
  Eigen::MatrixXd wpsYaw;
  Eigen::VectorXd opt_times_Yaw;
  double time_now = (ros::Time::now() - local_data_.start_time_).toSec();
  double yaw_sp, yaw_sv(0.0), yaw_sa(0.0), yaw_ep(end_yaw);

  if (is_static || local_data_.minco_yaw_traj_.getPieceNum() <= 0) {
    yaw_sp = start_yaw;
  } else {
    time_now = time_now > local_data_.minco_yaw_traj_.getTotalDuration()
                   ? local_data_.minco_yaw_traj_.getTotalDuration()
                   : time_now;
    yaw_sp = local_data_.minco_yaw_traj_.getPos(time_now).x();
    yaw_sv = local_data_.minco_yaw_traj_.getVel(time_now).x();
    yaw_sa = local_data_.minco_yaw_traj_.getAcc(time_now).x();
  }
  angleLimite(yaw_sp);
  static double yaw_dur = 0.3;
  // double yaw_dur = local_data_.duration_ / 12.0;
  double fwd_time = gcopter_config_->yaw_time_fwd;
  vector<double> look_fwd_wp;
  look_fwd_wp.push_back(yaw_sp);
  for (double t = yaw_dur; t < local_data_.duration_ + yaw_dur; t += yaw_dur) {
    if (t > local_data_.duration_) {
      double delta = local_data_.end_yaw_ - look_fwd_wp.back();
      angleLimite(delta);
      look_fwd_wp.push_back(look_fwd_wp.back() + delta);
      break;
    }
    Eigen::Vector3d p_c = local_data_.minco_traj_.getPos(t);
    double t_fwd = t + fwd_time;
    Eigen::Vector3d p_f =
        local_data_.minco_traj_.getPos(min(t_fwd, local_data_.duration_));
    Eigen::Vector2d dir = p_f.head(2) - p_c.head(2);
    if (dir.norm() < fabs(p_f.z() - p_c.z()) || dir.norm() < 0.05) {
      // if (p_f.z() - p_c.z() > 0) {
      //   look_fwd_wp.push_back(look_fwd_wp.back());
      // } else
      // if (dir.norm() > 0.1) {
      //   double yaw = atan2(dir.y(), dir.x()) + min(M_PI, yaw_dur *
      //   gcopter_config_->yaw_max_vel); double delta_yaw = yaw -
      //   look_fwd_wp.back(); angleLimite(delta_yaw);
      //   look_fwd_wp.push_back(look_fwd_wp.back() + delta_yaw);
      // } else
      look_fwd_wp.push_back(
          look_fwd_wp.back() +
          min(M_PI, yaw_dur * gcopter_config_->yaw_max_vel * 0.6));
    } else {
      double yaw = atan2(dir.y(), dir.x());
      double delta_yaw = yaw - look_fwd_wp.back();
      angleLimite(delta_yaw);
      look_fwd_wp.push_back(look_fwd_wp.back() + delta_yaw);
    }
  }

  vector<double> wp;
  double delta_yaw_max = gcopter_config_->yaw_max_vel * yaw_dur;
  std::random_device rd;
  std::mt19937 gen(rd()); // 使用Mersenne Twister作为随机数引擎

  std::normal_distribution<double> dist(0, 2e-3);

  for (int i = 0; i < look_fwd_wp.size(); i++) {
    if (i == 0) {
      wp.push_back(look_fwd_wp[i]);
      continue;
    }
    double last_yaw = wp.back();
    double diff2end_yaw = local_data_.end_yaw_ - last_yaw;
    angleLimite(diff2end_yaw);
    double time2end_yaw = abs(diff2end_yaw) / gcopter_config_->yaw_max_vel;
    double time_last = local_data_.duration_ - i * yaw_dur;
    double next_yaw;
    if (time_last <= time2end_yaw)
      next_yaw = local_data_.end_yaw_ + dist(gen);
    else
      next_yaw = look_fwd_wp[i] + dist(gen);

    double delta_yaw = next_yaw - last_yaw;
    angleLimite(delta_yaw);
    if (delta_yaw < 0 && delta_yaw < -delta_yaw_max) {
      delta_yaw = -delta_yaw_max;
    } else if (delta_yaw > 0 && delta_yaw > delta_yaw_max) {
      delta_yaw = delta_yaw_max;
    }
    wp.push_back(wp.back() + delta_yaw);
  }
  double delta2end = local_data_.end_yaw_ - wp.back();
  angleLimite(delta2end);
  local_data_.end_yaw_ = wp.back() + delta2end;
  yaw_ep = local_data_.end_yaw_;
  iniStateYaw << Eigen::Vector3d(yaw_sp, 0.0, 0.0),
      Eigen::Vector3d(yaw_sv, 0.0, 0.0), Eigen::Vector3d(yaw_sa, 0.0, 0.0);
  finStateYaw << Eigen::Vector3d(yaw_ep, 0.0, 0.0), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero();

  int pieceNUM = wp.size() - 2;
  if (pieceNUM <= 1) {
    opt_times_Yaw.resize(2);
    opt_times_Yaw[0] = local_data_.duration_ / 2.0;
    opt_times_Yaw[1] = local_data_.duration_ / 2.0;
    wpsYaw.resize(3, 1);
    wpsYaw(0, 0) = (wp[0] + wp[1]) / 2.0;
    wpsYaw(1, 0) = 0.0;
    wpsYaw(2, 0) = 0.0;
    pieceNUM = 2;
  } else {
    // 干掉最后一个值
    opt_times_Yaw.resize(wp.size() - 2);
    for (int i = 0; i < wp.size() - 3; i++) {
      opt_times_Yaw[i] = yaw_dur;
    }
    opt_times_Yaw[wp.size() - 3] =
        local_data_.duration_ - (wp.size() - 3) * yaw_dur;
    wpsYaw.resize(3, wp.size() - 3);
    for (int i = 1; i < wp.size() - 2; i++) {
      wpsYaw(0, i - 1) = wp[i];
      wpsYaw(1, i - 1) = 0.0;
      wpsYaw(2, i - 1) = 0.0;
    }
  }
  // double dur_yaw = 0.0;
  // for (int i = 0; i < opt_times_Yaw.size(); i++) {
  //   dur_yaw += opt_times_Yaw[i];
  // }
  // cout << "dur_p= " << local_data_.duration_ << " dur_yaw= " << dur_yaw <<
  // endl; cout << "start yaw = " << iniStateYaw.col(0).transpose() << endl;
  // cout << "end yaw = " << finStateYaw.col(0).transpose() << endl;
  // cout << "wpsYaw = " << endl;
  // for (int i = 0; i < wpsYaw.cols(); i++) {
  //   cout << wpsYaw.col(i).transpose()(0) << " ";
  // }
  // cout << endl;

  local_data_.minco_yaw_traj_.clear();
  traj_opt::TrajManager::YawRequest yaw_request;
  yaw_request.initial_state = iniStateYaw;
  yaw_request.terminal_state = finStateYaw;
  yaw_request.guide_points = wpsYaw;
  yaw_request.time_allocation = opt_times_Yaw;
  yaw_request.target_duration = local_data_.duration_;
  yaw_request.guide_weight = gcopter_config_->yaw_rho_vis;
  yaw_request.time_weight = gcopter_config_->yaw_rho_vis;
  if (!traj_manager_->optimizeYaw(yaw_request, local_data_.minco_yaw_traj_)) {
    std::cout << "optimize yaw failed!" << std::endl;
    return false;
  }
  return true;
}

bool FastPlannerManager::flyToSafeRegion(bool is_static) {
  Eigen::Vector3f min_bd, max_bd;
  for (int i = 0; i < 3; i++) {
    min_bd[i] = topo_graph_->odom_node_->center_[i] - 2.0;
    max_bd[i] = topo_graph_->odom_node_->center_[i] + 2.0;
  }
  PointVector Searched_Points;
  lidar_map_interface_->boxSearch(min_bd, max_bd, Searched_Points);
  std::vector<Eigen::Vector3d> surf_points;
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_origin(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tmp(
      new pcl::PointCloud<pcl::PointXYZ>);
  cloud_origin->points = Searched_Points;
  sor.setInputCloud(cloud_origin);
  sor.setLeafSize(0.2, 0.2, 0.2);
  sor.filter(*cloud_tmp);

  surf_points.reserve(cloud_tmp->points.size());
  for (const pcl::PointXYZ &point : cloud_tmp->points) {
    surf_points.emplace_back(point.x, point.y, point.z);
  }
  Eigen::Matrix<double, 6, 4> bd = Eigen::Matrix<double, 6, 4>::Zero();
  bd(0, 0) = 1.0;
  bd(1, 0) = -1.0;
  bd(2, 1) = 1.0;
  bd(3, 1) = -1.0;
  bd(4, 2) = 1.0;
  bd(5, 2) = -1.0;
  bd(0, 3) =
      -(std::min(topo_graph_->odom_node_->center_(0) + 2.0f,
                 lidar_map_interface_->lp_->global_map_max_boundary_[0]));
  bd(1, 3) = std::max(topo_graph_->odom_node_->center_(0) - 2.0f,
                      lidar_map_interface_->lp_->global_box_min_boundary_[0]);
  bd(2, 3) =
      -(std::min(topo_graph_->odom_node_->center_(1) + 2.0f,
                 lidar_map_interface_->lp_->global_map_max_boundary_[1]));
  bd(3, 3) = std::max(topo_graph_->odom_node_->center_(1) - 2.0f,
                      lidar_map_interface_->lp_->global_box_min_boundary_[1]);
  bd(4, 3) =
      -(std::min(topo_graph_->odom_node_->center_(2) + 1.0f,
                 lidar_map_interface_->lp_->global_map_max_boundary_[2]));
  bd(5, 3) = std::max(topo_graph_->odom_node_->center_(2) - 1.0f,
                      lidar_map_interface_->lp_->global_box_min_boundary_[2]);
  Eigen::Map<const Eigen::Matrix<double, 3, -1, Eigen::ColMajor>> pc(
      surf_points[0].data(), 3, surf_points.size());
  Eigen::MatrixX4d hp;
  firi::firi(bd, pc, topo_graph_->odom_node_->center_.cast<double>(),
             topo_graph_->odom_node_->center_.cast<double>(),
             hp); // 计算出包含a和b的凸包
  std::vector<Eigen::MatrixX4d> hPolys;
  hPolys.push_back(hp);
  hPolys.push_back(hp);
  Eigen::Vector3d inner;
  geo_utils::findInterior(hp, inner);
  Eigen::Vector4d bh;
  double time_now = (ros::Time::now() - local_data_.start_time_).toSec();
  Eigen::Matrix<double, 3, 4> iniState;
  Eigen::Vector3d dir =
      (inner - topo_graph_->odom_node_->center_.cast<double>()).normalized();
  if (is_static) {
    // TSP、更新地图等会阻塞里程计回调函数，导致这里的数据不准，所以只有static才用
    iniState << local_data_.curr_pos_, dir * 0.2, Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero();
    // iniState << topo_graph_->odom_node_->center_, local_data_.curr_vel_,
    // Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
  } else {
    time_now =
        time_now > local_data_.duration_ ? local_data_.duration_ : time_now;
    Eigen::Vector3d current_pose = local_data_.minco_traj_.getPos(time_now);
    Eigen::Vector3d curr_vel = local_data_.minco_traj_.getVel(time_now);
    Eigen::Vector3d curr_acc = local_data_.minco_traj_.getAcc(time_now);
    Eigen::Vector3d curr_jerk = local_data_.minco_traj_.getJer(time_now);
    // iniState << current_pose, curr_vel, curr_acc, curr_jerk;
    iniState << current_pose, dir * 0.2, Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero();
  }
  Eigen::Matrix<double, 3, 4> finState;
  ros::Time hpoly_gen_end = ros::Time::now();
  // iniState << topo_graph_->odom_node_->center_, local_data_.curr_vel_,
  // Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
  finState << inner, dir, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
  bh << iniState.topLeftCorner<3, 1>(), 1.0;
  int start_idx = -1;
  for (int i = hPolys.size() - 1; i >= 0; i--) {
    Eigen::MatrixX4d hp = hPolys[i];
    if ((((hp * bh).array() > -1.0e-6).cast<int>().sum() <= 0)) {
      start_idx = i;
      break;
    }
  }
  if (start_idx == -1) {
    ROS_ERROR("current position not in corridor");
    return false;
  }
  if (start_idx != 0) {
    hPolys.erase(hPolys.begin(), hPolys.begin() + start_idx);
  }
  sfc_gen::shortCut(hPolys);

  // ros::Duration(1.0).sleep();

  if (hPolys.size() < 2) {
    cout << "hPolys size < 2" << endl;
    return false;
  }
  gcopter_viz_->visualizePolytope(hPolys);
  auto local_data_backup = local_data_;
  local_data_.minco_traj_.clear();
  traj_manager_->setConfig(makeTrajOptConfig(*gcopter_config_));
  traj_opt::TrajManager::Request opt_request;
  opt_request.initial_state = iniState;
  opt_request.terminal_state = finState;
  opt_request.safe_corridor = hPolys;
  opt_request.penalty_weights = makePenaltyWeights(*gcopter_config_, true);
  opt_request.corridor_velocity_limits =
      Eigen::VectorXd::Constant(static_cast<int>(hPolys.size()),
                                std::min(gcopter_config_->backupMaxVel,
                                         gcopter_config_->maxVelMag));
  opt_request.time_weight = gcopter_config_->WeightSafeT;
  opt_request.min_total_time = 0.0;
  if (!traj_manager_->optimize(opt_request, local_data_.minco_traj_)) {
    std::cout << "optimize failed!" << std::endl;
    local_data_ = local_data_backup;
    return false;
  }
  if (local_data_.minco_traj_.getPieceNum() > 0) {
    // ROS_INFO_STREAM(
    // "local_data_.minco_traj_.getPieceNum(): " <<
    // local_data_.minco_traj_.getPieceNum());
    gcopter_viz_->visualize(local_data_.minco_traj_,
                            gcopter_config_->maxVelMag);
  } else {
    local_data_ = local_data_backup;
    std::cout << "traj empty!" << std::endl;
    return false;
  }
  ros::Time optimize_end_stamp = ros::Time::now();
  local_data_.traj_id_ += 1;
  local_data_.start_time_ = hpoly_gen_end;
  local_data_.start_pos_ = topo_graph_->odom_node_->center_.cast<double>();
  local_data_.duration_ = local_data_.minco_traj_.getTotalDuration();
  local_data_.backup_available_ = false;
  local_data_.backup_start_t_ = std::numeric_limits<double>::infinity();
  local_data_.backup_traj_.clear();
  local_data_.backup_yaw_traj_.clear();
  return true;
}

void FastPlannerManager::polyTraj2ROSMsg(traj_utils::PolyTraj &poly_msg,
                                         const ros::Time &start_time) {
  Eigen::VectorXd durs = local_data_.minco_traj_.getDurations();
  int piece_num = local_data_.minco_traj_.getPieceNum();
  poly_msg.drone_id = 0;
  poly_msg.traj_id = local_data_.traj_id_;
  poly_msg.start_time = start_time;
  poly_msg.order = 7;
  poly_msg.duration.resize(piece_num);
  poly_msg.coef_x.resize(8 * piece_num);
  poly_msg.coef_y.resize(8 * piece_num);
  poly_msg.coef_z.resize(8 * piece_num);
  for (int i = 0; i < piece_num; ++i) {
    poly_msg.duration[i] = durs(i);
    Eigen::Matrix<double, 3, 8> cMat =
        local_data_.minco_traj_.pieces[i].getCoeffMat();
    int i6 = i * 8;
    for (int j = 0; j < 8; j++) {
      poly_msg.coef_x[i6 + j] = cMat(0, j);
      poly_msg.coef_y[i6 + j] = cMat(1, j);
      poly_msg.coef_z[i6 + j] = cMat(2, j);
    }
  }
}

void FastPlannerManager::polyYawTraj2ROSMsg(traj_utils::PolyTraj &poly_msg,
                                            const ros::Time &start_time) {
  Eigen::VectorXd durs = local_data_.minco_yaw_traj_.getDurations();
  int piece_num = local_data_.minco_yaw_traj_.getPieceNum();
  poly_msg.drone_id = 0;
  poly_msg.traj_id = local_data_.traj_id_;
  poly_msg.start_time = start_time;
  poly_msg.order = 5;
  poly_msg.duration.resize(piece_num);
  poly_msg.coef_x.resize(6 * piece_num);
  poly_msg.coef_y.resize(6 * piece_num);
  poly_msg.coef_z.resize(6 * piece_num);
  for (int i = 0; i < piece_num; ++i) {
    poly_msg.duration[i] = durs(i);
    Eigen::Matrix<double, 3, 6> cMat =
        local_data_.minco_yaw_traj_.pieces[i].getCoeffMat();
    int i6 = i * 6;
    for (int j = 0; j < 6; j++) {
      poly_msg.coef_x[i6 + j] = cMat(0, j);
      poly_msg.coef_y[i6 + j] = cMat(1, j);
      poly_msg.coef_z[i6 + j] = cMat(2, j);
    }
  }
}

void FastPlannerManager::calculateTimelb(
    const vector<Eigen::Vector3d> &path2next_goal, const double &current_yaw,
    const double &goal_yaw, double &time_lb) {
  double start2fwd = 0.0, fwd2end = 0.0;
  if (path2next_goal.size() == 2) {
    Eigen::Vector3d diff = path2next_goal.back() - path2next_goal.front();
    if (pow(diff.x(), 2) + pow(diff.y(), 2) >= pow(diff.z(), 2) &&
        (diff.squaredNorm() - pow(diff.z(), 2) > 0.01)) {
      double fwd_yaw = atan2(diff.y(), diff.x());
      start2fwd = fwd_yaw - current_yaw;
      angleLimite(start2fwd);
      fwd2end = goal_yaw - fwd_yaw;
      angleLimite(fwd2end);

    } else {
      double diff = goal_yaw - current_yaw;
      angleLimite(diff);
      time_lb = fabs(diff) / gcopter_config_->yaw_max_vel;
      return;
    }
  } else {
    Eigen::Vector3d diff = path2next_goal[1] - path2next_goal[0];
    if (pow(diff.x(), 2) + pow(diff.y(), 2) >= pow(diff.z(), 2)) {
      double fwd_yaw = atan2(diff.y(), diff.x());
      start2fwd = fwd_yaw - current_yaw;
      angleLimite(start2fwd);
    } else {
      start2fwd = 0.0;
    }
    diff = path2next_goal[path2next_goal.size() - 1] -
           path2next_goal[path2next_goal.size() - 2];
    if (pow(diff.x(), 2) + pow(diff.y(), 2) >= pow(diff.z(), 2)) {
      double fwd_yaw = atan2(diff.y(), diff.x());
      fwd2end = fwd_yaw - goal_yaw;
      angleLimite(fwd2end);
    } else {
      fwd2end = 0.0;
    }
  }

  time_lb = fabs(start2fwd) / gcopter_config_->yaw_max_vel +
            fabs(fwd2end) / gcopter_config_->yaw_max_vel;
  return;
}
} // namespace fast_planner
