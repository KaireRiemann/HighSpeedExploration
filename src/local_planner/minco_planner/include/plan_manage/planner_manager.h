#ifndef _PLANNER_MANAGER_H_
#define _PLANNER_MANAGER_H_

#include <path_searching/bubble_astar.h>

#include <algorithm>
#include <plan_manage/plan_container.hpp>
#include <ros/ros.h>
#include <traj_utils/PolyTraj.h>
#include <lidar_map/lidar_map.h>
#include <random>
#include "gcopter/firi.hpp"
#include "gcopter/flatness.hpp"
#include "gcopter/sfc_gen.hpp"
#include "gcopter/trajectory.hpp"
#include "gcopter/voxel_map.hpp"
#include "misc/visualizer.hpp"

#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <pointcloud_topo/graph.h>
#include <pointcloud_topo/graph_visualizer.hpp>
#include <pointcloud_topo/parallel_bubble_astar.h>
#include <tf/tf.h>

namespace fast_planner {
namespace traj_opt {
class TrajManager;
}

// Fast Planner Manager
// Key algorithms of mapping and planning are called
struct GcopterConfig {
  std::string mapTopic;
  std::string targetTopic;
  double dilateRadiusSoft, dilateRadiusHard;
  double timeoutRRT;
  double maxVelMag;
  double maxAccMag;
  double maxBdrMag;
  double maxTiltAngle;
  double minThrust;
  double maxThrust;
  double vehicleMass;
  double gravAcc;
  double horizDrag;
  double vertDrag;
  double parasDrag;
  double speedEps;
  double weightT;
  double WeightSafeT;
  double energyWeight;
  std::vector<double> chiVec;
  double smoothingEps;
  int integralIntervs;
  double relCostTol;
  double corridor_size;
  double yaw_max_vel;
  double yaw_rho_vis;
  double yaw_time_fwd;
  bool dynamicVelocityEnable;
  double minSegmentVel;
  double openSegmentVel;
  double dynamicVelocityMinClearance;
  double dynamicVelocityOpenClearance;
  double dynamicVelocityClearanceMargin;
  bool nonstopTerminalVelocityEnable;
  double nonstopTerminalVelocityRatio;
  double nonstopTerminalMinPathLength;
  bool backupTrajEnable;
  double backupStartRatio;
  double backupMinStartTime;
  double backupMaxStartTime;
  double backupSampleDt;
  double backupSearchMargin;
  int backupPieceNum;
  double backupMaxVel;
  double backupMaxAcc;

  void init(const ros::NodeHandle &nh_priv) {
    nh_priv.getParam("DilateRadiusSoft", dilateRadiusSoft);
    nh_priv.getParam("DilateRadiusHard", dilateRadiusHard);
    nh_priv.getParam("MaxVelMag", maxVelMag);
    nh_priv.param("MaxAccMag", maxAccMag, 20.0);
    nh_priv.getParam("maxBdrMag", maxBdrMag);
    nh_priv.getParam("MaxTiltAngle", maxTiltAngle);
    nh_priv.getParam("MinThrust", minThrust);
    nh_priv.getParam("MaxThrust", maxThrust);
    nh_priv.getParam("VehicleMass", vehicleMass);
    nh_priv.getParam("GravAcc", gravAcc);
    nh_priv.getParam("HorizDrag", horizDrag);
    nh_priv.getParam("VertDrag", vertDrag);
    nh_priv.getParam("ParasDrag", parasDrag);
    nh_priv.getParam("SpeedEps", speedEps);
    nh_priv.getParam("WeightT", weightT);
    nh_priv.getParam("WeightSafeT", WeightSafeT);
    nh_priv.param("EnergyWeight", energyWeight, 1.0);
    nh_priv.getParam("ChiVec", chiVec);
    nh_priv.getParam("SmoothingEps", smoothingEps);
    nh_priv.getParam("IntegralIntervs", integralIntervs);
    nh_priv.getParam("RelCostTol", relCostTol);
    nh_priv.getParam("MaxCorridorSize", corridor_size);
    nh_priv.getParam("yaw_rho_vis", yaw_rho_vis);
    nh_priv.getParam("yaw_max_vel", yaw_max_vel);
    nh_priv.getParam("yaw_time_fwd", yaw_time_fwd);
    nh_priv.param("DynamicVelocityEnable", dynamicVelocityEnable, true);
    nh_priv.param("MinSegmentVel", minSegmentVel, 2.5);
    nh_priv.param("OpenSegmentVel", openSegmentVel, maxVelMag);
    nh_priv.param("DynamicVelocityMinClearance", dynamicVelocityMinClearance, 0.35);
    nh_priv.param("DynamicVelocityOpenClearance", dynamicVelocityOpenClearance, 2.5);
    nh_priv.param("DynamicVelocityClearanceMargin", dynamicVelocityClearanceMargin,
                  dilateRadiusHard + 0.15);
    nh_priv.param("NonstopTerminalVelocityEnable", nonstopTerminalVelocityEnable, false);
    nh_priv.param("NonstopTerminalVelocityRatio", nonstopTerminalVelocityRatio, 0.65);
    nh_priv.param("NonstopTerminalMinPathLength", nonstopTerminalMinPathLength, 8.0);
    nh_priv.param("BackupTrajEnable", backupTrajEnable, true);
    nh_priv.param("BackupStartRatio", backupStartRatio, 0.55);
    nh_priv.param("BackupMinStartTime", backupMinStartTime, 0.65);
    nh_priv.param("BackupMaxStartTime", backupMaxStartTime, 2.2);
    nh_priv.param("BackupSampleDt", backupSampleDt, 0.08);
    nh_priv.param("BackupSearchMargin", backupSearchMargin, 4.0);
    nh_priv.param("BackupPieceNum", backupPieceNum, 2);
    nh_priv.param("BackupMaxVel", backupMaxVel, std::min(maxVelMag, 6.0));
    nh_priv.param("BackupMaxAcc", backupMaxAcc, maxAccMag);
  }
};

class FastPlannerManager {
  // SECTION stable
public:
  typedef shared_ptr<FastPlannerManager> Ptr;
  FastPlannerManager();
  ~FastPlannerManager();
  void printTimeCost(double time_threhold, double time_cost, string printInfo);

  bool planExploreTraj(const vector<Eigen::Vector3f> &path, bool is_static);
  bool flyToSafeRegion(bool is_static);
  void polyTraj2ROSMsg(traj_utils::PolyTraj &poly_msg, const ros::Time &start_time);
  void polyYawTraj2ROSMsg(traj_utils::PolyTraj &poly_msg, const ros::Time &start_time);

  void initPlanModules(ros::NodeHandle &nh, ParallelBubbleAstar::Ptr &parallel_path_finder,
                       TopoGraph::Ptr &graph);

  bool checkTrajCollision(double &collision_time);
  bool checkTrajVelocity();
  bool hasCommittedBackup() const;
  double timeToCommittedBackup() const;

  bool YawTrajOpt(double &start_yaw, double &end_yaw, bool is_static, bool use_shorten_path);
  bool YawTrajwithoutOpt(double &start_yaw, double &end_yaw, bool is_static, bool use_shorten_path);
  void goalCallback(const geometry_msgs::PoseStampedConstPtr &msg);
  void posCallback(const nav_msgs::OdometryConstPtr &msg);
  bool YawInterpolationwithoutOpt(double &start, double &end, vector<double> &newYaw,
                                  vector<double> &newDur, double &CompT);
  void YawLookforward(const Trajectory<5> &pos_traj, double &start, double &end,
                      vector<double> &newYaw, vector<double> &newDur, double &CompT);
  void YawLookforwardwithoutOpt(double &start, double &end, vector<double> &newYaw,
                                vector<double> &newDur, double &CompT, bool use_short_path);
  void angleLimite(double &angle);
  void calculateTimelb(const vector<Eigen::Vector3d> &path2next_goal,
                                 const double &current_yaw, const double &goal_yaw, double &time_lb);

  double start_yaw, end_yaw;
  double is_static_yaw = false;

  ros::Subscriber goal_sub;
  ros::Subscriber pos_sub;
  ros::Publisher yaw_state_pub;

  LocalTrajData local_data_;
  double max_traj_len_;
  LIOInterface::Ptr lidar_map_interface_;
  unique_ptr<Visualizer> gcopter_viz_;
  unique_ptr<GcopterConfig> gcopter_config_;
  unique_ptr<traj_opt::TrajManager> traj_manager_;
  BubbleAstar::Ptr bubble_path_finder_;
  ParallelBubbleAstar::Ptr parallel_path_finder_;
  TopoGraph::Ptr topo_graph_;
  GraphVisualizer::Ptr graph_visualizer_;
  FastSearcher::Ptr fast_searcher_;
  bool use_mid360;
  double max_ray_length;
  double fov_up, fov_down;
  double lidar_pitch;

private:
  /* main planning algorithms & modules */
  shared_ptr<SDFMap> sdf_map_;

  // topology guided optimization

  void findCollisionRange(vector<Eigen::Vector3d> &colli_start, vector<Eigen::Vector3d> &colli_end,
                          vector<Eigen::Vector3d> &start_pts, vector<Eigen::Vector3d> &end_pts);

  Eigen::MatrixXd paramLocalTraj(double start_t, double &dt, double &duration);
  Eigen::MatrixXd reparamLocalTraj(const double &start_t, const double &duration, const double &dt);

  Eigen::VectorXd computeCorridorVelocityLimits(const vector<Eigen::MatrixX4d> &hPolys,
                                                const vector<Eigen::Vector3d> &path) const;
  bool generateBackupTrajectory(const Trajectory<7> &exp_traj,
                                const Trajectory<5> &exp_yaw_traj,
                                Trajectory<7> &backup_traj,
                                Trajectory<5> &backup_yaw_traj,
                                double &backup_start_t);
  bool commitTrajectory(const Trajectory<7> &exp_traj,
                        const Trajectory<5> &exp_yaw_traj,
                        const Trajectory<7> &backup_traj,
                        const Trajectory<5> &backup_yaw_traj,
                        double backup_start_t,
                        const ros::Time &start_time);

public:
  void planYawActMap(const Eigen::Vector3d &start_yaw);
  void test();
  void searchFrontier(const Eigen::Vector3d &p);

private:
  // Benchmark method, local exploration
public:
  bool localExplore(Eigen::Vector3d start_pt, Eigen::Vector3d start_vel, Eigen::Vector3d start_acc,
                    Eigen::Vector3d end_pt);

  // !SECTION
};
} // namespace fast_planner

#endif
