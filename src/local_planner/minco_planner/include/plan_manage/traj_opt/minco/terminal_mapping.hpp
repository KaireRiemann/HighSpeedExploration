#ifndef EPIC_TRAJ_OPT_MINCO_TERMINAL_MAPPING_HPP
#define EPIC_TRAJ_OPT_MINCO_TERMINAL_MAPPING_HPP

#include <Eigen/Core>

namespace epic_minco {

template <int DIM, int S>
class BoundaryStateMappingBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using BoundaryState = Eigen::Matrix<double, DIM, S>;

  virtual ~BoundaryStateMappingBase() = default;

  virtual bool enabled() const = 0;

  virtual int extraVariableDim() const { return 0; }

  virtual void setInitialExtraVariables(Eigen::Ref<Eigen::VectorXd> extra_vars) const {
    extra_vars.setZero();
  }

  virtual void mapBoundaryStates(const BoundaryState &nominal_head_state,
                                 const BoundaryState &nominal_tail_state,
                                 const Eigen::VectorXd &cache_T,
                                 const Eigen::Ref<const Eigen::VectorXd> &extra_vars,
                                 BoundaryState &mapped_head_state,
                                 BoundaryState &mapped_tail_state) const {
    mapped_head_state = nominal_head_state;
    mapped_tail_state = nominal_tail_state;
    (void)cache_T;
    (void)extra_vars;
  }

  virtual double addExtraVariableCost(const Eigen::Ref<const Eigen::VectorXd> &extra_vars,
                                      Eigen::Ref<Eigen::VectorXd> grad_extra) const {
    grad_extra.setZero();
    (void)extra_vars;
    return 0.0;
  }

  virtual void backwardBoundaryGradient(const BoundaryState &grad_head_state,
                                        const BoundaryState &grad_tail_state,
                                        const Eigen::VectorXd &cache_T,
                                        const Eigen::Ref<const Eigen::VectorXd> &extra_vars,
                                        Eigen::Ref<Eigen::VectorXd> grad_out) const {
    (void)grad_head_state;
    (void)grad_tail_state;
    (void)cache_T;
    (void)extra_vars;
    (void)grad_out;
  }

  virtual void backwardBoundaryTimeGradient(const BoundaryState &grad_head_state,
                                            const BoundaryState &grad_tail_state,
                                            const Eigen::VectorXd &cache_T,
                                            const Eigen::Ref<const Eigen::VectorXd> &extra_vars,
                                            Eigen::Ref<Eigen::VectorXd> grad_by_times) const {
    (void)grad_head_state;
    (void)grad_tail_state;
    (void)cache_T;
    (void)extra_vars;
    (void)grad_by_times;
  }
};

template <int DIM, int S>
using TerminalMappingBase = BoundaryStateMappingBase<DIM, S>;

template <int DIM, int S>
class FixedTerminalMapping final : public BoundaryStateMappingBase<DIM, S> {
 public:
  bool enabled() const override { return false; }
};

}  // namespace epic_minco

#endif
