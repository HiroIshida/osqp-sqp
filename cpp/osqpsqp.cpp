#include "osqpsqp.hpp"
#include <osqp++.h>

namespace osqpsqp {

bool EqualityConstraintBase::evaluate_full(const Eigen::VectorXd &x,
                                           Eigen::VectorXd &values,
                                           SMatrix &jacobian,
                                           Eigen::VectorXd &lower,
                                           Eigen::VectorXd &upper,
                                           size_t constraint_idx_head) {
  evaluate(x, values, jacobian, constraint_idx_head);
  auto jac_sliced = jacobian.middleRows(constraint_idx_head, get_cdim());
  auto value_sliced = values.segment(constraint_idx_head, get_cdim());
  auto tmp = jac_sliced * x - value_sliced;
  lower.segment(constraint_idx_head, get_cdim()) = tmp;
  upper.segment(constraint_idx_head, get_cdim()) = tmp;

  double max_error = value_sliced.array().abs().maxCoeff();
  bool is_feasible = max_error < tol_;
  return is_feasible;
}

bool InequalityConstraintBase::evaluate_full(const Eigen::VectorXd &x,
                                             Eigen::VectorXd &values,
                                             SMatrix &jacobian,
                                             Eigen::VectorXd &lower,
                                             Eigen::VectorXd &upper,
                                             size_t constraint_idx_head) {
  evaluate(x, values, jacobian, constraint_idx_head);
  auto jac_sliced = jacobian.middleRows(constraint_idx_head, get_cdim());
  auto value_sliced = values.segment(constraint_idx_head, get_cdim());
  lower.segment(constraint_idx_head, get_cdim()) =
      jac_sliced * x - value_sliced;
  upper.segment(constraint_idx_head, get_cdim()) = Eigen::VectorXd::Constant(
      get_cdim(), std::numeric_limits<double>::infinity());

  bool is_feasible = (value_sliced.array() > -tol_).all();
  return is_feasible;
}

void BoxConstraint::evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                             SMatrix &jacobian, size_t constraint_idx_head) {
  values.segment(constraint_idx_head, get_cdim()) = x;
  for (size_t i = 0; i < get_cdim(); i++) {
    jacobian.coeffRef(constraint_idx_head + i, i) = 1.0;
  }
}

bool BoxConstraint::evaluate_full(const Eigen::VectorXd &x,
                                  Eigen::VectorXd &values, SMatrix &jacobian,
                                  Eigen::VectorXd &lower,
                                  Eigen::VectorXd &upper,
                                  size_t constraint_idx_head) {
  evaluate(x, values, jacobian, constraint_idx_head);
  lower.segment(constraint_idx_head, get_cdim()) = lb_;
  upper.segment(constraint_idx_head, get_cdim()) = ub_;
  auto value_sliced = values.segment(constraint_idx_head, get_cdim());
  bool is_feasible = (value_sliced.array() >= lb_.array() - tol_).all() &&
                     (value_sliced.array() <= ub_.array() + tol_).all();
  return is_feasible;
}

bool ConstraintSet::evaluate_full(const Eigen::VectorXd &x,
                                  Eigen::VectorXd &values, SMatrix &jacobian,
                                  Eigen::VectorXd &lower,
                                  Eigen::VectorXd &upper) {
  size_t constraint_idx_head = 0;
  bool is_feasible = true;
  for (auto c : constraints_) {
    bool is_feasible_partial = c->evaluate_full(x, values, jacobian, lower,
                                                upper, constraint_idx_head);
    constraint_idx_head += c->get_cdim();
    is_feasible = is_feasible && is_feasible_partial;
  }
  return is_feasible;
}

size_t ConstraintSet::get_cdim() {
  size_t cdim = 0;
  for (auto c : constraints_) {
    cdim += c->get_cdim();
  }
  return cdim;
};

NLPSolver::NLPSolver(size_t nx, SMatrix P, Eigen::VectorXd q,
                     std::shared_ptr<ConstraintSet> cstset,
                     const NLPSolverOption &option)
    : P_(P), q_(q), cstset_(cstset), cstset_lower_(cstset->get_cdim()),
      cstset_upper_(cstset->get_cdim()), option_(option) {
  Eigen::VectorXd x_dummy = Eigen::VectorXd::Zero(nx);
  Eigen::VectorXd values(cstset->get_cdim());
  cstset_jacobian_ = SMatrix(cstset->get_cdim(), nx);
  cstset_values_ = Eigen::VectorXd(cstset->get_cdim());
  cstset_lower_ = Eigen::VectorXd(cstset->get_cdim());
  cstset_upper_ = Eigen::VectorXd(cstset->get_cdim());
  cstset_->evaluate_full(x_dummy, cstset_values_, cstset_jacobian_,
                         cstset_lower_, cstset_upper_);
}

void NLPSolver::solve(const Eigen::VectorXd &x0) {
  Eigen::VectorXd x = x0;
  double cost_prev = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < option_.max_iter; i++) {
    bool is_feasible = cstset_->evaluate_full(
        x, cstset_values_, cstset_jacobian_, cstset_lower_, cstset_upper_);
    double cost = (0.5 * x.transpose() * P_ * x + q_.transpose() * x)(0);
    double cost_diff = cost - cost_prev;
    bool ftol_satisfied =
        cost_diff <
        option_.ftol.value_or(std::numeric_limits<double>::infinity());
    if (is_feasible && ftol_satisfied) {
      break;
    }
    cost_prev = cost;

    Eigen::VectorXd cost_grad = P_ * x + q_;

    osqp::OsqpInstance instance;
    instance.objective_matrix = P_;
    instance.objective_vector = q_;
    instance.constraint_matrix = cstset_jacobian_;
    instance.lower_bounds = cstset_lower_;
    instance.upper_bounds = cstset_upper_;

    osqp::OsqpSolver solver;
    osqp::OsqpSettings settings;
    settings.verbose = option_.osqp_verbose;
    if (option_.osqp_force_deterministic) {
      settings.adaptive_rho_interval = 25.0;
    }

    const auto init_status = solver.Init(instance, settings);
    const auto osqp_exit_code = solver.Solve();
    Eigen::Map<const Eigen::VectorXd> primal_solution =
        solver.primal_solution();
    x = primal_solution;
    // TODO: do we need to update the dual solution?
    // Eigen::Map<const Eigen::VectorXd> dual_solution =
    // solver.dual_solution();
    std::cout << x << std::endl;
  }
}

} // namespace osqpsqp
