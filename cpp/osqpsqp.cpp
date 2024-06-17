#include "osqpsqp.hpp"
#include <osqp++.h>

namespace osqpsqp {

void EqualityConstraintInterface::evaluate_full(const Eigen::VectorXd &x,
                                                Eigen::VectorXd &values,
                                                SMatrix &jacobian,
                                                Eigen::VectorXd &lower,
                                                Eigen::VectorXd &upper,
                                                size_t constraint_idx_head) {
  evaluate(x, values, jacobian, constraint_idx_head);
  auto jac_sliced = jacobian.middleRows(constraint_idx_head, get_cdim());
  auto tmp = jac_sliced * x - values.segment(constraint_idx_head, get_cdim());
  lower.segment(constraint_idx_head, get_cdim()) = tmp;
  upper.segment(constraint_idx_head, get_cdim()) = tmp;
}

void InequalityConstraintInterface::evaluate_full(const Eigen::VectorXd &x,
                                                  Eigen::VectorXd &values,
                                                  SMatrix &jacobian,
                                                  Eigen::VectorXd &lower,
                                                  Eigen::VectorXd &upper,
                                                  size_t constraint_idx_head) {
  evaluate(x, values, jacobian, constraint_idx_head);
  auto jac_sliced = jacobian.middleRows(constraint_idx_head, get_cdim());
  lower.segment(constraint_idx_head, get_cdim()) =
      jac_sliced * x - values.segment(constraint_idx_head, get_cdim());
  upper.segment(constraint_idx_head, get_cdim()) = Eigen::VectorXd::Constant(
      get_cdim(), std::numeric_limits<double>::infinity());
}

void BoxConstraint::evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                             SMatrix &jacobian, size_t constraint_idx_head) {
  values.segment(constraint_idx_head, get_cdim()) = x;
  for (size_t i = 0; i < get_cdim(); i++) {
    jacobian.coeffRef(constraint_idx_head + i, i) = 1.0;
  }
}

void BoxConstraint::evaluate_full(const Eigen::VectorXd &x,
                                  Eigen::VectorXd &values, SMatrix &jacobian,
                                  Eigen::VectorXd &lower,
                                  Eigen::VectorXd &upper,
                                  size_t constraint_idx_head) {
  evaluate(x, values, jacobian, constraint_idx_head);
  lower.segment(constraint_idx_head, get_cdim()) = lb_;
  upper.segment(constraint_idx_head, get_cdim()) = ub_;
}

void ConstraintSet::evaluate_full(const Eigen::VectorXd &x,
                                  Eigen::VectorXd &values, SMatrix &jacobian,
                                  Eigen::VectorXd &lower,
                                  Eigen::VectorXd &upper) {
  size_t constraint_idx_head = 0;
  for (auto c : constraints_) {
    c->evaluate_full(x, values, jacobian, lower, upper, constraint_idx_head);
    constraint_idx_head += c->get_cdim();
  }
}

size_t ConstraintSet::get_cdim() {
  size_t cdim = 0;
  for (auto c : constraints_) {
    cdim += c->get_cdim();
  }
  return cdim;
};

NLPSolver::NLPSolver(size_t nx, SMatrix P, Eigen::VectorXd q,
                     std::shared_ptr<ConstraintSet> cstset)
    : P_(P), q_(q), cstset_(cstset), cstset_lower_(cstset->get_cdim()),
      cstset_upper_(cstset->get_cdim()) {
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
  size_t max_iter = 10;
  for (size_t i = 0; i < max_iter; i++) {
    cstset_->evaluate_full(x, cstset_values_, cstset_jacobian_, cstset_lower_,
                           cstset_upper_);
    double cost = (0.5 * x.transpose() * P_ * x + q_.transpose() * x)(0);
    Eigen::VectorXd cost_grad = P_ * x + q_;

    osqp::OsqpInstance instance;
    instance.objective_matrix = P_;
    instance.objective_vector = q_;
    instance.constraint_matrix = cstset_jacobian_;
    instance.lower_bounds = cstset_lower_;
    instance.upper_bounds = cstset_upper_;

    osqp::OsqpSolver solver;
    osqp::OsqpSettings settings;
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