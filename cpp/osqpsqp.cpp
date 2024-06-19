#include "osqpsqp.hpp"
#include <chrono>
#include <osqp++.h>
#include <sstream>
#include <string>

namespace osqpsqp {

bool ConstraintBase::check_jacobian(const Eigen::VectorXd &x, double eps,
                                    bool verbose) {
  SMatrix ana_jac = SMatrix(get_cdim(), nx_);
  Eigen::MatrixXd num_jac = SMatrix(get_cdim(), nx_);
  ana_jac.setZero();
  num_jac.setZero();

  { // numerical differentiation
    SMatrix whatever = SMatrix(get_cdim(), nx_);
    Eigen::VectorXd f0 = Eigen::VectorXd::Zero(get_cdim());
    evaluate(x, f0, whatever, 0);

    for (size_t i = 0; i < nx_; i++) {
      Eigen::VectorXd x_plus = x;
      x_plus(i) += eps;
      Eigen::VectorXd f1 = Eigen::VectorXd::Zero(get_cdim());
      evaluate(x_plus, f1, whatever, 0);
      num_jac.col(i) = (f1 - f0) * (1 / eps);
    }
  }

  { // analytical differentiation
    Eigen::VectorXd whatever = Eigen::VectorXd::Zero(get_cdim());
    evaluate(x, whatever, ana_jac, 0);
  }

  double max_diff = (num_jac - ana_jac.toDense()).cwiseAbs().maxCoeff();
  double check_eps = eps * 10;
  bool ok = max_diff < check_eps;
  if (verbose && !ok) {
    std::cout << "max_diff: " << max_diff << std::endl;
    std::cout << "check_eps: " << check_eps << std::endl;
    std::cout << "num_jac: " << std::endl << num_jac << std::endl;
    std::cout << "ana_jac: " << std::endl << ana_jac.toDense() << std::endl;
  }
  return ok;
}

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
  if (verbose_) {
    std::stringstream ss;
    ss << "eqconst(" << name_ << ") feasiblity: " << is_feasible;
    std::cout << ss.str() << std::endl;
  }
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
  if (verbose_) {
    std::stringstream ss;
    ss << "ineqconst(" << name_ << ") feasiblity: " << is_feasible;
    std::cout << ss.str() << std::endl;
  }
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
  if (verbose_) {
    std::cout << "box feasiblity: " << is_feasible << std::endl;
  }
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
  auto start = std::chrono::high_resolution_clock::now();

  osqp::OsqpSettings settings;
  settings.verbose = option_.osqp_verbose;
  settings.eps_abs = option_.osqp_eps_abs;
  settings.eps_rel = option_.osqp_eps_rel;
  if (option_.osqp_force_deterministic) {
    settings.adaptive_rho_interval = 25.0;
  }

  solution_ = x0;
  double cost_prev = std::numeric_limits<double>::infinity();

  for (size_t iter = 0; iter < option_.max_iter; iter++) {
    std::cout << "iteration: " << iter << std::endl;
    bool is_feasible =
        cstset_->evaluate_full(solution_, cstset_values_, cstset_jacobian_,
                               cstset_lower_, cstset_upper_);
    double cost = (0.5 * solution_.transpose() * P_ * solution_ +
                   q_.transpose() * solution_)(0);
    double cost_diff = cost - cost_prev;
    bool ftol_satisfied =
        cost_diff <
        option_.ftol.value_or(std::numeric_limits<double>::infinity());
    if (is_feasible && ftol_satisfied) {
      auto end = std::chrono::high_resolution_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      std::cout << "time: " << duration.count() << "ms" << std::endl;
      std::cout << "solved!" << std::endl;
      break;
    }
    cost_prev = cost;

    Eigen::VectorXd cost_grad = P_ * solution_ + q_;

    osqp::OsqpInstance instance;
    instance.objective_matrix = P_;
    instance.objective_vector = q_;
    instance.constraint_matrix = cstset_jacobian_;
    instance.lower_bounds = cstset_lower_;
    instance.upper_bounds = cstset_upper_;

    osqp::OsqpSolver solver;

    if (iter == 0) {
      for (size_t qp_relax_iter = 0; qp_relax_iter < option_.max_relax_iter;
           qp_relax_iter++) {
        if (qp_relax_iter > 0) {
          instance.lower_bounds -= Eigen::VectorXd::Constant(
              cstset_->get_cdim(), option_.relaxation);
          instance.upper_bounds += Eigen::VectorXd::Constant(
              cstset_->get_cdim(), option_.relaxation);
          std::cout << "relax!" << std::endl;
        }
        const auto init_status = solver.Init(instance, settings);
        const auto osqp_exit_code = solver.Solve();
        if (osqp_exit_code == osqp::OsqpExitCode::kOptimal) {
          break;
        }
        if (qp_relax_iter == option_.max_relax_iter - 1) {
          throw std::runtime_error("OSQP failed to solve the problem.");
        }
      }
    } else {
      const auto init_status = solver.Init(instance, settings);
      const auto osqp_exit_code = solver.Solve();
      if (osqp_exit_code != osqp::OsqpExitCode::kOptimal) {
        // throw std::runtime_error("OSQP failed to solve the problem.");
        std::cout << "failed to solve" << std::endl;
        return;
      }
    }

    Eigen::Map<const Eigen::VectorXd> primal_solution =
        solver.primal_solution();
    solution_ = primal_solution;
    // TODO: do we need to update the dual solution?
    // Eigen::Map<const Eigen::VectorXd> dual_solution =
    // solver.dual_solution();
  }
  // std::cout << solution_ << std::endl;
}

} // namespace osqpsqp
