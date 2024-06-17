#include "osqp_sqp.hpp"
#include <memory>

using namespace osqp_sqp;

class MyEqConst : public InequalityConstraintInterface {
public:
  MyEqConst() {
    values.resize(1);
    jacobian.resize(1, 2);
  }

  void evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                SMatrix &jacobian,
                size_t constraint_idx_head) {
    auto head = constraint_idx_head;
    double x0 = x(0);
    double x1 = x(1);
    values(head) = x1 - x0 * x0;
    jacobian.coeffRef(head, 0) = -2.0 * x(0);
    jacobian.coeffRef(head, 1) = 1.0;
  }

  size_t get_cdim() { return 1; }

  const SMatrix &get_jacobian() { return jacobian; }
  const Eigen::VectorXd &get_values() { return values; }

  Eigen::VectorXd values;
  SMatrix jacobian;
};

class MyIneqConst : public EqualityConstraintInterface {
public:
  MyIneqConst() {
  }

  void evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                SMatrix &jacobian,
                size_t constraint_idx_head) {
    double x0 = x(0);
    double x1 = x(1);
    values(constraint_idx_head) = - ((x0 - 1.0) * (x0 - 1.0) + (x1 - 1.0) * (x1 - 1.0) - 1.0);
    jacobian.coeffRef(0, 0) = -2.0 * (x(0) - 1.0);
    jacobian.coeffRef(0, 1) = -2.0 * (x(1) - 1.0);
  }
  size_t get_cdim() { return 1; }
};


int main() {
  Eigen::VectorXd lb = Eigen::VectorXd::Constant(2, -3.0);
  Eigen::VectorXd ub = Eigen::VectorXd::Constant(2, 3.0);


  std::shared_ptr<ConstraintInterface> box = std::make_shared<BoxConstraint>(lb, ub);
  std::shared_ptr<ConstraintInterface> eq = std::make_shared<MyEqConst>();
  std::shared_ptr<ConstraintInterface> ineq = std::make_shared<MyIneqConst>();
  std::vector<std::shared_ptr<ConstraintInterface>> constraints{ineq, eq};
  auto cst = std::make_shared<ConstraintSet>(constraints);
  std::cout << cst->get_cdim() << std::endl;

  SMatrix P(2, 2);
  P.coeffRef(0, 0) = 1.0;
  P.coeffRef(1, 1) = 1.0;

  auto solver = NLPSolver(2, P, Eigen::VectorXd::Zero(2), cst);
  solver.solve(Eigen::VectorXd::Zero(2));
}
