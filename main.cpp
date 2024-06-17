#include "osqpsqp.hpp"
#include <memory>

using namespace osqpsqp;

class MyEqConst : public EqualityConstraintInterface {
public:
  void evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                SMatrix &jacobian, size_t constraint_idx_head) {
    auto head = constraint_idx_head;
    double x0 = x(0);
    double x1 = x(1);
    values(head) = x1 - x0 * x0;
    jacobian.coeffRef(head, 0) = -2.0 * x(0);
    jacobian.coeffRef(head, 1) = 1.0;
  }
  size_t get_cdim() { return 1; }
};

class MyIneqConst : public InequalityConstraintInterface {
public:
  void evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                SMatrix &jacobian, size_t constraint_idx_head) {
    double x0 = x(0);
    double x1 = x(1);
    values(constraint_idx_head) =
        -((x0 - 1.0) * (x0 - 1.0) + (x1 - 1.0) * (x1 - 1.0) - 1.0);
    jacobian.coeffRef(constraint_idx_head, 0) = -2.0 * (x(0) - 1.0);
    jacobian.coeffRef(constraint_idx_head, 1) = -2.0 * (x(1) - 1.0);
  }
  size_t get_cdim() { return 1; }
};

int main() {
  Eigen::VectorXd lb = Eigen::VectorXd::Constant(1, 1);
  Eigen::VectorXd ub = Eigen::VectorXd::Constant(3.0, 3.0);

  auto cstset = std::make_shared<ConstraintSet>();
  cstset->add(std::make_shared<BoxConstraint>(lb, ub));
  cstset->add(std::make_shared<MyIneqConst>());
  cstset->add(std::make_shared<MyEqConst>());
  SMatrix P(2, 2);
  P.coeffRef(0, 0) = 1.0;
  P.coeffRef(1, 1) = 1.0;
  auto solver = NLPSolver(2, P, Eigen::VectorXd::Zero(2), cstset);
  solver.solve(Eigen::VectorXd::Zero(2));
}
