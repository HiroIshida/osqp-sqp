#include "osqpsqp.hpp"
#include <gtest/gtest.h>
#include <memory>

using namespace osqpsqp;

class MyEqConst : public EqualityConstraintBase {
public:
  using EqualityConstraintBase::EqualityConstraintBase;
  void evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                SMatrix &jacobian, size_t constraint_idx_head) {
    // y = x^2
    auto head = constraint_idx_head;
    double x0 = x(0);
    double x1 = x(1);
    values(head) = x1 - x0 * x0;
    jacobian.coeffRef(head, 0) = -2.0 * x(0);
    jacobian.coeffRef(head, 1) = 1.0;
  }
  size_t get_cdim() { return 1; }
};

class MyIneqConst : public InequalityConstraintBase {
public:
  using InequalityConstraintBase::InequalityConstraintBase;
  void evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                SMatrix &jacobian, size_t constraint_idx_head) {
    // (x - 1)^2 + (y - 1)^2 <= 1
    double x0 = x(0);
    double x1 = x(1);
    values(constraint_idx_head) =
        -((x0 - 1.0) * (x0 - 1.0) + (x1 - 1.0) * (x1 - 1.0) - 1.0);
    jacobian.coeffRef(constraint_idx_head, 0) = -2.0 * (x(0) - 1.0);
    jacobian.coeffRef(constraint_idx_head, 1) = -2.0 * (x(1) - 1.0);
  }
  size_t get_cdim() { return 1; }
};

void test_solver_simple(bool with_box) {
  auto cstset = std::make_shared<ConstraintSet>();
  if(with_box) {
    Eigen::VectorXd lb = Eigen::VectorXd::Constant(2, 1);
    Eigen::VectorXd ub = Eigen::VectorXd::Constant(2, 3.0);
    cstset->add(std::make_shared<BoxConstraint>(lb, ub, 1e-6));
  }
  cstset->add(std::make_shared<MyIneqConst>(2, "ineq", 1e-6));
  cstset->add(std::make_shared<MyEqConst>(2, "eq", 1e-6));
  SMatrix P(2, 2);
  P.coeffRef(0, 0) = 1.0;
  P.coeffRef(1, 1) = 1.0;
  NLPSolverOption option;
  option.ftol = 1e-6;
  option.osqp_eps_abs = 1e-6;
  option.osqp_eps_rel = 1e-6;
  option.relaxation = 0.3;
  auto solver = NLPSolver(2, P, Eigen::VectorXd::Zero(2), cstset, option);
  auto ret = solver.solve(Eigen::VectorXd::Zero(2));
  EXPECT_EQ(ret, NLPStatus::Success);
  if(with_box) {
    EXPECT_NEAR(solver.solution_(0), 1.0, 1e-5);
    EXPECT_NEAR(solver.solution_(1), 1.0, 1e-5);
  } else {
    EXPECT_NEAR(solver.solution_(0), 0.425787, 1e-5);
    EXPECT_NEAR(solver.solution_(1), 0.181294, 1e-5);
  }
}

TEST(SOLVE_SIMPLE, SQPTEST) {
  test_solver_simple(false);
  test_solver_simple(true);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
