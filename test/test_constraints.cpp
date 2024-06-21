#include <gtest/gtest.h>
#include <cmath>
#include "osqpsqp.hpp"

using namespace osqpsqp;

class EqDummy : public EqualityConstraintBase {
public:
  using EqualityConstraintBase::EqualityConstraintBase;
  void evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                SMatrix &jacobian, size_t constraint_idx_head) {
    auto head = constraint_idx_head;
    values(head) = x(0) * x(0) + x(0) * x(1);
    values(head + 1) = x(1) * x(1) + x(0) * x(1);
    jacobian.coeffRef(head, 0) = 2 * x(0) + x(1);
    jacobian.coeffRef(head, 1) = x(0);
    jacobian.coeffRef(head + 1, 0) = x(1);
    jacobian.coeffRef(head + 1, 1) = 2 * x(1) + x(0);
  }
  size_t get_cdim() { return 2; }
};

class IneqDummy : public InequalityConstraintBase {
public:
  using InequalityConstraintBase::InequalityConstraintBase;
  void evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                SMatrix &jacobian, size_t constraint_idx_head) {
    auto head = constraint_idx_head;
    values(head) = x(0) + x(0) * x(1);
    values(head + 1) = x(1) + x(0) * x(1);
    jacobian.coeffRef(head, 0) = 1.0 + x(1);
    jacobian.coeffRef(head, 1) = x(0);
    jacobian.coeffRef(head + 1, 0) = x(1);
    jacobian.coeffRef(head + 1, 1) = 1.0 + x(0);
  }
  size_t get_cdim() { return 2; }
};

TEST(ConstraintSet, SQPTest) {
  Eigen::VectorXd lb = Eigen::VectorXd::Constant(2, 1);
  Eigen::VectorXd ub = Eigen::VectorXd::Constant(2, 3.0);
  auto cstset = ConstraintSet();
  cstset.add(std::make_shared<BoxConstraint>(lb, ub));
  cstset.add(std::make_shared<EqDummy>(2));
  cstset.add(std::make_shared<IneqDummy>(2));

  // test jacobian
  for(auto c : cstset.constraints_){
    EXPECT_TRUE(c->check_jacobian(1e-6));
  }

  Eigen::VectorXd x = Eigen::VectorXd::Constant(2, 1.0);
  Eigen::VectorXd values = Eigen::VectorXd::Constant(cstset.get_cdim(), 0.0);
  SMatrix jacobian = SMatrix(cstset.get_cdim(), 2);
  Eigen::VectorXd lower = Eigen::VectorXd::Constant(cstset.get_cdim(), 0.0);
  Eigen::VectorXd upper = Eigen::VectorXd::Constant(cstset.get_cdim(), 0.0);
  cstset.evaluate_full(x, values, jacobian, lower, upper);

  // test jacobian
  Eigen::MatrixXd jacobian_expected = Eigen::MatrixXd::Zero(cstset.get_cdim(), 2);
  jacobian_expected << 1.0, 0.0,
                       0.0, 1.0,
                       2 * x(0) + x(1), x(0),
                       x(1), 2 * x(1) + x(0),
                       1.0 + x(1), x(0),
                       x(1), 1.0 + x(0);
  double max_error = (jacobian.toDense() - jacobian_expected).cwiseAbs().maxCoeff();
  EXPECT_LT(max_error, 1e-10);
  
  // test values
  Eigen::VectorXd values_expected = Eigen::VectorXd::Zero(cstset.get_cdim());
  values_expected << x(0), x(1), x(0) + x(0) * x(1), x(1) + x(0) * x(1), x(0) + x(0) * x(1), x(1) + x(0) * x(1);
  max_error = (values - values_expected).cwiseAbs().maxCoeff();
  EXPECT_LT(max_error, 1e-10);

  // test lower
  Eigen::VectorXd lower_expected = Eigen::VectorXd::Zero(cstset.get_cdim());
  lower_expected << 1.,
                 1.,
                 3 * x(0) + x(1) - (x(0) * x(0) + x(0) * x(1)),
                 3 * x(1) + x(0) - (x(1) * x(1) + x(0) * x(1)),
                 1.0 + x(1) + x(0) - (x(0) + x(0) * x(1)),
                 1.0 + x(0) + x(1) - (x(1) + x(0) * x(1));
  max_error = (lower - lower_expected).cwiseAbs().maxCoeff();
  EXPECT_LT(max_error, 1e-10);

  // test upper
  Eigen::VectorXd upper_expected = Eigen::VectorXd::Zero(cstset.get_cdim());
  upper_expected << 3.,
                 3.,
                 3 * x(0) + x(1) - (x(0) * x(0) + x(0) * x(1)),
                 3 * x(1) + x(0) - (x(1) * x(1) + x(0) * x(1)),
                 std::numeric_limits<double>::infinity(),
                 std::numeric_limits<double>::infinity();
  max_error = (upper.head(4) - upper_expected.head(4)).cwiseAbs().maxCoeff();
  EXPECT_LT(max_error, 1e-10);
  EXPECT_EQ(upper(4), std::numeric_limits<double>::infinity());
  EXPECT_EQ(upper(5), std::numeric_limits<double>::infinity());
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
