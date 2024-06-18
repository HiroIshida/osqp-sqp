#include "osqpsqp.hpp"
#include <iostream>

using namespace osqpsqp;


void add_identity_to_sparse_jacobian(
    SMatrix &jac,
    size_t start_i, size_t start_j, size_t size, double identity_coef) {
  // check if the size is valid
  for(size_t i = 0; i < size; i++) {
    jac.coeffRef(start_i + i, start_j + i) += identity_coef;
  }
}

class DifferentialConstraint : public EqualityConstraintBase {
  public:
  DifferentialConstraint(size_t T, double dt) : 
    EqualityConstraintBase(6 * T, "diff", 1e-6, true), T_(T), dt_(dt) {}

  size_t get_cdim() override {
    return 4 * (T_ - 1);
  }

  void evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                SMatrix &jacobian, size_t constraint_idx_head) override {
    // x is a vector of size 6*T . 6 consists of (position, velocity, acceleration) for each time step

    // fill values
    size_t c_head = constraint_idx_head;
    for(size_t t = 0; t < T_ - 1; t++) {
      size_t x_head_now = t * 6;
      size_t x_head_next = (t + 1) * 6;
      auto x_now = x.segment(x_head_now, 2);
      auto v_now = x.segment(x_head_now + 2, 2);
      auto a_now = x.segment(x_head_now + 4, 2);
      auto x_next = x.segment(x_head_next, 2);
      auto v_next = x.segment(x_head_next + 2, 2);
      values.segment(c_head, 2) = x_next - (x_now + v_now * dt_ + 0.5 * a_now * dt_ * dt_);
      values.segment(c_head + 2, 2) = v_next - (v_now + a_now * dt_);
      c_head += 4;
    }

    // fill jacobian matrix
    size_t i_head = constraint_idx_head;
    for(size_t t = 0; t < T_ - 1; t++) {
      size_t j_head = 6 * t;
      // diff-const for x_t+1 = x_t + v_t * dt + 0.5 * a_t * dt^2
      add_identity_to_sparse_jacobian(jacobian, i_head, j_head, 2, -1.0);
      add_identity_to_sparse_jacobian(jacobian, i_head, j_head + 2, 2, -dt_);
      add_identity_to_sparse_jacobian(jacobian, i_head, j_head + 4, 2, -0.5 * dt_ * dt_);
      add_identity_to_sparse_jacobian(jacobian, i_head, j_head + 6, 2, 1.0);

      // diff-const for v_t+1 = v_t + a_t * dt
      add_identity_to_sparse_jacobian(jacobian, i_head + 2, j_head + 2, 2, -1.0);
      add_identity_to_sparse_jacobian(jacobian, i_head + 2, j_head + 4, 2, -dt_);
      add_identity_to_sparse_jacobian(jacobian, i_head + 2, j_head + 8, 2, 1.0);

      i_head += 4;
    }
  }

  private:
  size_t T_;
  double dt_;
};

class EndPointsConstraint : public EqualityConstraintBase {
  public:
  EndPointsConstraint(size_t T, 
      const Eigen::Vector2d &start,
      const Eigen::Vector2d &goal)
      : EqualityConstraintBase(6 * T, "end-point", 1e-6, true), T_(T), start_(start), goal_(goal) {}

  size_t get_cdim() override {
    return 8;
  }

  void evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                SMatrix &jacobian, size_t constraint_idx_head) override {
    // fill values (start)
    size_t c_head = constraint_idx_head;
    size_t x_head = 0;
    values.segment(c_head, 2) = x.segment(x_head, 2) - start_;
    values.segment(c_head + 2, 2) = x.segment(x_head + 2, 2);

    // fill values (goal)
    c_head += 4;
    x_head = (T_ - 1) * 6;
    values.segment(c_head, 2) = x.segment(x_head, 2) - goal_;
    values.segment(c_head + 2, 2) = x.segment(x_head + 2, 2);

    // fill jacobian matrix (start)
    size_t i_head = constraint_idx_head;
    size_t j_head = 0;
    add_identity_to_sparse_jacobian(jacobian, i_head, j_head, 2, 1.0);
    add_identity_to_sparse_jacobian(jacobian, i_head + 2, j_head + 2, 2, 1.0);

    // fill jacobian matrix (goal)
    i_head += 4;
    j_head = 6 * (T_ - 1);
    add_identity_to_sparse_jacobian(jacobian, i_head, j_head, 2, 1.0);
    add_identity_to_sparse_jacobian(jacobian, i_head + 2, j_head + 2, 2, 1.0);
  }

  private:
  size_t T_;
  Eigen::VectorXd start_;
  Eigen::VectorXd goal_;
};

int main() {
  size_t T = 60;
  auto diff_con = std::make_shared<DifferentialConstraint>(T, 0.1);
  if(!diff_con->check_jacobian(1e-6, true)) {
    std::cout << "Jacobian is wrong" << std::endl;
  }
  auto goal_con = std::make_shared<EndPointsConstraint>(T, Eigen::Vector2d(0.1, 0.1), Eigen::Vector2d(0.5, 0.5));
  if(!goal_con->check_jacobian(1e-6, true)) {
    std::cout << "Jacobian is wrong" << std::endl;
  }
  auto cstset = std::make_shared<ConstraintSet>();
  cstset->add(diff_con);
  cstset->add(goal_con);

  // prepare quadratic cost \sum_{i \in T} u_i^2
  SMatrix P(cstset->nx_, cstset->nx_);
  for(size_t i = 0; i < T; i++) {
    P.coeffRef(6 * i + 4, 6 * i + 4) = 1.0;
    P.coeffRef(6 * i + 5, 6 * i + 5) = 1.0;
  }
  auto solver = NLPSolver(cstset->nx_, P, Eigen::VectorXd::Zero(cstset->nx_), cstset);
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(cstset->nx_);
  solver.solve(x0);
}
