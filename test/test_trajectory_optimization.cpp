#include "osqpsqp.hpp"
#include <iostream>
#include <fstream>
#include <gtest/gtest.h>

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
    EqualityConstraintBase(6 * T, "diff", 1e-4, true), T_(T), dt_(dt) {}

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
      : EqualityConstraintBase(6 * T, "end-point", 1e-2, true), T_(T), start_(start), goal_(goal) {}

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


class CollisionConstraint : public InequalityConstraintBase { 
public:
  struct Circle {
    Eigen::Vector2d center;
    double radius;
  };

  CollisionConstraint(size_t T, double dt) : InequalityConstraintBase(6 * T, "collision", 1e-4, true), T_(T) {
    obstacles_.push_back(Circle{Eigen::Vector2d(0.25, 0.4), 0.2});
    obstacles_.push_back(Circle{Eigen::Vector2d(0.75, 0.7), 0.15});
  }

  void evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                SMatrix &jacobian, size_t constraint_idx_head) override {
    size_t c_head = constraint_idx_head;
    for(size_t t = 0; t < T_; t++) {
      size_t x_head = t * 6;
      auto x_now = x.segment(x_head, 2);
      size_t closest_obstacle_idx = 0;
      double min_dist = std::numeric_limits<double>::max();
      for(size_t i = 0; i < obstacles_.size(); i++) {
        double dist = (x_now - obstacles_[i].center).norm() - obstacles_[i].radius;
        if(dist < min_dist) {
          min_dist = dist;
          closest_obstacle_idx = i;
        }
      }
      values(c_head) = min_dist;
      auto diff = x_now - obstacles_[closest_obstacle_idx].center;
      jacobian.coeffRef(c_head, x_head) = diff(0) / diff.norm();
      jacobian.coeffRef(c_head, x_head + 1) = diff(1) / diff.norm();
      c_head += 1;
    }
  }

  size_t get_cdim() override {
    return 2 * T_ * obstacles_.size();
  }
  size_t T_;
  std::vector<Circle> obstacles_;
};

TEST(SOLVE_TRAJOPT, SQPTEST){
  size_t T = 120;
  double dt = 0.2;
  Eigen::Vector2d start(0.1, 0.1);
  Eigen::Vector2d goal(0.9, 0.95);  // make asymmetric

  auto diff_con = std::make_shared<DifferentialConstraint>(T, dt);
  if(!diff_con->check_jacobian(1e-6, true)) {
    throw std::runtime_error("diff eq is wrong");
  }
  auto goal_con = std::make_shared<EndPointsConstraint>(T, start, goal);
  if(!goal_con->check_jacobian(1e-6, true)) {
    throw std::runtime_error("end eq is wrong");
  }

  auto collision_con = std::make_shared<CollisionConstraint>(T, dt);
  if(!goal_con->check_jacobian(1e-6, true)) {
    throw std::runtime_error("coll ineq is wrong");
  }

  // create box constraint
  Eigen::VectorXd lb = Eigen::VectorXd::Zero(T * 6);
  Eigen::VectorXd ub = Eigen::VectorXd::Zero(T * 6);
  for(size_t i = 0; i < T; i++) {
    // 0 < x < 1, -0.3 < v < 0.3, -0.1 < a < 0.1
    lb.segment(6 * i, 2) = Eigen::Vector2d(0.0, 0.0);
    ub.segment(6 * i, 2) = Eigen::Vector2d(1.0, 1.0);
    lb.segment(6 * i + 2, 2) = Eigen::Vector2d(-0.3, -0.3);
    ub.segment(6 * i + 2, 2) = Eigen::Vector2d(0.3, 0.3);
    lb.segment(6 * i + 4, 2) = Eigen::Vector2d(-0.1, -0.1);
    ub.segment(6 * i + 4, 2) = Eigen::Vector2d(0.1, 0.1);
  }
  auto box_con = std::make_shared<BoxConstraint>(lb, ub, 1e-4, true);

  auto cstset = std::make_shared<ConstraintSet>();
  cstset->add(diff_con);
  cstset->add(goal_con);
  cstset->add(collision_con);
  cstset->add(box_con);

  // prepare quadratic cost \sum_{i \in T} u_i^2
  SMatrix P(cstset->nx_, cstset->nx_);
  for(size_t i = 0; i < T; i++) {
    P.coeffRef(6 * i + 4, 6 * i + 4) = 1.0;
    P.coeffRef(6 * i + 5, 6 * i + 5) = 1.0;
  }

  // create straight-line initial guess
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(cstset->nx_);
  auto diff_per_t = (goal - start) / (T - 1);
  for(size_t i = 0; i < T; i++) {
    x0.segment(6 * i, 2) = start + diff_per_t * i;
    x0.segment(6 * i + 2, 2) = diff_per_t / dt;
  }

  auto option = NLPSolverOption();
  option.max_iter = 200;
  auto solver = NLPSolver(cstset->nx_, P, Eigen::VectorXd::Zero(cstset->nx_), cstset, option);
  auto ret = solver.solve(x0);
  EXPECT_EQ(ret, NLPStatus::Success);

  std::ofstream ofs("/tmp/osqp_sqp_cpp_test-double_integrator.csv");
  for(size_t i = 0; i < T; i++) {
    for(size_t j = 0; j < 6; j++) {
      ofs << solver.solution_(6 * i + j) << std::endl;
    }
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
