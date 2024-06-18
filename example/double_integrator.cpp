#include "osqpsqp.hpp"

using namespace osqpsqp;


void add_identity_to_sparse_jacobian(
    SMatrix &jac,
    size_t start_i, size_t start_j, size_t size, double identity_coef) {
  for(size_t i = 0; i < size; i++) {
    jac.coeffRef(start_i + i, start_j + i) += identity_coef;
  }
}

class DifferentialConstraint : public EqualityConstraintBase {
  public:
  DifferentialConstraint(size_t T, double dt) : T_(T), dt_(dt) {}

  size_t get_cdim() override {
    return 4 * (T_ - 1);
  }

  void evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                SMatrix &jacobian, size_t constraint_idx_head) override {
    // x is a vector of size 6*T . 6 consists of (position, velocity, acceleration) for each time step
    // X -> position, U -> velocity, U -> acceleration
    auto S = Eigen::Map<const Eigen::MatrixXd>(x.data(), T_, 6);
    auto X = S.block(0, 0, T_, 2);
    auto V = S.block(0, 2, T_, 2);
    auto A = S.block(0, 4, T_, 2);
    auto X0 = X.block(0, 0, T_-1, 2);
    auto V0 = V.block(0, 0, T_-1, 2);
    auto U0 = A.block(0, 0, T_-1, 2);
    auto X1 = X.block(1, 0, T_-1, 2);
    auto V1 = V.block(1, 0, T_-1, 2);
    auto U1 = A.block(1, 0, T_-1, 2);

    auto X_t_est = X0 + V0 * dt_ + 0.5 * U0 * dt_ * dt_;
    auto V_t_est = V0 + U0 * dt_;

    auto X_residual = X1 - X_t_est;
    auto V_residual = V1 - V_t_est;
    auto target = Eigen::Map<Eigen::VectorXd>(values.data(), 4 * (T_ - 1));
    target << X_residual, V_residual;

    // compute jacobian matrix
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

class GoalConstraint : public EqualityConstraintBase {
  public:
  GoalConstraint(size_t T, const Eigen::Vector2d &goal)
      : T_(T), goal_(goal) {}

  size_t get_cdim() override {
    return 4;
  }

  void evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                SMatrix &jacobian, size_t constraint_idx_head) override {
    auto S = Eigen::Map<const Eigen::MatrixXd>(x.data(), T_, 6);
    auto X = S.block(0, 0, T_, 2);
    auto V = S.block(0, 2, T_, 2);
    auto X_goal = goal_.head(2);
    auto V_goal = goal_.tail(2);
    auto X_t = X.row(T_ - 1);
    auto V_t = V.row(T_ - 1);
    auto X_residual = X_t - X_goal;
    auto V_residual = V_t - V_goal;
    auto target = Eigen::Map<Eigen::VectorXd>(values.data(), 2);
    target << X_residual, V_residual;

    // compute jacobian matrix
    size_t i_head = constraint_idx_head;
    size_t j_head = 6 * (T_ - 1);
    add_identity_to_sparse_jacobian(jacobian, i_head, j_head, 2, 1.0);
    add_identity_to_sparse_jacobian(jacobian, i_head + 2, j_head + 2, 2, 1.0);
  }

  private:
  size_t T_;
  Eigen::VectorXd goal_;
};

int main() {

}
