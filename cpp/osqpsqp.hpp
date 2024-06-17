#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <optional>

namespace osqpsqp {

using SMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor>;

struct ConstraintInterface {
  virtual void evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                        SMatrix &jacobian, size_t constraint_idx_head) = 0;

  virtual void evaluate_full(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                             SMatrix &jacobian, Eigen::VectorXd &lower,
                             Eigen::VectorXd &upper,
                             size_t constraint_idx_head) = 0;
  virtual size_t get_cdim() = 0;
  virtual ~ConstraintInterface() {}
};

struct EqualityConstraintInterface : public ConstraintInterface {
  void evaluate_full(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                     SMatrix &jacobian, Eigen::VectorXd &lower,
                     Eigen::VectorXd &upper,
                     size_t constraint_idx_head) override;
};

struct InequalityConstraintInterface : public ConstraintInterface {
  void evaluate_full(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                     SMatrix &jacobian, Eigen::VectorXd &lower,
                     Eigen::VectorXd &upper,
                     size_t constraint_idx_head) override;
};

class BoxConstraint : public ConstraintInterface {
public:
  BoxConstraint(const Eigen::VectorXd &lb, const Eigen::VectorXd &ub)
      : lb_(lb), ub_(ub) {}
  void evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                SMatrix &jacobian, size_t constraint_idx_head);
  void evaluate_full(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                     SMatrix &jacobian, Eigen::VectorXd &lower,
                     Eigen::VectorXd &upper,
                     size_t constraint_idx_head) override;
  inline size_t get_cdim() override { return lb_.size(); }
  Eigen::VectorXd lb_;
  Eigen::VectorXd ub_;
};

class ConstraintSet {
public:
  ConstraintSet()
      : constraints_(std::vector<std::shared_ptr<ConstraintInterface>>()) {}

  void add(std::shared_ptr<ConstraintInterface> c) {
    constraints_.push_back(c);
  }

  void evaluate_full(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                     SMatrix &jacobian, Eigen::VectorXd &lower,
                     Eigen::VectorXd &upper);

  size_t get_cdim();

  std::vector<std::shared_ptr<ConstraintInterface>> constraints_;
};

struct NLPSolverOption {
  size_t max_iter = 20;
  double ceq_tol = 1e-4;
  double cineq_tol = 1e-4;
  std::optional<double> ftol = 1e-3;
  // if ftol=std::nullopt, then return immediately when a feasible solution is
  // found
};

class NLPSolver {
public:
  NLPSolver(size_t nx, SMatrix P, Eigen::VectorXd q,
            std::shared_ptr<ConstraintSet> cstset,
            const NLPSolverOption &option = NLPSolverOption());
  void solve(const Eigen::VectorXd &x0);

  SMatrix P_;
  Eigen::VectorXd q_;
  std::shared_ptr<ConstraintSet> cstset_;
  Eigen::VectorXd cstset_lower_;
  Eigen::VectorXd cstset_upper_;
  Eigen::VectorXd cstset_values_;
  SMatrix cstset_jacobian_;
  NLPSolverOption option_;
};

} // namespace osqpsqp
