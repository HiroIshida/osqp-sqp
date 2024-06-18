#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <optional>

namespace osqpsqp {

using SMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor>;

struct ConstraintBase {
  virtual void evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                        SMatrix &jacobian, size_t constraint_idx_head) = 0;

  virtual bool evaluate_full(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                             SMatrix &jacobian, Eigen::VectorXd &lower,
                             Eigen::VectorXd &upper,
                             size_t constraint_idx_head) = 0;
  virtual size_t get_cdim() = 0;
  virtual ~ConstraintBase() {}
  double tol = 1e-6;
};

struct EqualityConstraintBase : public ConstraintBase {
  bool evaluate_full(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                     SMatrix &jacobian, Eigen::VectorXd &lower,
                     Eigen::VectorXd &upper,
                     size_t constraint_idx_head) override;
};

struct InequalityConstraintBase : public ConstraintBase {
  bool evaluate_full(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                     SMatrix &jacobian, Eigen::VectorXd &lower,
                     Eigen::VectorXd &upper,
                     size_t constraint_idx_head) override;
};

class BoxConstraint : public ConstraintBase {
public:
  BoxConstraint(const Eigen::VectorXd &lb, const Eigen::VectorXd &ub)
      : lb_(lb), ub_(ub) {}
  void evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                SMatrix &jacobian, size_t constraint_idx_head) override;
  bool evaluate_full(const Eigen::VectorXd &x, Eigen::VectorXd &values,
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
      : constraints_(std::vector<std::shared_ptr<ConstraintBase>>()) {}

  void add(std::shared_ptr<ConstraintBase> c) { constraints_.push_back(c); }

  bool evaluate_full(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                     SMatrix &jacobian, Eigen::VectorXd &lower,
                     Eigen::VectorXd &upper);

  size_t get_cdim();

  std::vector<std::shared_ptr<ConstraintBase>> constraints_;
};

struct NLPSolverOption {
  size_t max_iter = 20;
  std::optional<double> ftol = 1e-3;
  bool osqp_verbose = false;
  bool osqp_force_deterministic = false;
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
