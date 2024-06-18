#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <optional>
#include <sstream>
#include <string>

namespace osqpsqp {

using SMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor>;

struct ConstraintBase {
  ConstraintBase(size_t nx, double tol = 1e-6) : tol_(tol), nx_(nx) {}
  virtual void evaluate(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                        SMatrix &jacobian, size_t constraint_idx_head) = 0;

  virtual bool evaluate_full(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                             SMatrix &jacobian, Eigen::VectorXd &lower,
                             Eigen::VectorXd &upper,
                             size_t constraint_idx_head) = 0;
  virtual size_t get_cdim() = 0;

  bool check_jacobian(const Eigen::VectorXd &x, double eps,
                      bool verbose = false);
  bool check_jacobian(double eps, bool verbose = false, size_t n_trial = 1) {
    bool result = true;
    for (size_t i = 0; i < n_trial; i++) {
      bool partial = check_jacobian(Eigen::VectorXd::Random(nx_), eps, verbose);
      result = result && partial;
    }
    return result;
  }

  virtual ~ConstraintBase() {}
  size_t nx_;
  double tol_;
};

struct EqualityConstraintBase : public ConstraintBase {
  using ConstraintBase::ConstraintBase;
  bool evaluate_full(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                     SMatrix &jacobian, Eigen::VectorXd &lower,
                     Eigen::VectorXd &upper,
                     size_t constraint_idx_head) override;
};

struct InequalityConstraintBase : public ConstraintBase {
  using ConstraintBase::ConstraintBase;
  bool evaluate_full(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                     SMatrix &jacobian, Eigen::VectorXd &lower,
                     Eigen::VectorXd &upper,
                     size_t constraint_idx_head) override;
};

class BoxConstraint : public ConstraintBase {
public:
  BoxConstraint(const Eigen::VectorXd &lb, const Eigen::VectorXd &ub,
                double tol = 1e-6)
      : ConstraintBase(lb.size(), tol), lb_(lb), ub_(ub) {}
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
      : nx_(0), constraints_(std::vector<std::shared_ptr<ConstraintBase>>()) {}

  void add(std::shared_ptr<ConstraintBase> c) {
    if (nx_ != 0 && c->nx_ != nx_) {
      std::stringstream ss;
      ss << "ConstraintSet::add: nx_ mismatch: " << nx_ << " but " << c->nx_;
      throw std::runtime_error(ss.str());
    }
    nx_ = c->nx_;
    constraints_.push_back(c);
  }

  bool evaluate_full(const Eigen::VectorXd &x, Eigen::VectorXd &values,
                     SMatrix &jacobian, Eigen::VectorXd &lower,
                     Eigen::VectorXd &upper);

  size_t get_cdim();

  size_t nx_;
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
