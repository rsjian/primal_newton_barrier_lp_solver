#include "Eigen/Eigen"

#include <iostream>
using namespace std;

typedef double NT;
typedef Eigen::Matrix<NT, Eigen::Dynamic, 1> VT;
typedef Eigen::Matrix <NT, Eigen::Dynamic, Eigen::Dynamic> MT;

std::tuple<MT, VT, VT> convert_standard_form(const MT& A, const VT& b, const VT& c)
{
    MT A_(A.rows(), A.cols() + b.rows());
    VT b_ = b;
    VT c_(c.rows() + A.rows());
    A_ << A, MT::Identity(b.rows(), b.rows()); // add slack variables
    c_ << c, VT::Zero(A.rows()); // zero cost for each slack variable
    // if b[i] < 0, multiply A[i] and b[i] by -1
    A_.array().colwise() *= 1 - 2*(b_.array() < 0).cast<double>();
    b_ = b_.array().abs();
    return {A_, b_, c_};
}

std::tuple<MT, VT, VT> convert_feasibility_lp(const MT& A, const VT& b, const VT& c)
{
    MT A_(A.rows(), A.cols() + b.rows());
    VT b_ = b;
    VT c_(c.rows() + b.rows());
    A_ << A, MT::Identity(b.rows(), b.rows());
    c_ << VT::Zero(c.rows()), VT::Ones(b.rows());
    return {A_, b_, c_};
}

NT line_search(const VT& x, const VT& dx)
{
    NT alpha = 1.;
    while (!((x + alpha * dx).array() > 0).all()) {
        alpha *= 0.1;
    }
    return alpha;
}

// min cx
// x >= 0
// Ax = b
VT solve_lp_standard_inner(const MT& A, const VT& b, const VT& c, NT gamma, const VT& x0)
{
    VT x = x0;
    while (true) {
        MT X = x.asDiagonal();
        MT X2 = X*X;
        VT lambda = (A*X2*A.transpose()).ldlt().solve(A*X2*c - gamma*A*x);
        VT dk = x + 1/gamma*X2*(A.transpose()*lambda - c);
        NT alpha = line_search(x, dk);
        x += alpha * dk;
        //cout << "x: " << x << endl;
        if (dk.squaredNorm() <= 1e-7)
            break;
    }
    return x;
}

VT solve_lp_standard(const MT& A, const VT& b, const VT& c, const VT& x0)
{
    NT gamma = 1.0;
    VT x = x0;
    for (int i = 0; i < 12; ++i) {
        x = solve_lp_standard_inner(A, b, c, gamma, x);
        gamma *= 0.1;
    }
    cout << "final cost: " << c.dot(x) << endl;
    return x;
}

// min cx
// x >= 0
// Ax <= b
VT solve_lp_nonstandard(const MT& A, const VT& b, const VT& c)
{
    cout << "solve_lp" << endl;
    MT A_, A2;
    VT b_, b2;
    VT c_, c2;
    std::tie(A_, b_, c_) = convert_standard_form(A, b, c);
    std::tie(A2, b2, c2) = convert_feasibility_lp(A_, b_, c_);
    VT x2(c_.rows() + b_.rows());
    x2 << 1e-8 * VT::Ones(c_.rows()), VT::Zero(b_.rows());
    for (int i = 0; i < b_.rows(); ++i)
        x2[c_.rows() + i] = b_[i] - A2.row(i).dot(x2);
    cout << A_ << "\n\n";
    cout << b_ << "\n\n";
    cout << c_ << "\n\n";
    cout << A2 << "\n\n";
    cout << b2 << "\n\n";
    cout << c2 << "\n\n";
    cout << x2 << "\n\n";

    VT x_feasible_opt = solve_lp_standard(A2, b2, c2, x2);
    cout << x_feasible_opt << "\n\n" << endl;
    double feasibility_opt = c2.dot(x_feasible_opt);
    cout << "feasiblity opt: " << feasibility_opt << endl;
    if (feasibility_opt >= 1e-6) {
        cerr << "Unable to find a feasible starting point" << endl;
        return {};
    }
    VT x_ = x_feasible_opt(Eigen::seq(0, c_.rows() - 1));
    VT x_opt = solve_lp_standard(A_, b_, c_, x_);
    return x_opt;
}

MT A {
    {1, 3},
    {2, 1},
    {-3, -3}
};
VT b {{2, 2, -1}};
VT c {{2, -2}};

int main()
{
    VT x_opt = solve_lp_nonstandard(A, b, c);
    cout << x_opt << endl;
    cout << endl;

    VT x_opt2 = x_opt(Eigen::seq(0, c.rows() - 1));
    cout << "found solution: " << endl;
    cout << x_opt2 << endl;
    return 0;
}
