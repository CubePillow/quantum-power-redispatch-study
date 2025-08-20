
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include <iostream>
#include <numpy/arrayobject.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <set>
#include <tuple>
#include <vector>

namespace py = pybind11;

#define num_t float
typedef Eigen::SparseMatrix<num_t, Eigen::ColMajor> sp_mat_col;
typedef Eigen::SparseMatrix<num_t, Eigen::RowMajor> sp_mat_row;

// * Symmetric Matrix Class

class SparseSymmetric {
  private:
    bool frozen;
    size_t N;
    sp_mat_col _m;

  public:
    SparseSymmetric(size_t N) : frozen(false), N(N), _m(N, N) { _m.setZero(); }

    size_t dim() { return N; }

    size_t idx(size_t i, size_t j) { return (i < j) ? i : j; }
    size_t jdx(size_t i, size_t j) { return (i < j) ? j : i; }

    bool set(size_t i, size_t j, float value) {
        if (frozen)
            return false;
        _m.insert(idx(i, j), jdx(i, j)) = value;
        return true;
    }

    num_t get(size_t i, size_t j) {
        size_t t_idx = idx(i, j);
        size_t t_jdx = jdx(i, j);
        if (_m.valuePtr() == NULL) {
            return 0.0;
        }
        return _m.coeff(t_idx, t_jdx);
    }

    bool add(size_t i, size_t j, num_t value) {
        if (frozen)
            return false;
        size_t t_idx = idx(i, j);
        size_t t_jdx = jdx(i, j);
        _m.coeffRef(t_idx, t_jdx) += value;
        return true;
    }

    bool set_mat(const Eigen::MatrixXf &mat) {
        if (frozen)
            return false;
        assert(N == mat.rows() && N == mat.cols());
        Eigen::MatrixXf upper_tri(mat.rows(), mat.cols());
        upper_tri.setZero();
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                upper_tri(idx(i, j), jdx(i, j)) += mat(i, j);
            }
        }
        _m = upper_tri.sparseView();
        return true;
    }

    void freeze() { _m.makeCompressed(); }

    const Eigen::SparseMatrix<num_t> &data() const { return _m; }

    Eigen::VectorXf energies(const Eigen::MatrixXf &xs) {
        if (xs.cols() != _m.rows()) {
            throw std::invalid_argument("Dimensions do not match.");
        }
        Eigen::MatrixXf inter = xs * _m;

        Eigen::VectorXf es(xs.rows());
        for (size_t i = 0; i < xs.rows(); i++) {
            es(i) = inter.row(i).dot(xs.row(i));
        }
        return es;
    }

    Eigen::MatrixXf mul_quad(const Eigen::MatrixXf &A) {
        return A.transpose() * _m * A;
    }

    Eigen::MatrixXf mul_lr(const Eigen::MatrixXf &L, const Eigen::MatrixXf &R) {
        return L * _m * R;
    }

    py::array_t<num_t> to_numpy() {
        Eigen::MatrixXf dense_mat = Eigen::MatrixXf(_m);
        return py::array_t<num_t>(
            {dense_mat.rows(), dense_mat.cols()},
            {sizeof(num_t), sizeof(num_t) * dense_mat.cols()},
            dense_mat.data());
    }

    void print_sparse_m() {
        std::cout << "Sparse Matrix: \n";
        for (int k = 0; k < _m.outerSize(); ++k) {
            for (sp_mat_col::InnerIterator it(_m, k); it; ++it) {
                if (it.value() != 0) {
                    PySys_WriteStdout("Row: %d, Col: %d, Value: %f\n",
                                      int(it.row()), int(it.col()), it.value());
                }
            }
        }
    }

    double mean() {
        double mu = 0.0;
        for (int k = 0; k < _m.outerSize(); ++k) {
            for (sp_mat_col::InnerIterator it(_m, k); it; ++it) {
                if (it.row() == it.col()) {
                    mu += it.value() / 2;
                } else {
                    mu += it.value() / 4;
                }
            }
        }
        return mu;
    }

    double variance() {
        Eigen::VectorXf row_sums(N);
        row_sums.setZero();
        Eigen::VectorXf col_sums(N);
        col_sums.setZero();
        double var = 0.0;
        for (int k = 0; k < _m.outerSize(); ++k) {
            for (sp_mat_col::InnerIterator it(_m, k); it; ++it) {
                row_sums(it.row()) += it.value();
                col_sums(it.col()) += it.value();
                if (it.row() != it.col()) {
                    var += pow(it.value(), 2);
                }
            }
        }
        var += 2 * row_sums.dot(col_sums);
        var += row_sums.dot(row_sums);
        var += col_sums.dot(col_sums);
        return var / 16;
    }

    double mean_one_hot(size_t n, size_t k) {
        assert(n * k == N);
        double mu1 = 0.0;

#pragma omp parallel for reduction(+ : mu1)
        for (size_t i = 0; i < N; ++i) {
            mu1 += _m.coeff(i, i);
        }

        double mu2 = 0.0;
#pragma omp parallel for collapse(2) reduction(+ : mu2)
        for (size_t a = 0; a < n; ++a) {
            for (size_t i = 0; i < k; ++i) {
                for (size_t b = a + 1; b < n; ++b) {
#pragma omp simd reduction(+ : mu2)
                    for (size_t j = 0; j < k; ++j) {
                        mu2 += _m.coeff(a * k + i, b * k + j);
                    }
                }
            }
        }
        return (mu1 / k) + (mu2 / (k * k));
    }

    double iterate_covar(size_t k, size_t i, size_t j, num_t val) {
        assert(k != 0);
        double res = 0.0;
        size_t a = i / k;
        // PySys_WriteStdout("%ld / %ld = %ld", i, k, a);
        size_t b = j / k;
        double mul;
        if (i == j) {
            mul = 1.0 / (double)k;
        } else if (a != b) {
            mul = 1.0 / (double)(k * k);
        } else {
            return 0;
        }
        for (int col_it = 0; col_it < _m.outerSize(); ++col_it) {
            for (sp_mat_col::InnerIterator it(_m, col_it); it; ++it) {
                double mul_pos = mul;
                size_t i2 = it.row();
                size_t j2 = it.col();
                num_t val2 = it.value();
                size_t a2 = i2 / k;
                size_t b2 = j2 / k;
                if (a2 == b2 && i2 != j2) {
                    continue;
                } else if (a2 > a && a2 > b && b2 > a && b2 > b) {
                  break;
                } else if (a2 != a && a2 != b && b2 != a && b2 != b) {
                    continue;
                }
                if (i2 == j2) { // => a2 == b2
                    if ((a2 == a) && (i2 != i) || (a2 == b) && (i2 != j)) {
                        mul_pos = 0;
                    } else if (a2 != a && a2 != b) {
                        mul_pos = mul_pos / k;
                    }
                    res += val * val2 * (mul_pos - (mul / k));
                } else {
                    if ((a2 == a) && (i2 != i) || (a2 == b) && (i2 != j)) {
                        mul_pos = 0;
                    } else if (a2 != a && a2 != b) {
                        mul_pos = mul_pos / k;
                    }
                    if ((b2 == a) && (j2 != i) || (b2 == b) && (j2 != j)) {
                        mul_pos = 0;
                    } else if (b2 != a && b2 != b) {
                        mul_pos = mul_pos / k;
                    }
                    res += val * val2 * (mul_pos - (mul / (k * k)));
                }
            }
        }
        return res;
    }

    double variance_one_hot(size_t n, size_t k) {
        assert(n * k == N);
        _m.makeCompressed();
        double var = 0.0;
    #pragma omp parallel for reduction(+:var)
        for (int col_it = 0; col_it < _m.outerSize(); ++col_it) {
          double local_var = 0.0;
            for (sp_mat_col::InnerIterator it(_m, col_it); it; ++it) {
                local_var += iterate_covar(k, it.row(), it.col(), it.value());
            }
            var += local_var;
        }
        return var;
    }
};

PYBIND11_MAKE_OPAQUE(SparseSymmetric)

SparseSymmetric *create_sparse_symmetric(size_t N) {
    return new SparseSymmetric(N);
}

SparseSymmetric *to_sparse_symmetric(const Eigen::MatrixXf &mat) {
    SparseSymmetric *Q = new SparseSymmetric(mat.cols());
    Q->set_mat(mat);
    return Q;
}

void destroy_sparse_symmetric(SparseSymmetric *Q) { delete Q; }

// * Util
template <size_t n>
void get_idxes(size_t *idxes, size_t idx, const size_t *const &bases) {
#pragma unroll
    for (size_t i = 0; i < n; ++i) {
        size_t b = bases[n - 1 - i];
        idxes[n - 1 - i] = idx % b;
        idx -= idxes[n - 1 - i];
        idx /= b;
    }
}

template <size_t n> size_t get_idx(size_t *idxes, const size_t *const &bases) {

    size_t stride = 1;
    size_t result = 0;
#pragma unroll
    for (size_t i = 0; i < n; ++i) {
        result += stride * idxes[n - 1 - i];
        stride *= bases[n - 1 - i];
    }
    return result;
}

// * QUBO
// ** Norm Element
double norm_ele(const size_t &T, const size_t &n, const size_t &idx,
                const size_t &jdx, const double &lb, const double &ub,
                const double &ele) {
    double new_ele = (idx == jdx) ? ele - (lb / (T * n)) : ele;
    return new_ele / (ub - lb);
}
// ** Switch Cost
// *** Util
double get_switch_upper_bound(const size_t &T, const size_t &n, const size_t &k,
                              double *SwC) {
    size_t bases[] = {n, k, k};
    double ub = 0;
    for (size_t j = 0; j < n; ++j) {
        double max_swc = 0;
        for (size_t i1 = 0; i1 < k; ++i1) {
            for (size_t i2 = 0; i2 < k; ++i2) {
                const size_t idx = get_idx<3>(new size_t[]{j, i1, i2}, bases);
                max_swc = std::fmax(max_swc, SwC[idx]);
            }
        }
        ub = ub + (max_swc * (T - 1));
    }
    return ub;
}

double get_qubo_switch_cost_ij(const size_t &idx, const size_t &jdx,
                               const size_t &T, const size_t &n,
                               const size_t &k, double *SwC) {

    const size_t bases[] = {T, n, k};
    const size_t N = T * n * k;
    if (idx >= N || jdx >= N)
        return 0;

    size_t idxes[3];
    get_idxes<3>(idxes, idx, bases);

    size_t jdxes[3];
    get_idxes<3>(jdxes, jdx, bases);

    const size_t swc_idx = get_idx<3>(
        new size_t[]{idxes[1], idxes[2], jdxes[2]}, new size_t[]{n, k, k});
    if (idxes[1] == jdxes[1] && jdxes[0] == idxes[0] + 1) {
        return SwC[swc_idx];
    }

    return 0;
}
// *** Main
bool update_qubo_switch_cost(SparseSymmetric &Q, const double &lam,
                             const size_t &T, py::array_t<double> SwC) {
    py::buffer_info SwC_buf = SwC.request();
    const size_t n = SwC_buf.shape[0];
    const size_t k = SwC_buf.shape[1];
    double *SwC_data = static_cast<double *>(SwC_buf.ptr);
    const size_t N = T * n * k;

    double lb = 0;
    double ub = get_switch_upper_bound(T, n, k, SwC_data);

    // #pragma omp parallel for collapse(2)
    for (size_t idx = 0; idx < N; ++idx) {
        for (size_t jdx = 0; jdx < N; ++jdx) {
            const float val =
                lam *
                norm_ele(T, n, idx, jdx, lb, ub,
                         get_qubo_switch_cost_ij(idx, jdx, T, n, k, SwC_data));
            Q.add(idx, jdx, val);
        }
    }
    return true;
}

// ** Cost
// *** Util
double *get_cost_lb_ub(double *C, const size_t &T, const size_t &n,
                       const size_t &k) {
    static double bounds[] = {0, 0};
    for (int t = 0; t < T; ++t) {
        for (int j = 0; j < n; ++j) {
            size_t idx = (t * n * k) + (j * k);
            bounds[0] += C[idx];
            bounds[1] += C[idx + k - 1];
        }
    }
    return bounds;
}

double get_qubo_cost_ij(const size_t &idx, const size_t &jdx, const size_t &N,
                        double *C) {
    return (idx == jdx && idx < N) ? C[idx] : 0;
}
// *** Main
bool update_qubo_cost(SparseSymmetric &Q, const double &lam,
                      py::array_t<double> cost) {
    py::buffer_info cost_buf = cost.request();
    const size_t &T = cost_buf.shape[0];
    const size_t &n = cost_buf.shape[1];
    const size_t &k = cost_buf.shape[2];
    double *cost_data = static_cast<double *>(cost_buf.ptr);
    const size_t N = T * n * k;

    double lb, ub;
    double *bs = get_cost_lb_ub(cost_data, T, n, k);

    // #pragma omp parallel for
    for (size_t idx = 0; idx < N; ++idx) {
        const float val =
            lam * norm_ele(T, n, idx, idx, bs[0], bs[1],
                           get_qubo_cost_ij(idx, idx, N, cost_data));
        Q.add(idx, idx, val);
    }
    return true;
}
// ** Overload
// *** Util
py::array_t<double> get_overload_norm_bounds(const py::array_t<double> &M,
                                             const py::array_t<double> &S,
                                             const py::array_t<double> &PE,
                                             double *oLB, double *oUB) {
    py::module np = py::module::import("numpy");
    py::dict kwargs;
    kwargs["axis"] = 2;
    auto np_T = np.attr("transpose");
    np.attr("transpose")(np.attr("min")(PE, 2)).cast<py::array_t<double>>();
    py::array_t<double> min_loads =
        np.attr("matmul")(S, np_T(np.attr("min")(PE, 2)))
            .cast<py::array_t<double>>();
    py::array_t<double> max_loads =
        np.attr("matmul")(S, np_T(np.attr("max")(PE, 2)))
            .cast<py::array_t<double>>();
    py::array_t<double> ret_norms =
        np.attr("zeros_like")(M).cast<py::array_t<double>>();
    double lb = 0;
    double ub = 0;
    for (size_t l = 0; l < M.shape(0); l++) {
        for (size_t t = 0; t < M.shape(1); t++) {
            double min = M.at(l, t) - max_loads.at(l, t);
            double max = M.at(l, t) - min_loads.at(l, t);
            double min_norm = min / max;
            double max_norm = 1;
            ub += (pow(min_norm, 2) / 2) - min_norm + 1;
            lb += (pow(max_norm, 2) / 2) - max_norm + 1;
            ret_norms.mutable_at(l, t) = (1 / max);
        }
    }
    *oLB = lb;
    *oUB = ub;
    return ret_norms;
}

double get_qubo_overload_unbalanced_ij(
    const size_t &idx, const size_t &jdx, const size_t &t,
    const py::array_t<double> &PE, const py::array_t<double> &nS,
    const py::array_t<double> &SnS, const py::array_t<double> &MnS,
    const double &offset, const double &m1, const double &m2, const double &lb,
    const double &ub) {
    // PySys_WriteStdout("\n((%d, %d), (%d, %d))", (int)t, (int)idx, (int)t,
    // (int)jdx);
    const ssize_t &T = PE.shape(0);
    const size_t &n = PE.shape(1);
    const size_t &k = PE.shape(2);

    const size_t bases[] = {n, k};
    if (idx >= n * k || jdx >= n * k)
        return 0;

    size_t idxes[2];

    get_idxes<2>(idxes, idx, bases);

    const size_t &j1 = idxes[0];
    const size_t &i1 = idxes[1];

    size_t jdxes[2];

    get_idxes<2>(jdxes, jdx, bases);

    const size_t &j2 = jdxes[0];
    const size_t &i2 = jdxes[1];
    const double pe1 = PE.at(t, j1, i1);
    const double pe2 = PE.at(t, j2, i2);
    const double sns = SnS.at(j1, j2);
    double res = pe1 * pe2 * sns / 2;
    if (idx == jdx) {
        const double ns = nS.at(t, j1);
        const double mns = MnS.at(t, j1);
        res += offset - (m1 - pe1 * ns) + (m2 / 2) - (pe1 * mns);
    }
    return norm_ele(T, n, idx, jdx, lb, ub, res);
}
// *** Main
bool update_qubo_overload_unbalanced(SparseSymmetric &Q, const double lam,
                                     py::array_t<double> M,
                                     py::array_t<double> S,
                                     py::array_t<double> PE) {
    py::module np = py::module::import("numpy");
    py::buffer_info PE_buf = PE.request();
    const size_t &T = PE_buf.shape[0];
    const size_t &n = PE_buf.shape[1];
    const size_t &k = PE_buf.shape[2];

    // double *cost_data = static_cast<double *>(cost_buf.ptr);
    py::buffer_info M_buf = M.request();
    const size_t &L = M.shape(0);

    const size_t nk = n * k;

    double lb = 0;
    double ub = 0;
    py::array_t<double> norms = get_overload_norm_bounds(M, S, PE, &lb, &ub);

    double offset = ((double)(L * T)) / ((double)(T * n));
    py::array_t<double> normS =
        np.attr("matmul")(np.attr("transpose")(norms), S)
            .cast<py::array_t<double>>();
    py::array_t<double> Mnorms2 =
        np.attr("zeros_like")(M).cast<py::array_t<double>>();
    double m1 = 0;
    double m2 = 0;
    for (size_t l = 0; l < L; ++l) {
        for (size_t t = 0; t < T; ++t) {
            Mnorms2.mutable_at(l, t) = M.at(l, t) * pow(norms.at(l, t), 2);
            m1 += (M.at(l, t) * norms.at(l, t)) / (n * T);
            m2 += pow(M.at(l, t) * norms.at(l, t), 2) / (n * T);
        }
    }
    // PySys_WriteStdout("\nm1: %f, m2: %f", m1, m2);
    // PySys_WriteStdout("\nlb: %f, ub: %f\n", lb, ub);
    py::array_t<double> MnS =
        np.attr("matmul")(np.attr("transpose")(Mnorms2), S)
            .cast<py::array_t<double>>();

    // #pragma omp parallel for collapse(2)
    for (size_t t = 0; t < T; ++t) {
        py::array_t<double> norm_t =
            np.attr("zeros")(L).cast<py::array_t<double>>();
        for (size_t l = 0; l < L; l++) {
            norm_t.mutable_at(l) = pow(norms.at(l, t), 2);
            assert(norm_t.at(l) == pow(norms.at(l, t), 2));
        }
        // py::array_t<double> norm_t_diag =
        //     np.attr("diag")(norm_t).cast<py::array_t<double>>();
        // PySys_WriteStdout("vec len: %d\n", int(norm_t.shape(0)));
        // PySys_WriteStdout("%d,%d", int(norm_t_diag.shape(0)),
        // int(norm_t_diag.shape(1))); PySys_WriteStdout("hi1");
        // np.attr("matmul")(np.attr("diag")(norm_t), S);
        // PySys_WriteStdout("hi2");
        py::array_t<double> SnS =
            np.attr("matmul")(np.attr("transpose")(S),
                              np.attr("matmul")(np.attr("diag")(norm_t), S));
        for (size_t nk_idx = 0; nk_idx < nk; ++nk_idx) {
            size_t idx = nk_idx + (t * nk);
            for (size_t nk_jdx = 0; nk_jdx < nk; ++nk_jdx) {
                size_t jdx = nk_jdx + (t * nk);
                const double val = lam * get_qubo_overload_unbalanced_ij(
                                             nk_idx, nk_jdx, t, PE, normS, SnS,
                                             MnS, offset, m1, m2, lb, ub);
                Q.add(idx, jdx, val);
            }
        }
    }
    return true;
}

// ** Power
// *** Util
double get_power_ub(const py::array_t<double> &PE,
                    const py::array_t<double> &Tau) {
    double pet1 = 0.0;
    double pet2 = 0.0;
    for (size_t t = 0; t < PE.shape(0); t++) {
        double pe1 = 0.0;
        double pe2 = 0.0;
        for (size_t j = 0; j < PE.shape(1); j++) {
            pe1 += PE.at(t, j, 0);
            pe2 += PE.at(t, j, PE.shape(2) - 1);
        }
        pet1 += pow(pe1 - Tau.at(t), 2);
        pet2 += pow(pe2 - Tau.at(t), 2);
    }
    return std::max(pet1, pet2);
}

double get_qubo_power_ij(const size_t &idx, const size_t &jdx, const size_t &t,
                         const double &tau, const py::array_t<double> &PE,
                         const py::array_t<double> &Tau, const double &lb,
                         const double &ub) {
    // PySys_WriteStdout("\n((%d, %d), (%d, %d))", (int)t, (int)idx, (int)t,
    // (int)jdx);
    const ssize_t &T = PE.shape(0);
    const size_t &n = PE.shape(1);
    const size_t &k = PE.shape(2);

    const size_t bases[] = {n, k};
    if (idx >= n * k || jdx >= n * k)
        return 0;

    size_t idxes[2];

    get_idxes<2>(idxes, idx, bases);

    const size_t &j1 = idxes[0];
    const size_t &i1 = idxes[1];

    size_t jdxes[2];

    get_idxes<2>(jdxes, jdx, bases);

    const size_t &j2 = jdxes[0];
    const size_t &i2 = jdxes[1];
    const double pe1 = PE.at(t, j1, i1);
    const double pe2 = PE.at(t, j2, i2);
    double res = pe1 * pe2;
    if (idx == jdx) {
        res += tau - (2 * Tau.at(t) * pe1);
    }
    return norm_ele(T, n, idx, jdx, lb, ub, res);
}

// *** Main
bool update_qubo_power(SparseSymmetric &Q, const double &lam,
                       py::array_t<double> PE, py::array_t<double> Tau) {
    py::module np = py::module::import("numpy");
    py::module linalg = py::module::import("numpy.linalg");
    py::buffer_info PE_buf = PE.request();
    const size_t &T = PE_buf.shape[0];
    const size_t &n = PE_buf.shape[1];
    const size_t &k = PE_buf.shape[2];
    const size_t nk = n * k;
    const size_t N = T * n * k;

    double lb = 0;
    double ub = get_power_ub(PE, Tau);

    double tau = pow(linalg.attr("norm")(Tau).cast<double>(), 2) / (n * T);
    // #pragma omp parallel for
    for (size_t t = 0; t < T; t++) {
        for (size_t nk_idx = 0; nk_idx < nk; ++nk_idx) {
            size_t idx = nk_idx + (t * nk);
            for (size_t nk_jdx = 0; nk_jdx < nk; ++nk_jdx) {
                size_t jdx = nk_jdx + (t * nk);
                const float val = lam * get_qubo_power_ij(nk_idx, nk_jdx, t,
                                                          tau, PE, Tau, lb, ub);
                Q.add(idx, jdx, val);
            }
        }
    }
    return true;
}

// * Module
PYBIND11_MODULE(qe_sparse, m) {
    py::class_<SparseSymmetric>(m, "SparseSymmetric", py::module_local())
        .def("to_numpy", &SparseSymmetric::to_numpy)
        .def("dim", &SparseSymmetric::dim)
        .def("freeze", &SparseSymmetric::freeze)
        .def("energies", &SparseSymmetric::energies)
        .def("mul_quad", &SparseSymmetric::mul_quad)
        .def("mul_lr", &SparseSymmetric::mul_lr)
        .def("set_mat", &SparseSymmetric::set_mat)
        .def("mean", &SparseSymmetric::mean)
        .def("variance", &SparseSymmetric::variance)
        .def("variance_one_hot", &SparseSymmetric::variance_one_hot)
        .def("mean_one_hot", &SparseSymmetric::mean_one_hot);
    m.def("create_sparse_symmetric", &create_sparse_symmetric,
          "Create new SparseSymmetric Matrix of size N");
    m.def("to_sparse_symmetric", &to_sparse_symmetric,
          "Convert numpy mat into SparseSymmetric Matrix");
    m.def("update_qubo_switch_cost", &update_qubo_switch_cost,
          "add switch cost to qubo Q");
    m.def("update_qubo_cost", &update_qubo_cost, "add cost to qubo Q");
    m.def("update_qubo_overload_unbalanced", &update_qubo_overload_unbalanced,
          "add overload to qubo Q");
    m.def("update_qubo_power", &update_qubo_power, "add power to qubo Q");
}
