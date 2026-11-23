
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

// * Symmetric Matrix Class

class DenseSymmetric {
  private:
    size_t N;
    Eigen::MatrixX<num_t> _m;

  public:
    DenseSymmetric(size_t N) : N(N), _m(N, N) { _m.setZero(); }

    size_t dim() { return N; }

    size_t idx(size_t i, size_t j) { return (i < j) ? i : j; }
    size_t jdx(size_t i, size_t j) { return (i < j) ? j : i; }

    bool set(size_t i, size_t j, float value) {
        _m(idx(i, j), jdx(i, j)) = value;
        return true;
    }

    num_t get(size_t i, size_t j) {
        size_t t_idx = idx(i, j);
        size_t t_jdx = jdx(i, j);
        return _m(t_idx, t_jdx);
    }

    bool add(size_t i, size_t j, num_t value) {
        size_t t_idx = idx(i, j);
        size_t t_jdx = jdx(i, j);
        _m.coeffRef(t_idx, t_jdx) += value;
        return true;
    }

    bool set_mat(const Eigen::MatrixXf &mat) {
        assert(N == mat.rows() && N == mat.cols());
        _m.setZero();
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                _m(idx(i, j), jdx(i, j)) += mat(i, j);
            }
        }
        return true;
    }

    const Eigen::MatrixX<num_t> &data() const { return _m; }

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
        return py::array_t<num_t>({_m.rows(), _m.cols()},
                                  {sizeof(num_t), sizeof(num_t) * _m.cols()},
                                  _m.data());
    }

    double mean() {
        double mu = 0.0;
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = i; j < N; ++j) {
                if (i == j) {
                    mu += _m(i, j) / 2;
                } else {
                    mu += _m(i, j) / 4;
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
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = i; j < N; ++j) {
                row_sums(i) += _m(i, j);
                col_sums(j) += _m(i, j);
                if (i != j) {
                    var += pow(_m(i, j), 2);
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
                        mu2 += _m(a * k + i, b * k + j);
                    }
                }
            }
        }
        return (mu1 / k) + (mu2 / (k * k));
    }

    double iterate_covar(size_t k, size_t i1, size_t j1, size_t a2, size_t b2) {
        assert(k != 0);
        assert(a2 != b2);
        double val = _m(i1, j1);
        double res = 0.0;
        size_t a1 = i1 / k;
        size_t b1 = j1 / k;
        double mul;
        if (i1 == j1) {
          mul = 1 / (double)k;
        } else if (a1 != b1) {
          mul = 1 / (double)(k * k);
        } else {
            return 0;
        }
        for (size_t oj = 0; oj < k; ++oj) {
            size_t j2 = b2 * k + oj;
            double mul_pos = mul;
            if ((b2 == a1 && j2 != i1) || (b2 == b1 && j2 != j1)) {
                mul_pos = 0;
            } else if (b2 != a1 && b2 != b1) {
              mul_pos = mul_pos / (double)k;
            }
            for (size_t oi = 0; oi < k; ++oi) {
                size_t i2 = a2 * k + oi;
                num_t val2 = _m(i2, j2);
                double cur_mul_pos = mul_pos;
                if ((a2 == a1 && i2 != i1) || (a2 == b1 && i2 != j1)) {
                    cur_mul_pos = 0;
                } else if (a2 != a1 && a2 != b1) {
                  cur_mul_pos = cur_mul_pos / (double)k;
                }
                res += val * val2 * (cur_mul_pos - (mul / (double)(k * k)));
            }
        }
        return res;
    }

    double variance_one_hot(size_t n, size_t k) {
        assert(n * k == N);
        double var = 0.0;
#pragma omp parallel for reduction(+ : var) schedule(dynamic)
        for (size_t i1 = 0; i1 < n * k; ++i1) {
            double local_var = 0.0;
            size_t a1 = i1 / k;
            for (size_t o=0; o < k; ++o) {
                size_t i2 = a1 * k + o;
                local_var +=
                    _m(i1, i1) * _m(i2, i2) *
                    (i2 == i1 ? (1.0 / (double)k) - (1.0 / pow((double)k, 2))
                              : -1.0 / pow((double)k, 2));
            }
            for (size_t b = 0; b < n; ++b) {
                if (a1 == b) {
                    continue;
                }
                local_var += iterate_covar(k, i1, i1, a1, b) +
                             iterate_covar(k, i1, i1, b, a1);
            }
            for (size_t j1 = 0; j1 < n * k; ++j1) {
                size_t b1 = j1 / k;
                if (a1 == b1) {
                    continue;
                }
                local_var += iterate_covar(k, i1, j1, a1, b1) +
                             iterate_covar(k, i1, j1, b1, a1);
                for (size_t o = 0; o < k; ++o) {
                    size_t i2 = a1 * k + o;
                    size_t j2 = b1 * k + o;
                    local_var += _m(i1, j1) * _m(i2, i2) *
                                 (i1 == i2 ? (1 / pow((double)k, 2)) -
                                                 (1 / pow((double)k, 3))
                                           : -1 / pow((double)k, 3));
                    local_var += _m(i1, j1) * _m(j2, j2) *
                                 (j1 == j2 ? (1 / pow((double)k, 2)) -
                                                 (1 / pow((double)k, 3))
                                           : -1 / pow((double)k, 3));
                }
                for (size_t a2 = 0; a2 < n; ++a2) {
                    if (a2 == a1 || a2 == b1) {
                        continue;
                    }
                    local_var += iterate_covar(k, i1, j1, a1, a2);
                    local_var += iterate_covar(k, i1, j1, a2, a1);
                    local_var += iterate_covar(k, i1, j1, b1, a2);
                    local_var += iterate_covar(k, i1, j1, a2, b1);
                }
            }
            var += local_var;
        }
        return var;
    }
};

PYBIND11_MAKE_OPAQUE(DenseSymmetric)

DenseSymmetric *create_dense_symmetric(size_t N) {
    return new DenseSymmetric(N);
}

DenseSymmetric *to_dense_symmetric(const Eigen::MatrixXf &mat) {
    DenseSymmetric *Q = new DenseSymmetric(mat.cols());
    Q->set_mat(mat);
    return Q;
}

void destroy_dense_symmetric(DenseSymmetric *Q) { delete Q; }

// * Module
PYBIND11_MODULE(qe_builder, m) {
    py::class_<DenseSymmetric>(m, "DenseSymmetric", py::module_local())
        .def("to_numpy", &DenseSymmetric::to_numpy)
        .def("dim", &DenseSymmetric::dim)
        .def("energies", &DenseSymmetric::energies)
        .def("mul_quad", &DenseSymmetric::mul_quad)
        .def("mul_lr", &DenseSymmetric::mul_lr)
        .def("set_mat", &DenseSymmetric::set_mat)
        .def("mean", &DenseSymmetric::mean)
        .def("variance", &DenseSymmetric::variance)
        .def("variance_one_hot", &DenseSymmetric::variance_one_hot)
        .def("mean_one_hot", &DenseSymmetric::mean_one_hot);
    m.def("create_dense_symmetric", &create_dense_symmetric,
          "Create new DenseSymmetric Matrix of size N");
    m.def("to_dense_symmetric", &to_dense_symmetric,
          "Convert numpy mat into DenseSymmetric Matrix");
}
