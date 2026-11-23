#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <iostream>
#include <tuple>

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/random/bitgen.h>
#include <numpy/random/distributions.h>

#define _USE_MATH_DEFINES
#include <cmath>

#include <omp.h>

#include "bfloat16.hpp"
#define numeric_t float
#define npy_t NPY_FLOAT

#include <utility>

#include <fenv.h>

bitgen_t *B = NULL;


inline PyObject *getNamed(const char *ckey, PyObject *args, PyObject *kwargs,
                          const size_t idx = -1, PyObject *def = 0) {
    const size_t len = PyTuple_GET_SIZE(args);
    size_t klen = 0;
    if (kwargs)
        klen = PyDict_Size(kwargs);

    PyObject *res = NULL;

    if (idx < len) {
        res = PyTuple_GET_ITEM(args, idx);
        Py_INCREF(res);
        return res;
    }

    if (klen) {
        res = PyDict_GetItemString(kwargs, ckey);
    }

    if (!res && def) {
        return def;
    }

    Py_XINCREF(res);
    return res;
}

// * Matrix Class
template <class T> class Symmetric {
  private:
    size_t N = 0;
    size_t L = 0;
    T *data;

    inline size_t idx(size_t i, size_t j) const {
        if (j < i) {
            const size_t t = j;
            j = i;
            i = t;
        }
        return L - ((N - i) * ((N - i) + 1) / 2) + (j - i);
    }

  public:
    Symmetric(const size_t &_N) {
        N = _N;
        L = N * (N + 1) / 2;

        data = new T[L];
        clear();
    }

    ~Symmetric() { delete[] data; }

    size_t getN() const { return N; }

    size_t getL() const { return L; }

    void clear() { memset(data, 0, L * sizeof(T)); }

    float get(size_t i, size_t j) const { return data[idx(i, j)]; }

    void set(size_t i, size_t j, T val) { data[idx(i, j)] = val; }

    void add(size_t i, size_t j, T val) { data[idx(i, j)] += val; }

    T* get_data() {return data; }

    void mul(size_t i, size_t j, T val) { data[idx(i, j)] *= val; }
};

PyObject *py_getN(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *s_capsule = getNamed("matrix", args, kwargs, 0);

    if (!PyCapsule_IsValid(s_capsule, "sbf16")) {
        PyErr_SetString(PyExc_TypeError, "data invalid");
        return 0;
    }

    Symmetric<bfloat16> *S =
        (Symmetric<bfloat16> *)PyCapsule_GetPointer(s_capsule, "sbf16");

    PyObject *result = PyLong_FromUnsignedLong(S->getN());
    return result;
}

PyObject *py_getL(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *s_capsule = getNamed("matrix", args, kwargs, 0);

    if (!PyCapsule_IsValid(s_capsule, "sbf16")) {
        PyErr_SetString(PyExc_TypeError, "data invalid");
        return 0;
    }

    Symmetric<bfloat16> *S =
        (Symmetric<bfloat16> *)PyCapsule_GetPointer(s_capsule, "sbf16");

    PyObject *result = PyLong_FromUnsignedLong(S->getL());
    return result;
}

PyObject *py_clear(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *s_capsule = getNamed("matrix", args, kwargs, 0);

    if (!PyCapsule_IsValid(s_capsule, "sbf16")) {
        PyErr_SetString(PyExc_TypeError, "data invalid");
        return 0;
    }

    Symmetric<bfloat16> *S =
        (Symmetric<bfloat16> *)PyCapsule_GetPointer(s_capsule, "sbf16");
    S->clear();

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *py_get(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *s_capsule = getNamed("matrix", args, kwargs, 0);

    if (!PyCapsule_IsValid(s_capsule, "sbf16")) {
        PyErr_SetString(PyExc_TypeError, "data invalid");
        return 0;
    }

    const size_t i =
        PyLong_AsUnsignedLong(getNamed("i", args, kwargs, 1, Py_None));
    const size_t j =
        PyLong_AsUnsignedLong(getNamed("j", args, kwargs, 2, Py_None));

    Symmetric<bfloat16> *S =
        (Symmetric<bfloat16> *)PyCapsule_GetPointer(s_capsule, "sbf16");

    PyObject *result = PyFloat_FromDouble((double)S->get(i, j));
    return result;
}

PyObject *py_set(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *s_capsule = getNamed("matrix", args, kwargs, 0);

    if (!PyCapsule_IsValid(s_capsule, "sbf16")) {
        PyErr_SetString(PyExc_TypeError, "data invalid");
        return 0;
    }

    const size_t i =
        PyLong_AsUnsignedLong(getNamed("i", args, kwargs, 1, Py_None));
    const size_t j =
        PyLong_AsUnsignedLong(getNamed("j", args, kwargs, 2, Py_None));

    const float val =
        (float)PyFloat_AsDouble(getNamed("val", args, kwargs, 3, Py_None));

    Symmetric<bfloat16> *S =
        (Symmetric<bfloat16> *)PyCapsule_GetPointer(s_capsule, "sbf16");

    S->set(i, j, val);

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *py_add(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *s_capsule = getNamed("matrix", args, kwargs, 0);

    if (!PyCapsule_IsValid(s_capsule, "sbf16")) {
        PyErr_SetString(PyExc_TypeError, "data invalid");
        return 0;
    }

    const size_t i =
        PyLong_AsUnsignedLong(getNamed("i", args, kwargs, 1, Py_None));
    const size_t j =
        PyLong_AsUnsignedLong(getNamed("j", args, kwargs, 2, Py_None));

    const float val =
        (float)PyFloat_AsDouble(getNamed("val", args, kwargs, 3, Py_None));

    Symmetric<bfloat16> *S =
        (Symmetric<bfloat16> *)PyCapsule_GetPointer(s_capsule, "sbf16");

    S->add(i, j, val);

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *py_mul(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *s_capsule = getNamed("matrix", args, kwargs, 0);

    if (!PyCapsule_IsValid(s_capsule, "sbf16")) {
        PyErr_SetString(PyExc_TypeError, "data invalid");
        return 0;
    }

    const size_t i =
        PyLong_AsUnsignedLong(getNamed("i", args, kwargs, 1, Py_None));
    const size_t j =
        PyLong_AsUnsignedLong(getNamed("j", args, kwargs, 2, Py_None));

    const float val =
        (float)PyFloat_AsDouble(getNamed("val", args, kwargs, 3, Py_None));

    Symmetric<bfloat16> *S =
        (Symmetric<bfloat16> *)PyCapsule_GetPointer(s_capsule, "sbf16");

    S->mul(i, j, val);

    Py_INCREF(Py_None);
    return Py_None;
}

void deconstruct_symmetric_bf16(PyObject *sbf16_capsule) {
    Symmetric<bfloat16> *S =
        (Symmetric<bfloat16> *)PyCapsule_GetPointer(sbf16_capsule, "sbf16");
    delete S;
}

PyObject *create_symmetric_bf16(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
    const size_t N =
        PyLong_AsUnsignedLong(getNamed("N", args, kwargs, 0, Py_None));
    Symmetric<bfloat16> *S = new Symmetric<bfloat16>(N);
    PyObject *res = PyCapsule_New(S, "sbf16", deconstruct_symmetric_bf16);
    return res;
}

PyObject *to_numpy(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *s_capsule = getNamed("matrix", args, kwargs, 0);

    if (!PyCapsule_IsValid(s_capsule, "sbf16")) {
        PyErr_SetString(PyExc_TypeError, "data invalid");
        return 0;
    }

    Symmetric<bfloat16> *S =
        (Symmetric<bfloat16> *)PyCapsule_GetPointer(s_capsule, "sbf16");

    const size_t N = S->getN();

    const npy_intp _t[2] = {(long)N, (long)N};
    PyArrayObject *result =
        (PyArrayObject *)PyArray_SimpleNew(2, _t, NPY_FLOAT32);

    float *result_raw = (float *)PyArray_DATA(result);

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            const float val = S->get(i, j);
            result_raw[i * N + j] = val;
        }
    }

    return (PyObject *)result;
}

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
// * Matrix prod func

// * QUBO
// ** Norm Element
double norm_ele(const size_t &T, const size_t &n, const size_t &idx,
                const size_t &jdx, const double &lb, const double &ub,
                const double &ele) {
    double new_ele = (idx == jdx) ? ele - (lb / (T * n)) : ele;
    return new_ele / (ub - lb);
}
// ** Switch Cost

double get_switch_upper_bound(PyArrayObject *SwC, const size_t &T) {
    npy_intp *shape = PyArray_DIMS(SwC);
    const size_t &n = shape[0];
    const size_t &k = shape[1];
    double ub = 0;
    for (size_t j = 0; j < n; ++j) {
        double max_swc = 0;
        for (size_t i1 = 0; i1 < k; ++i1) {
            for (size_t i2 = 0; i2 < k; ++i2) {
                double cur = *((double *)PyArray_GETPTR3(SwC, j, i1, i2));
                max_swc = std::fmax(max_swc, cur);
            }
        }
        ub = ub + (max_swc * (T - 1));
    }
    return ub;
}

double get_qubo_switch_cost_ij(const size_t &idx, const size_t &jdx,
                               const size_t &T, const size_t &n,
                               const size_t &k, PyArrayObject *SwC) {

    const size_t bases[] = {T, n, k};
    const size_t N = T * n * k;
    if (idx >= N || jdx >= N)
        return 0;

    size_t idxes[3];

    get_idxes<3>(idxes, idx, bases);

    const size_t &t1 = idxes[0];
    const size_t &j1 = idxes[1];
    const size_t &i1 = idxes[2];

    size_t jdxes[3];

    get_idxes<3>(jdxes, jdx, bases);

    const size_t &t2 = jdxes[0];
    const size_t &j2 = jdxes[1];
    const size_t &i2 = jdxes[2];

    if (j1 == j2 && t2 == t1 + 1) {
        return *((double *)PyArray_GETPTR3(SwC, j1, i1, i2));
    }

    return 0;
}

PyObject *update_qubo_switch_cost(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
    PyObject *s_capsule = getNamed("matrix", args, kwargs, 0);

    if (!PyCapsule_IsValid(s_capsule, "sbf16")) {
        PyErr_SetString(PyExc_TypeError, "data invalid");
        return 0;
    }

    Symmetric<bfloat16> *Q =
        (Symmetric<bfloat16> *)PyCapsule_GetPointer(s_capsule, "sbf16");

    const float lam =
        (float)PyFloat_AsDouble(getNamed("lam", args, kwargs, 1, Py_None));

    const size_t T =
        PyLong_AsUnsignedLong(getNamed("T", args, kwargs, 2, Py_None));
    const size_t n =
        PyLong_AsUnsignedLong(getNamed("n", args, kwargs, 3, Py_None));
    const size_t k =
        PyLong_AsUnsignedLong(getNamed("k", args, kwargs, 4, Py_None));

    const size_t N = T * n * k;

    PyArrayObject *SwC = (PyArrayObject *)getNamed("SwC", args, kwargs, 5);

    if (SwC == NULL || !PyArray_Check(SwC)) {
        PyErr_SetString(PyExc_TypeError, "data invalid");
        return 0;
    }

    double lb = 0;
    double ub = get_switch_upper_bound(SwC, T);

    // #pragma omp parallel for collapse(2)
    for (size_t idx = 0; idx < N; ++idx) {
        for (size_t jdx = idx; jdx < N; ++jdx) {
            const float val =
                lam * norm_ele(T, n, idx, jdx, lb, ub,
                               get_qubo_switch_cost_ij(idx, jdx, T, n, k, SwC));
            Q->add(idx, jdx, val);
        }
    }

    return s_capsule;
}

// ** Cost QUBO
void get_cost_lb_ub(PyArrayObject *C, double *lb, double *ub) {
    npy_intp *shape = PyArray_DIMS(C);
    const size_t &T = shape[0];
    const size_t &n = shape[1];
    const size_t &k = shape[2];
    *lb = 0;
    *ub = 0;
    for (int t = 0; t < T; ++t) {
        for (int j = 0; j < n; ++j) {
            *lb = *lb + *((double *)PyArray_GETPTR3(C, t, j, 0));
            *ub = *ub + *((double *)PyArray_GETPTR3(C, t, j, k - 1));
        }
    }
}

double get_qubo_cost_ij(const size_t &idx, const size_t &jdx, const size_t &T,
                        const size_t &n, const size_t &k, PyArrayObject *cost) {
    const size_t bases[] = {T, n, k};
    const size_t N = T * n * k;
    if (idx == jdx && idx < N) {
        size_t idxes[3];
        get_idxes<3>(idxes, idx, bases);

        const size_t &t = idxes[0];
        const size_t &j = idxes[1];
        const size_t &i = idxes[2];

        return *((double *)PyArray_GETPTR3(cost, t, j, i));
    }
    return 0;
}

PyObject *update_qubo_cost(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *s_capsule = getNamed("matrix", args, kwargs, 0);

    if (!PyCapsule_IsValid(s_capsule, "sbf16")) {
        PyErr_SetString(PyExc_TypeError, "data invalid");
        return 0;
    }

    Symmetric<bfloat16> *Q =
        (Symmetric<bfloat16> *)PyCapsule_GetPointer(s_capsule, "sbf16");

    const float lam =
        (float)PyFloat_AsDouble(getNamed("lam", args, kwargs, 1, Py_None));

    PyArrayObject *cost =
        (PyArrayObject *)getNamed("cost_vals", args, kwargs, 2);

    npy_intp *shape = PyArray_DIMS(cost);
    const size_t &T = shape[0];
    const size_t &n = shape[1];
    const size_t &k = shape[2];

    const size_t N = T * n * k;

    double lb, ub;
    get_cost_lb_ub(cost, &lb, &ub);

    // #pragma omp parallel for
    for (size_t idx = 0; idx < N; ++idx) {
        const float val =
            lam * norm_ele(T, n, idx, idx, lb, ub,
                           get_qubo_cost_ij(idx, idx, T, n, k, cost));
        Q->add(idx, idx, val);
    }

    return s_capsule;
}

// ** Overlord QUBO
void get_overload_norm_bounds(PyArrayObject *M, PyArrayObject *S,
                              PyArrayObject *PE, PyArrayObject *oNorms,
                              double *oLB, double *oUB) {
    npy_intp *shape = PyArray_DIMS(M);
    const size_t &L = shape[0];
    const size_t &T = shape[1];
    PyArrayObject *min_loads = (PyArrayObject *)PyArray_MatrixProduct(
        (PyObject *)S,
        PyArray_Transpose((PyArrayObject *)PyArray_Min(PE, 2, NULL), NULL));
    PyArrayObject *max_loads = (PyArrayObject *)PyArray_MatrixProduct(
        (PyObject *)S,
        PyArray_Transpose((PyArrayObject *)PyArray_Max(PE, 2, NULL), NULL));
    *oNorms = *(PyArrayObject *)PyArray_NewLikeArray(M, NPY_KEEPORDER, NULL, 1);
    double lb = 0;
    double ub = 0;
    double *M_arr = (double *)PyArray_DATA(M);
    double *max_loads_arr = (double *)PyArray_DATA(max_loads);
    double *min_loads_arr = (double *)PyArray_DATA(min_loads);
    double *oNorms_arr = (double *)PyArray_DATA(oNorms);
    for (int i = 0; i < PyArray_SIZE(M); i++) {
        double min = M_arr[i] - max_loads_arr[i];
        double max = M_arr[i] - min_loads_arr[i];
        double min_norm = min / max;
        double max_norm = max / max;
        ub += (pow(min_norm, 2) / 2) - min_norm + 1;
        lb += (pow(max_norm, 2) / 2) - max_norm + 1;
        oNorms_arr[i] = (1 / max);
    }
    *oLB = lb;
    *oUB = ub;
}
double get_qubo_overload_unbalanced_ij(const size_t &idx, const size_t &jdx,
                                       const size_t &t, PyArrayObject *PE,
                                       PyArrayObject *nS, PyArrayObject *SnS,
                                       PyArrayObject *MnS, const double &offset,
                                       const double &m1, const double &m2,
                                       const double &lb, const double &ub) {
    // PySys_WriteStdout("\n((%d, %d), (%d, %d))", (int)t, (int)idx, (int)t,
    // (int)jdx);
    npy_intp *shape = PyArray_DIMS(PE);
    const size_t &T = shape[0];
    const size_t &n = shape[1];
    const size_t &k = shape[2];

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
    // PySys_WriteStdout("((%d,%d,%d),(%d,%d,%d))", (int)t, (int)j1, (int)i1,
    // (int)t, (int)j2, (int)i2); PySys_WriteStdout("sns Size: %d",
    // (int)PyArray_SIZE(SnS));
    const double pe1 = *((double *)PyArray_GETPTR3(PE, t, j1, i1));
    const double pe2 = *((double *)PyArray_GETPTR3(PE, t, j2, i2));
    const double sns = *((double *)PyArray_GETPTR2(SnS, j1, j2));
    double res = pe1 * pe2 * sns / 2;
    // double res = 0.0;
    if (idx == jdx) {
        const double ns = *((double *)PyArray_GETPTR2(nS, t, j1));
        const double mns = *((double *)PyArray_GETPTR2(MnS, t, j1));
        res += offset - (m1 - pe1 * ns) + (m2 / 2) - (pe1 * mns);
        // res += offset + (m2 / 2) - m1;
        // res += offset - m1;
    }
    return norm_ele(T, n, idx, jdx, lb, ub, res);
}

PyObject *update_qubo_overload_unbalanced(PyObject *self, PyObject *args,
                                          PyObject *kwargs) {
    PyObject *s_capsule = getNamed("matrix", args, kwargs, 0);

    if (!PyCapsule_IsValid(s_capsule, "sbf16")) {
        PyErr_SetString(PyExc_TypeError, "data invalid");
        return 0;
    }

    Symmetric<bfloat16> *Q =
        (Symmetric<bfloat16> *)PyCapsule_GetPointer(s_capsule, "sbf16");

    const float lam =
        (float)PyFloat_AsDouble(getNamed("lam", args, kwargs, 1, Py_None));
    PyArrayObject *M = (PyArrayObject *)getNamed("M", args, kwargs, 2);
    PyArrayObject *S = (PyArrayObject *)getNamed("S", args, kwargs, 3);
    PyArrayObject *PE = (PyArrayObject *)getNamed("PE", args, kwargs, 4);

    npy_intp *shape = PyArray_DIMS(PE);
    const size_t &T = shape[0];
    const size_t &n = shape[1];
    const size_t &k = shape[2];
    shape = PyArray_DIMS(M);
    const size_t &L = shape[0];

    const size_t nk = n * k;

    PyArrayObject *norms =
        (PyArrayObject *)PyArray_NewLikeArray(M, NPY_KEEPORDER, NULL, 1);
    double lb = 0;
    double ub = 0;
    get_overload_norm_bounds(M, S, PE, norms, &lb, &ub);

    double offset = ((double)(L * T)) / ((double)(T * n));
    PyArrayObject *normS = (PyArrayObject *)PyArray_MatrixProduct(
        PyArray_Transpose(norms, NULL), (PyObject *)S);
    PyArrayObject *Mnorms2 =
        (PyArrayObject *)PyArray_NewLikeArray(M, NPY_KEEPORDER, NULL, 1);
    double *Mnorms2_arr = (double *)PyArray_DATA(Mnorms2);
    double *M_arr = (double *)PyArray_DATA(M);
    double *norms_arr = (double *)PyArray_DATA(norms);
    double m1 = 0;
    double m2 = 0;
    for (int i = 0; i < T * L; ++i) {
        Mnorms2_arr[i] = M_arr[i] * norms_arr[i] * norms_arr[i];
        m1 += (M_arr[i] * norms_arr[i]) / (n * T);
        m2 += pow(M_arr[i] * norms_arr[i], 2) / (n * T);
    }
    // PySys_WriteStdout("\nm1: %f, m2: %f", m1, m2);
    // PySys_WriteStdout("\nlb: %f, ub: %f\n", lb, ub);
    PyArrayObject *MnS = (PyArrayObject *)PyArray_MatrixProduct(
        PyArray_Transpose(Mnorms2, NULL), (PyObject *)S);

    // #pragma omp parallel for collapse(2)
    for (size_t t = 0; t < T; ++t) {
        double *SnS_arr = (double *)malloc(n * n * sizeof(double));
        for (size_t j1 = 0; j1 < n; ++j1) {
            for (size_t j2 = 0; j2 < n; ++j2) {
                size_t arr_idx = j1 * n + j2;
                SnS_arr[arr_idx] = 0;
                for (size_t l1 = 0; l1 < L; ++l1) {
                    double s1 = *(double *)PyArray_GETPTR2(S, l1, j1);
                    double norm_val = *(double *)PyArray_GETPTR2(norms, l1, t);
                    double s2 = *(double *)PyArray_GETPTR2(S, l1, j2);
                    SnS_arr[arr_idx] += s1 * norm_val * norm_val * s2;
                }
            }
        }
        npy_intp dims[2] = {0};
        dims[0] = n;
        dims[1] = n;
        PyArrayObject *SnS = (PyArrayObject *)PyArray_SimpleNewFromData(
            2, dims, NPY_FLOAT64, (void *)SnS_arr);
        Py_INCREF(SnS);

        for (size_t nk_idx = 0; nk_idx < nk; ++nk_idx) {
            size_t idx = nk_idx + (t * nk);
            for (size_t nk_jdx = nk_idx; nk_jdx < nk; ++nk_jdx) {
                size_t jdx = nk_jdx + (t * nk);
                const double val = lam * get_qubo_overload_unbalanced_ij(
                    nk_idx, nk_jdx, t, PE, normS, SnS, MnS, offset, m1, m2, lb,
                    ub);
                Q->add(idx, jdx, val);
            }
        }
    }
    return s_capsule;
}

// ** Power QUBO
double get_power_eq_ub(PyArrayObject *PE, PyArrayObject *Tau) {
    npy_intp *shape = PyArray_DIMS(PE);
    const size_t &T = shape[0];
    const size_t &n = shape[1];
    const size_t &k = shape[2];

    double ub1 = 0;
    double ub2 = 0;
    for (int t = 0; t < T; ++t) {
        double cur_tau = *(double *)PyArray_GETPTR1(Tau, t);
        double sum1 = -1 * cur_tau;
        double sum2 = -1 * cur_tau;
        for (int j = 0; j < n; ++j) {
            sum1 += *(double *)PyArray_GETPTR3(PE, t, j, 0);
            sum2 += *(double *)PyArray_GETPTR3(PE, t, j, k - 1);
        }
        ub1 += pow(sum1, 2);
        ub2 += pow(sum2, 2);
    }
    // PySys_WriteStdout("UB1: %f, UB2: %f", ub1, ub2);
    return std::max(ub1, ub2);
}

PyObject *update_qubo_power_eq(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
    PyObject *s_capsule = getNamed("matrix", args, kwargs, 0);

    if (!PyCapsule_IsValid(s_capsule, "sbf16")) {
        PyErr_SetString(PyExc_TypeError, "data invalid");
        return 0;
    }

    Symmetric<bfloat16> *Q =
        (Symmetric<bfloat16> *)PyCapsule_GetPointer(s_capsule, "sbf16");

    const float lam =
        (float)PyFloat_AsDouble(getNamed("lam", args, kwargs, 1, Py_None));

    PyArrayObject *PE = (PyArrayObject *)getNamed("PE", args, kwargs, 2);

    PyArrayObject *Tau = (PyArrayObject *)getNamed("Tau", args, kwargs, 3);

    npy_intp *shape = PyArray_DIMS(PE);
    const size_t &T = shape[0];
    const size_t &n = shape[1];
    const size_t &k = shape[2];

    const size_t N = T * n * k;
    const size_t nk = n * k;
    const size_t bases[] = {T, n, k};
    const size_t nk_bases[] = {n, k};

    double tau_c = 0.0;
    double *Tau_arr = (double *)PyArray_DATA(Tau);
    for (int tid = 0; tid < PyArray_SIZE(Tau); ++tid) {
        tau_c += pow(Tau_arr[tid], 2) / (n * T);
    }

    double lb = 0.0;
    double ub = get_power_eq_ub(PE, Tau);

    // #pragma omp parallel for collapse(2)
    for (size_t t = 0; t < T; ++t) {
        const double cur_Tau = *((double *)PyArray_GETPTR1(Tau, t));
        for (size_t nk_idx = 0; nk_idx < nk; ++nk_idx) {
            size_t idx = nk_idx + (t * nk);
            size_t idxes[2];
            get_idxes<2>(idxes, nk_idx, nk_bases);
            const size_t &j1 = idxes[0];
            const size_t &i1 = idxes[1];

            const double pe1 = *((double *)PyArray_GETPTR3(PE, t, j1, i1));

            for (size_t nk_jdx = nk_idx; nk_jdx < nk; ++nk_jdx) {
                size_t jdx = nk_jdx + (t * nk);
                size_t idxes[2];
                get_idxes<2>(idxes, nk_jdx, nk_bases);
                const size_t &j2 = idxes[0];
                const size_t &i2 = idxes[1];

                const double pe2 = *((double *)PyArray_GETPTR3(PE, t, j2, i2));
                double res = pe1 * pe2;
                if (nk_idx == nk_jdx) {
                  res += tau_c - 2 * cur_Tau * pe1;
                }
                Q->add(idx, jdx, lam * norm_ele(T, n, idx, jdx, lb, ub, res));
            }
        }
    }
    return s_capsule;
}

static PyMethodDef methods[] = {
    {"create_symmetric_bf16", (PyCFunction)create_symmetric_bf16,
     METH_VARARGS | METH_KEYWORDS,
     "Create a symmetric NxN matrix with 16bit precision (bf16)."},
    {"dim", (PyCFunction)py_getN, METH_VARARGS | METH_KEYWORDS,
     "Return the dimension N of a symmetric NxN matrix."},
    {"len", (PyCFunction)py_getL, METH_VARARGS | METH_KEYWORDS,
     "Return the storage length of a symmetric NxN matrix."},
    {"clear", (PyCFunction)py_clear, METH_VARARGS | METH_KEYWORDS,
     "Set all values to 0."},
    {"get", (PyCFunction)py_get, METH_VARARGS | METH_KEYWORDS,
     "Get value at location (i,j)."},
    {"set", (PyCFunction)py_set, METH_VARARGS | METH_KEYWORDS,
     "Set val at location (i,j)."},
    {"add", (PyCFunction)py_add, METH_VARARGS | METH_KEYWORDS,
     "Add val at location (i,j)."},
    {"mul", (PyCFunction)py_add, METH_VARARGS | METH_KEYWORDS,
     "Mult val to location (i,j)."},
    {"update_qubo_switch_cost", (PyCFunction)update_qubo_switch_cost,
     METH_VARARGS | METH_KEYWORDS, "Construct large QUBO matrix."},
    {"update_qubo_cost", (PyCFunction)update_qubo_cost,
     METH_VARARGS | METH_KEYWORDS, "Construct large QUBO matrix."},
    {"update_qubo_overload_unbalanced",
     (PyCFunction)update_qubo_overload_unbalanced, METH_VARARGS | METH_KEYWORDS,
     "Construct large QUBO matrix."},
    {"update_qubo_power_eq", (PyCFunction)update_qubo_power_eq,
     METH_VARARGS | METH_KEYWORDS, "Construct large QUBO matrix."},
    {"to_numpy", (PyCFunction)to_numpy, METH_VARARGS | METH_KEYWORDS,
     "Convert to 2-dim numpy array."},
    {0, 0, 0, 0}};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "qe_native", 0, -1,
                                    methods};

PyMODINIT_FUNC PyInit_qe_native() {
    import_array();

    PyObject *nprand = PyImport_ImportModule("numpy.random");
    PyObject *def_rng = PyObject_GetAttrString(nprand, "default_rng");
    PyObject *pyDefRng = PyObject_CallNoArgs(def_rng);
    PyObject *bit_generator = PyObject_GetAttrString(pyDefRng, "bit_generator");
    PyObject *bitgen_capsule = PyObject_GetAttrString(bit_generator, "capsule");
    B = (bitgen_t *)PyCapsule_GetPointer(bitgen_capsule, "BitGenerator");

    return PyModule_Create(&module);
}
