#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <iostream>

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

void deconstruct_prob_qubo(PyObject *qubo_capsule) {
    Symmetric<bfloat16> *S =
        (Symmetric<bfloat16> *)PyCapsule_GetPointer(qubo_capsule, "qubo");
    delete S;
}

void get_tji(size_t T, size_t n, size_t k, size_t idx, size_t *ids) {
    size_t base = T * n * k;
    // t
    base = base / T;
    ids[0] = idx / base;
    idx = idx % base;
    // j
    base = base / n;
    ids[1] = idx / base;
    idx = idx % base;
    // i
    base = base / k;
    ids[2] = idx / base;
    idx = idx % base;
}

PyObject *create_prob_qubo(PyObject *self, PyObject *args, PyObject *kwargs) {
    const size_t T =
        PyLong_AsUnsignedLong(getNamed("T", args, kwargs, 0, Py_None));
    const size_t n =
        PyLong_AsUnsignedLong(getNamed("n", args, kwargs, 1, Py_None));
    const size_t k =
        PyLong_AsUnsignedLong(getNamed("k", args, kwargs, 2, Py_None));
    PyArrayObject *SwC = (PyArrayObject *)PyArray_FROM_OT(
        getNamed("SwC", args, kwargs, 3, Py_None), NPY_FLOAT);
    const size_t N = T * n * k;
    Symmetric<bfloat16> *S = new Symmetric<bfloat16>(N);
    size_t tji_1[3] = {0, 0, 0};
    size_t tji_2[3] = {0, 0, 0};
    for (int idx = 0; idx < N; idx++) {
        get_tji(T, n, k, idx, tji_1);
        for (int jdx = 0; jdx < N; jdx++) {
            get_tji(T, n, k, jdx, tji_2);
            if (tji_1[2] > 0 and tji_1[0] < T - 1 and
                tji_2[0] == tji_1[0] + 1 and tji_2[2] == tji_1[2] - 1) {
                float *e = (float *)PyArray_GETPTR3(SwC, tji_1[1], tji_1[2],
                                                    tji_1[2] - 1);
                S->add(idx, jdx, *e);
            }
            if (tji_1[2] < k - 1 and tji_1[0] < T - 1 and
                tji_2[0] == tji_1[0] + 1 and tji_2[2] == tji_1[2] + 1) {
                float *e = (float *)PyArray_GETPTR3(SwC, tji_1[1], tji_1[2],
                                                    tji_1[2] - 1);
                S->add(idx, jdx, *e);
            }
        }
    }
    PyObject *res = PyCapsule_New(S, "sbf16", deconstruct_prob_qubo);
    return res;
}
