#!/usr/bin/env python
from os.path import join
from platform import system

import pybind11
from numpy import get_include as numpy_incl
from setuptools import Extension, setup

SYSTEM = system()
if SYSTEM == "Windows":
    C_LINK_FLAGS = []
    C_COMP_FLAGS = ["/O2", "/openmp:experimental"]
else:  # GCC flags for Linux
    C_LINK_FLAGS = ["-fopenmp"]
    C_COMP_FLAGS = [
        "-O3",
        "-fopenmp",
        "-march=native",
        "-fdiagnostics-color=always",
    ]
#    C_COMP_FLAGS = ['-O0', '-g', '-fopenmp', '-march=native', '-fdiagnostics-color=always']

NP_INCL = numpy_incl()

setup(
    ext_modules=[
        Extension(
            name="qe_sparse",
            sources=["./quantum_energy/src/qubo_sparse.cpp"],
            libraries=["npymath", "npyrandom"],
            include_dirs=[
                NP_INCL, join('.' , 'quantum_energy', 'src'),
                pybind11.get_include(),
                join('.' , 'quantum_energy', 'src', 'eigen'),
            ],
            library_dirs=[
                join(NP_INCL, "..", "lib"),
                join(NP_INCL, "..", "..", "random", "lib"),
            ],  # no official way to retrieve these directories
            extra_compile_args=C_COMP_FLAGS,
            extra_link_args=C_LINK_FLAGS,
        ),
        Extension(
            name='qe_builder',
            sources=['./quantum_energy/src/qubo_builder.cpp'],
            libraries=['npymath','npyrandom'],
            include_dirs=[
                NP_INCL, join('.' , 'quantum_energy', 'src'),
                pybind11.get_include(),
                join('.' , 'quantum_energy', 'src', 'eigen'),
            ],
            library_dirs=[
                join(NP_INCL, '..', 'lib'),
                join(NP_INCL, '..', '..', 'random', 'lib')], # no official way to retrieve these directories
            extra_compile_args=C_COMP_FLAGS,
            extra_link_args=C_LINK_FLAGS
        )
    ],
    packages=[
        "quantum_energy",
        "quantum_energy.problem",
        "quantum_energy.solver",
    ],
    install_requires=["pybind11"],
    setup_requires=["pybind11"],
)
