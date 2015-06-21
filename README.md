# mr3smp

## Overview

The project provides code for a _multi-threaded_ variant of the algorithm of _Multiple Relatively Robust Representations_. In most cases it is the fastest solver for dense symmetric and Hermitian eigenproblems. The code is free to use and edit and therefore can be used by users as well as library developers to integrate the code or using it as start for there own implementation. 

The MRRR algorithm is a solver for the symmetric tridiagonal eigenproblem, which lies at the heart of direct methods for dense symmetric and Hermitian eigenproblems. For convenience we therefore include, besides the tridiagonal solver, routines for the following dense problems:
 * _symmetric:_ A*x = lambda*x, with A=A^T^
 * _Hermitian:_ A*x = lambda*x, with A=A^H^
 * _generalized symmetric-definite:_ A*x = lambda*B*x or A*B*x = lambda*x or B*A =  lambda*x, with A=A^T^, B=B^T^ and B definite
 * _generalized Hermitian-definite:_ A*x = lambda*B*x or A*B*x = lambda*x or B*A = lambda*x, with A=A^H^, B=B^H^ and B definite

The matrices A and B can be stored either full or, although not recommended, in packed storage format. The routines for the dense problems require linking to LAPACK and an optimized BLAS. Please refer to the README file for more details.


## Mixed precision 

Mixed precision routines are used to improve the accuracy of the solver (as depicted below) at moderate cost. It can also be used to improve parallelism. 


## Usage

Please read USAGE.txt for more about how to use mr3smp.


## Citing 

When you use this code, kindly reference the following paper paper:
{{{
@article{Petschow2011:254,
    author  = "Matthias Petschow and Paolo Bientinesi",
    title   = "MR$^3$-SMP: A Symmetric Tridiagonal Eigensolver for Multi-Core Architectures",
    journal = "Parallel Computing",
    year    = 2011,
    volume  = 37,
    number  = 12,
}
}}}
