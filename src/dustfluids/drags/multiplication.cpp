//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file multiplication.cpp
//! Compute the multiplications between matrixes.

// C++ headers
#include <algorithm>   // min,max
#include <limits>
#include <cstring>    // strcmp
#include <sstream>

// Athena++ headers
#include "../../defs.hpp"
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../dustfluids.hpp"
#include "dust_gas_drag.hpp"

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif


// Matrix Multiplication
void DustGasDrag::Multiplication(const AthenaArray<Real> &a_matrix,
                  const AthenaArray<Real> &b_matrix, AthenaArray<Real> &c_matrix)
{
  const int m_a = a_matrix.GetDim2();
  const int n_a = a_matrix.GetDim1();

  const int m_b = b_matrix.GetDim2();
  const int n_b = b_matrix.GetDim1();

  const int m_c = c_matrix.GetDim2();
  const int n_c = c_matrix.GetDim1();

  if ( m_b > 1 ) {
    if ( (n_a != m_b) || (m_a != m_c) || (n_b != n_c) ) {
      std::stringstream msg;
      msg << "### FATAL ERROR in DustGasDrag::Multiplication, Bad Dimensions." << std::endl;
      ATHENA_ERROR(msg);
    }

    c_matrix.ZeroClear();

    for(int m=0; m<m_a; ++m) {
      for(int n=0; n<n_b; ++n) {
#pragma omp simd
        for(int s=0; s<m_b; ++s) {
          c_matrix(m, n) += a_matrix(m, s)*b_matrix(s, n);
        }
      }
    }
  }
  else {
    if ( (n_a != n_b) || (n_a != n_c) || (m_c != 1) ) {
      std::stringstream msg;
      msg << "### FATAL ERROR in DustGasDrag::Multiplication, Bad Dimensions." << std::endl;
      ATHENA_ERROR(msg);
    }

    c_matrix.ZeroClear();
    for(int m=0; m<m_a; ++m) {
#pragma omp simd
      for(int n=0; n<n_b; ++n) {
        c_matrix(n) += a_matrix(m, n)*b_matrix(m);
      }
    }
  }
  return;
}


void DustGasDrag::Multiplication(const Real a_num, const AthenaArray<Real> &b_matrix,
                                  AthenaArray<Real> &c_matrix)
{
  const int m_b = b_matrix.GetDim2();
  const int n_b = b_matrix.GetDim1();

  const int m_c = c_matrix.GetDim2();
  const int n_c = c_matrix.GetDim1();

  if ( (m_b != m_c) || (n_b != n_c) ) {
    std::stringstream msg;
    msg << "### FATAL ERROR in DustGasDrag::Multiplication, Bad Dimensions." << std::endl;
    ATHENA_ERROR(msg);
  }

	c_matrix.ZeroClear();
  for(int m=0; m<m_b; ++m) {
#pragma omp simd
    for(int n=0; n<n_c; ++n) {
			c_matrix(m, n) = a_num*b_matrix(m, n);
		}
	}
  return;
}


void DustGasDrag::Multiplication(const Real a_num, AthenaArray<Real> &b_matrix)
{
  const int m_b = b_matrix.GetDim2();
  const int n_b = b_matrix.GetDim1();

  for(int m=0; m<m_b; ++m) {
#pragma omp simd
    for(int n=0; n<n_b; ++n) {
			b_matrix(m, n) *= a_num;
		}
	}
  return;
}
