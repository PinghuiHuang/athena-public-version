//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file addition.cpp
//! Compute the addition between matrixes.

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


// Matrix Addition
void DustGasDrag::Addition(const AthenaArray<Real> &a_matrix, const Real b_num,
                           const AthenaArray<Real> &b_matrix, AthenaArray<Real> &c_matrix)
{
  const int m_a = a_matrix.GetDim2();
  const int n_a = a_matrix.GetDim1();

  const int m_b = b_matrix.GetDim2();
  const int n_b = b_matrix.GetDim1();

  const int m_c = c_matrix.GetDim2();
  const int n_c = c_matrix.GetDim1();

  if ( (m_a != m_b) || (m_a != m_c) || (n_a != n_b) || (n_a != n_c) ) {
    std::stringstream msg;
    msg << "### FATAL ERROR in DustGasDrag::Addition, Bad Dimensions." << std::endl;
    ATHENA_ERROR(msg);
  }

  for(int m=0; m<m_a; ++m) {
#pragma omp simd
    for(int n=0; n<n_a; ++n) {
			c_matrix(m, n) = a_matrix(m, n) + b_num*b_matrix(m, n);
    }
  }

  return;
}


void DustGasDrag::Addition(AthenaArray<Real> &a_matrix, const Real b_num, const AthenaArray<Real> &b_matrix)
{
  const int m_a = a_matrix.GetDim2();
  const int n_a = a_matrix.GetDim1();

  const int m_b = b_matrix.GetDim2();
  const int n_b = b_matrix.GetDim1();

  if ( (m_a != m_b) || (n_a != n_b) ) {
    std::stringstream msg;
    msg << "### FATAL ERROR in DustGasDrag::Addition, Bad Dimensions." << std::endl;
    ATHENA_ERROR(msg);
  }

  for(int m=0; m<m_b; ++m) {
#pragma omp simd
    for(int n=0; n<n_b; ++n) {
			a_matrix(m, n) += b_num*b_matrix(m, n);
    }
  }
  return;
}


void DustGasDrag::Addition(const Real a_num, const Real b_num,
                      const AthenaArray<Real> &b_matrix, AthenaArray<Real> &c_matrix)
{
  const int m_b = b_matrix.GetDim2();
  const int n_b = b_matrix.GetDim1();

  const int m_c = c_matrix.GetDim2();
  const int n_c = c_matrix.GetDim1();

  if ( (m_b != m_c) || (n_b != n_c) ) {
    std::stringstream msg;
    msg << "### FATAL ERROR in DustGasDrag::Addition, Bad Dimensions." << std::endl;
    ATHENA_ERROR(msg);
  }

  Real delta;
  for(int m=0; m<m_b; ++m) {
#pragma omp simd
    for(int n=0; n<n_b; ++n) {
      m == n ? delta = 1.0 : delta = 0.0;
			c_matrix(m, n) = a_num*delta + b_num*b_matrix(m, n);
    }
  }
  return;
}


void DustGasDrag::Addition(const Real a_num, const Real b_num,
                            AthenaArray<Real> &b_matrix)
{
  const int m_b = b_matrix.GetDim2();
  const int n_b = b_matrix.GetDim1();

  Real delta;
  for(int m=0; m<m_b; ++m) {
#pragma omp simd
    for(int n=0; n<n_b; ++n) {
      m == n ? delta = 1.0 : delta = 0.0;
			b_matrix(m, n) = a_num*delta + b_num*b_matrix(m, n);
    }
  }
  return;
}
