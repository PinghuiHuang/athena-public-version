//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file dc_simple.cpp
//  \brief piecewise constant (donor cell) reconstruction
//  Operates on the entire nx4 range of a single AthenaArray<Real> input (no MHD).
//  No assumptions of hydrodynamic fluid variable input; no characteristic projection.

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "reconstruction.hpp"

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::DonorCellX1()
//  \brief reconstruct L/R surfaces of the i-th cells

void Reconstruction::DonorCellX1_DustFluids(const int k, const int j, const int il, const int iu,
                                 const AthenaArray<Real> &prim_df,
                                 AthenaArray<Real> &prim_df_l, AthenaArray<Real> &prim_df_r) {
  const int nu = prim_df.GetDim4() - 1;

  // compute L/R states for each variable
  for (int n=0; n<=nu; ++n) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      prim_df_l(n,i+1) =  prim_df_r(n,i) = prim_df(n,k,j,i);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::DonorCellX2()
//  \brief


void Reconstruction::DonorCellX2_DustFluids(const int k, const int j, const int il, const int iu,
                                 const AthenaArray<Real> &prim_df,
                                 AthenaArray<Real> &prim_df_l, AthenaArray<Real> &prim_df_r) {
  const int nu = prim_df.GetDim4() - 1;
  // compute L/R states for each variable
  for (int n=0; n<=nu; ++n) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      prim_df_l(n,i) = prim_df_r(n,i) = prim_df(n,k,j,i);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::DonorCellX3()
//  \brief

void Reconstruction::DonorCellX3_DustFluids(const int k, const int j, const int il, const int iu,
                                 const AthenaArray<Real> &prim_df,
                                 AthenaArray<Real> &prim_df_l, AthenaArray<Real> &prim_df_r) {
  const int nu = prim_df.GetDim4() - 1;
  // compute L/R states for each variable
  for (int n=0; n<=nu; ++n) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      prim_df_l(n,i) = prim_df_r(n,i) = prim_df(n,k,j,i);
    }
  }
  return;
}
