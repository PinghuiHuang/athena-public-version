//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file plm_simple.cpp
//  \brief  piecewise linear reconstruction for both uniform and non-uniform meshes
//  Operates on the entire nx4 range of a single AthenaArray<Real> input (no MHD).
//  No assumptions of hydrodynamic fluid variable input; no characteristic projection.

// REFERENCES:
// (Mignone) A. Mignone, "High-order conservative reconstruction schemes for finite volume
// methods in cylindrical and spherical coordinates", JCP, 270, 784 (2014)
//========================================================================================

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "reconstruction.hpp"

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::PiecewiseLinearX1()
//  \brief

void Reconstruction::PiecewiseLinearX1_DustFluids(
    const int k, const int j, const int il, const int iu,
    const AthenaArray<Real> &prim_df,
    AthenaArray<Real> &prim_df_l, AthenaArray<Real> &prim_df_r) {
  Coordinates *pco = pmy_block_->pcoord;
  // set work arrays to shallow copies of scratch arrays
  AthenaArray<Real> &qc = scr1_ni_df_, &d_prim_dfl = scr2_ni_df_, &dprim_df_r = scr3_ni_df_,
                   &d_prim_dfm = scr4_ni_df_;
  const int nu = prim_df.GetDim4() - 1;

  // compute L/R slopes for each variable
  for (int n=0; n<=nu; ++n) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      // renamed dw* -> d_prim_df* from plm.cpp
      d_prim_dfl(n,i) = (prim_df(n,k,j,i  ) - prim_df(n,k,j,i-1));
      dprim_df_r(n,i) = (prim_df(n,k,j,i+1) - prim_df(n,k,j,i  ));
      qc(n,i) = prim_df(n,k,j,i);
    }
  }

  //std::cout << " In PiecewiseLinearX1 #2 ========, ph->u.getSize() is " << pmy_block_->phydro->u.GetSize() << std::endl;
  // Apply simplified van Leer (VL) limiter expression for a Cartesian-like coordinate
  // with uniform mesh spacing
  if (uniform[X1DIR] && !curvilinear[X1DIR]) {
    for (int n=0; n<=nu; ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
      for (int i=il; i<=iu; ++i) {
        Real d_prim_df2 = d_prim_dfl(n,i)*dprim_df_r(n,i);
        d_prim_dfm(n,i) = 2.0*d_prim_df2/(d_prim_dfl(n,i) + dprim_df_r(n,i));
        if (d_prim_df2 <= 0.0) d_prim_dfm(n,i) = 0.0;
      }
    }

  //std::cout << " In PiecewiseLinearX1 #2.1 |||||||||||, ph->u.getSize() is " << pmy_block_->phydro->u.GetSize() << std::endl;
    // Apply general VL limiter expression w/ the Mignone correction for a Cartesian-like
    // coordinate with nonuniform mesh spacing or for any curvilinear coordinate spacing
  } else {
    for (int n=0; n<=nu; ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
      for (int i=il; i<=iu; ++i) {
        Real d_prim_dfF =  dprim_df_r(n,i)*pco->dx1f(i)/pco->dx1v(i);
        Real d_prim_dfB =  d_prim_dfl(n,i)*pco->dx1f(i)/pco->dx1v(i-1);
        Real d_prim_df2 = d_prim_dfF*d_prim_dfB;
        // cf, cb -> 2 (uniform Cartesian mesh / original VL value) w/ vanishing curvature
        // (may not exactly hold for nonuniform meshes, but converges w/ smooth
        // nonuniformity)
        Real cf = pco->dx1v(i  )/(pco->x1f(i+1) - pco->x1v(i)); // (Mignone eq 33)
        Real cb = pco->dx1v(i-1)/(pco->x1v(i  ) - pco->x1f(i));
        // (modified) VL limiter (Mignone eq 37)
        // (dQ^F term from eq 31 pulled into eq 37, then multiply by (dQ^F/dQ^F)^2)
        d_prim_dfm(n,i) = (d_prim_df2*(cf*d_prim_dfB + cb*d_prim_dfF)/
                    (SQR(d_prim_dfB) + SQR(d_prim_dfF) + d_prim_df2*(cf + cb - 2.0)));
        if (d_prim_df2 <= 0.0) d_prim_dfm(n,i) = 0.0; // ---> no concern for divide-by-0 in above line

        // Real v = d_prim_dfB/d_prim_dfF;
        // monotoniced central (MC) limiter (Mignone eq 38)
        // (std::min calls should avoid issue if divide-by-zero causes v=Inf)
        //d_prim_dfm(n,i) = d_prim_dfF*std::max(0.0, std::min(0.5*(1.0 + v), std::min(cf, cb*v)));
      }
    }
  }

  // compute ql_(i+1/2) and prim_df_r_(i-1/2) using limited slopes
  for (int n=0; n<=nu; ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
    for (int i=il; i<=iu; ++i) {
      // Mignone equation 30
      prim_df_l(n,i+1) = qc(n,i) + ((pco->x1f(i+1) - pco->x1v(i))/pco->dx1f(i))*d_prim_dfm(n,i);
      prim_df_r(n,i  ) = qc(n,i) - ((pco->x1v(i  ) - pco->x1f(i))/pco->dx1f(i))*d_prim_dfm(n,i);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::PiecewiseLinearX2()
//  \brief

void Reconstruction::PiecewiseLinearX2_DustFluids(
    const int k, const int j, const int il, const int iu,
    const AthenaArray<Real> &prim_df,
    AthenaArray<Real> &prim_df_l, AthenaArray<Real> &prim_df_r) {
  Coordinates *pco = pmy_block_->pcoord;
  // set work arrays to shallow copies of scratch arrays
  AthenaArray<Real> &qc = scr1_ni_df_, &d_prim_dfl = scr2_ni_df_,
                   &dprim_df_r = scr3_ni_df_, &d_prim_dfm = scr4_ni_df_;
  const int nu = prim_df.GetDim4() - 1;

  // compute L/R slopes for each variable
  for (int n=0; n<=nu; ++n) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      // renamed dw* -> d_prim_df* from plm.cpp
      d_prim_dfl(n,i) = (prim_df(n,k,j  ,i) - prim_df(n,k,j-1,i));
      dprim_df_r(n,i) = (prim_df(n,k,j+1,i) - prim_df(n,k,j  ,i));
      qc(n,i) = prim_df(n,k,j,i);
    }
  }

  // Apply simplified van Leer (VL) limiter expression for a Cartesian-like coordinate
  // with uniform mesh spacing
  if (uniform[X2DIR] && !curvilinear[X2DIR]) {
    for (int n=0; n<=nu; ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
      for (int i=il; i<=iu; ++i) {
        Real d_prim_df2 = d_prim_dfl(n,i)*dprim_df_r(n,i);
        d_prim_dfm(n,i) = 2.0*d_prim_df2/(d_prim_dfl(n,i) + dprim_df_r(n,i));
        if (d_prim_df2 <= 0.0) d_prim_dfm(n,i) = 0.0;
      }
    }

    // Apply general VL limiter expression w/ the Mignone correction for a Cartesian-like
    // coordinate with nonuniform mesh spacing or for any curvilinear coordinate spacing
  } else {
    Real cf = pco->dx2v(j  )/(pco->x2f(j+1) - pco->x2v(j));
    Real cb = pco->dx2v(j-1)/(pco->x2v(j  ) - pco->x2f(j));
    Real dxF = pco->dx2f(j)/pco->dx2v(j); // dimensionless, not technically a dx quantity
    Real dxB = pco->dx2f(j)/pco->dx2v(j-1);
    for (int n=0; n<=nu; ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
      for (int i=il; i<=iu; ++i) {
        Real d_prim_dfF =  dprim_df_r(n,i)*dxF;
        Real d_prim_dfB =  d_prim_dfl(n,i)*dxB;
        Real d_prim_df2 = d_prim_dfF*d_prim_dfB;
        // (modified) VL limiter (Mignone eq 37)
        d_prim_dfm(n,i) = (d_prim_df2*(cf*d_prim_dfB + cb*d_prim_dfF)/
                    (SQR(d_prim_dfB) + SQR(d_prim_dfF) + d_prim_df2*(cf + cb - 2.0)));
        if (d_prim_df2 <= 0.0) d_prim_dfm(n,i) = 0.0; // ---> no concern for divide-by-0 in above line

        // Real v = d_prim_dfB/d_prim_dfF;
        // // monotoniced central (MC) limiter (Mignone eq 38)
        // // (std::min calls should avoid issue if divide-by-zero causes v=Inf)
        // d_prim_dfm(n,i) = d_prim_dfF*std::max(0.0, std::min(0.5*(1.0 + v), std::min(cf, cb*v)));
      }
    }
  }

  // compute ql_(j+1/2) and prim_df_r_(j-1/2) using limited slopes
  // dimensionless, not technically a "dx" quantity
  Real dxp = (pco->x2f(j+1) - pco->x2v(j))/pco->dx2f(j);
  Real dxm = (pco->x2v(j  ) - pco->x2f(j))/pco->dx2f(j);
  for (int n=0; n<=nu; ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
    for (int i=il; i<=iu; ++i) {
      prim_df_l(n,i) = qc(n,i) + dxp*d_prim_dfm(n,i);
      prim_df_r(n,i) = qc(n,i) - dxm*d_prim_dfm(n,i);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::PiecewiseLinearX3()
//  \brief

void Reconstruction::PiecewiseLinearX3_DustFluids(
    const int k, const int j, const int il, const int iu,
    const AthenaArray<Real> &prim_df,
    AthenaArray<Real> &prim_df_l, AthenaArray<Real> &prim_df_r) {
  Coordinates *pco = pmy_block_->pcoord;
  // set work arrays to shallow copies of scratch arrays
  AthenaArray<Real> &qc = scr1_ni_df_, &d_prim_dfl = scr2_ni_df_, &dprim_df_r = scr3_ni_df_,
                   &d_prim_dfm = scr4_ni_df_;
  const int nu = prim_df.GetDim4() - 1;

  // compute L/R slopes for each variable
  for (int n=0; n<=nu; ++n) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      // renamed dw* -> d_prim_df* from plm.cpp
      d_prim_dfl(n,i) = (prim_df(n,k  ,j,i) - prim_df(n,k-1,j,i));
      dprim_df_r(n,i) = (prim_df(n,k+1,j,i) - prim_df(n,k  ,j,i));
      qc(n,i) = prim_df(n,k,j,i);
    }
  }

  // Apply simplified van Leer (VL) limiter expression for a Cartesian-like coordinate
  // with uniform mesh spacing
  if (uniform[X3DIR]) {
    for (int n=0; n<=nu; ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
      for (int i=il; i<=iu; ++i) {
        Real d_prim_df2 = d_prim_dfl(n,i)*dprim_df_r(n,i);
        d_prim_dfm(n,i) = 2.0*d_prim_df2/(d_prim_dfl(n,i) + dprim_df_r(n,i));
        if (d_prim_df2 <= 0.0) d_prim_dfm(n,i) = 0.0;
      }
    }

    // Apply original VL limiter's general expression for a Cartesian-like coordinate with
    // nonuniform mesh spacing
  } else {
    Real dxF = pco->dx3f(k)/pco->dx3v(k);
    Real dxB = pco->dx3f(k)/pco->dx3v(k-1);
    for (int n=0; n<=nu; ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
      for (int i=il; i<=iu; ++i) {
        Real d_prim_dfF =  dprim_df_r(n,i)*dxF;
        Real d_prim_dfB =  d_prim_dfl(n,i)*dxB;
        Real d_prim_df2 = d_prim_dfF*d_prim_dfB;
        // original VL limiter (Mignone eq 36)
        d_prim_dfm(n,i) = 2.0*d_prim_df2/(d_prim_dfF + d_prim_dfB);
        // d_prim_df2 > 0 ---> d_prim_dfF, d_prim_dfB are nonzero and have the same sign ----> no risk for
        // (d_prim_dfF + d_prim_dfB) = 0 cancellation causing a divide-by-0 in the above line
        if (d_prim_df2 <= 0.0) d_prim_dfm(n,i) = 0.0;
      }
    }
  }

  // compute ql_(k+1/2) and prim_df_r_(k-1/2) using limited slopes
  Real dxp = (pco->x3f(k+1) - pco->x3v(k))/pco->dx3f(k);
  Real dxm = (pco->x3v(k  ) - pco->x3f(k))/pco->dx3f(k);
  for (int n=0; n<=nu; ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
    for (int i=il; i<=iu; ++i) {
      prim_df_l(n,i) = qc(n,i) + dxp*d_prim_dfm(n,i);
      prim_df_r(n,i) = qc(n,i) - dxm*d_prim_dfm(n,i);
    }
  }
  return;
}
