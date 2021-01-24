//======================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file rotating_system_srcterms_dustfluids.cpp
//! \brief Adds coriolis force and centrifugal force
//======================================================================================

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../mesh/mesh.hpp"
#include "../dustfluids.hpp"

// this class header
#include "dustfluids_srcterms.hpp"

//--------------------------------------------------------------------------------------
//! \fn void DustFluidsSourceTerms::RotatingSystemSourceTermsDustFluids
//!             (const Real dt, const AthenaArray<Real> *flux_df,
//!              const AthenaArray<Real> &prim_df, AthenaArray<Real> &cons_df)
//! \brief source terms for the rotating system

void DustFluidsSourceTerms::RotatingSystemSourceTermsDustFluids
                 (const Real dt, const AthenaArray<Real> *flux_df,
                  const AthenaArray<Real> &prim_df, AthenaArray<Real> &cons_df) {
  MeshBlock *pmb = pmy_dustfluids_->pmy_block;
  if(std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    // dM1/dt = 2 \rho vc vp /r +\rho vc^2/r
    // dM2/dt = -2 \rho vc vr /r
    // dE/dt  = \rho vc^2 vr /r
    // vc     = r \Omega
    for (int n=0; n<NDUSTFLUIDS; ++n) {
      int dust_id = n;
      int rho_id  = 4*dust_id;
      int v1_id   = rho_id + 1;
      int v2_id   = rho_id + 2;
      int v3_id   = rho_id + 3;
      for (int k=pmb->ks; k<=pmb->ke; ++k) {
        for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
          for (int i=pmb->is; i<=pmb->ie; ++i) {
            Real den    = prim_df(rho_id, k, j, i);
            Real mom1   = den*prim_df(v1_id, k, j, i);
            Real ri     = pmb->pcoord->coord_src1_i_(i);
            Real rv     = pmb->pcoord->x1v(i);
            Real vc     = rv*Omega_0_;
            Real src    = SQR(vc); // (rOmega)^2
            Real flux_c = 0.5*(flux_df[X1DIR](rho_id, k, j, i)+flux_df[X1DIR](rho_id, k, j, i+1));
            cons_df(v1_id, k, j, i) += dt*ri*(2.0*vc*(den*prim_df(v2_id, k, j, i))+den*src);
            cons_df(v2_id, k, j, i) -= dt*ri*vc*(mom1 + flux_c);
          }
        }
      }
    }
  } else if(std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    // dM1/dt = 2 \rho vc vp / r
    //          +\rho (vc)^2 / r
    // dM2/dt = 2 \rho vp cot(\theta) vc / r
    //          + \rho cot(\theta) (vc)^2 /r
    // dM3/dt = -\rho vr (2 vc)/r
    //          -\rho vt (2 vc) cot(\theta) /r
    // dE/dt  = \rho vr (vc)^2/r
    //          + \rho vt (vc)^2 cot(\theta) /r
    // vc     = r sin(\theta)\Omega
    for (int n=0; n<NDUSTFLUIDS; ++n) {
      int dust_id = n;
      int rho_id  = 4*dust_id;
      int v1_id   = rho_id + 1;
      int v2_id   = rho_id + 2;
      int v3_id   = rho_id + 3;
      for (int k=pmb->ks; k<=pmb->ke; ++k) {
        for (int j=pmb->js; j<=pmb->je; ++j) {
          Real cv1 = pmb->pcoord->coord_src1_j_(j); // cot(theta)
          Real cv3 = pmb->pcoord->coord_src3_j_(j); // cot(\theta)
          Real sv  = std::sin(pmb->pcoord->x2v(j)); // sin(\theta)
#pragma omp simd
          for (int i=pmb->is; i<=pmb->ie; ++i) {
            Real den     = prim_df(rho_id,k,j,i);
            Real rv      = pmb->pcoord->x1v(i);
            Real ri      = pmb->pcoord->coord_src1_i_(i); // 1/r
            Real vc      = rv*sv*Omega_0_;
            Real src     = SQR(vc);                       // vc^2
            Real force   = den*ri*(2.0*vc*prim_df(v3_id, k, j, i)+src);
            Real flux_xc = 0.5*(flux_df[X1DIR](rho_id, k, j, i+1)+flux_df[X1DIR](rho_id, k, j, i));
            Real flux_yc = 0.5*(flux_df[X2DIR](rho_id, k, j+1, i)+flux_df[X2DIR](rho_id, k, j, i));
            cons_df(v1_id, k, j, i) += dt*force;
            cons_df(v2_id, k, j, i) += dt*force*cv1;
            cons_df(v3_id, k, j, i) -= dt*ri*vc*(den*prim_df(v1_id, k, j, i)+flux_xc
                                           +cv3*(den*prim_df(v2_id, k, j, i)+flux_yc));
          }
        }
      }
    }
  }
  return;
}
