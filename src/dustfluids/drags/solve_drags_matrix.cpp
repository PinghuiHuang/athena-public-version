//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file dustfluids_diffusion.cpp
//  \brief Compute dustfluids fluxes corresponding to diffusion processes.

// C++ headers
#include <algorithm>   // min,max
#include <limits>
#include <cstring>    // strcmp

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


void DustGasDrag::SingleDustNoFeedbackImplicit(MeshBlock *pmb, const int stage, const Real dt,
      const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &u, AthenaArray<Real> &cons_df) {

  const bool f2          = pmb->pmy_mesh->f2;
  const bool f3          = pmb->pmy_mesh->f3;
  Coordinates *pco       = pmb->pcoord;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  int dust_id = 0;
  int rho_id  = 4*dust_id;
  int v1_id   = rho_id + 1;
  int v2_id   = rho_id + 2;
  int v3_id   = rho_id + 3;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        // Alias the primitives of gas
        const Real &gas_d  = w(IDN, k, j, i);
        const Real &gas_v1 = w(IVX, k, j, i);
        const Real &gas_v2 = w(IVY, k, j, i);
        const Real &gas_v3 = w(IVZ, k, j, i);
        const Real &gas_p  = w(IPR, k, j, i);

        // Alias the primitives of dust
        const Real &dust_d  = prim_df(rho_id, k, j, i);
        const Real &dust_v1 = prim_df(v1_id,  k, j, i);
        const Real &dust_v2 = prim_df(v2_id,  k, j, i);
        const Real &dust_v3 = prim_df(v3_id,  k, j, i);

        // Alias the conserves of dust
        Real &dust_m1 = cons_df(v1_id, k, j, i);
        Real &dust_m2 = cons_df(v2_id, k, j, i);
        Real &dust_m3 = cons_df(v3_id, k, j, i);

        // Calculate the collisional parameters of dust and gas
        Real alpha_dg = 1.0/stopping_time(dust_id,k,j,i);
        Real alpha_gd = dust_d/gas_d*alpha_dg;

        // Update the Momentum of gas and dust
        Real A11   = 1.0 + alpha_gd*dt;
        Real A12   = -alpha_gd*dt;
        Real A21   = -alpha_dg*dt;
        Real A22   = 1.0 + alpha_dg*dt;
        Real deter = A11*A22 - A12*A21;

        Real dust_v1_new = (A11*dust_v1 - A21*gas_v1)/deter;
        Real dust_v2_new = (A11*dust_v2 - A21*gas_v2)/deter;
        Real dust_v3_new = (A11*dust_v3 - A21*gas_v3)/deter;

        Real delta_dust_m1 = dust_d*(dust_v1_new - dust_v1);
        Real delta_dust_m2 = dust_d*(dust_v2_new - dust_v2);
        Real delta_dust_m3 = dust_d*(dust_v3_new - dust_v3);

        dust_m1 += delta_dust_m1;
        dust_m2 += delta_dust_m2;
        dust_m3 += delta_dust_m3;

      }
    }
  }
  return;
}

void DustGasDrag::SingleDustFeedbackImplicit(MeshBlock *pmb, const int stage, const Real dt,
      const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      AthenaArray<Real> &u, AthenaArray<Real> &cons_df) {

  DustFluids  *pdf = pmb->pdustfluids;
  const bool f2          = pmb->pmy_mesh->f2;
  const bool f3          = pmb->pmy_mesh->f3;
  Coordinates *pco       = pmb->pcoord;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  int dust_id = 0;
  int rho_id  = 4*dust_id;
  int v1_id   = rho_id + 1;
  int v2_id   = rho_id + 2;
  int v3_id   = rho_id + 3;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        // Alias the primitives of gas
        const Real &gas_d  = w(IDN, k, j, i);
        const Real &gas_v1 = w(IVX, k, j, i);
        const Real &gas_v2 = w(IVY, k, j, i);
        const Real &gas_v3 = w(IVZ, k, j, i);
        const Real &gas_p  = w(IPR, k, j, i);

        // Alias the conserves of gas
        Real &gas_m1 = u(IM1, k, j, i);
        Real &gas_m2 = u(IM2, k, j, i);
        Real &gas_m3 = u(IM3, k, j, i);
        Real &gas_e  = u(IEN, k, j, i);

        // Alias the primitives of dust
        const Real &dust_d  = prim_df(rho_id, k, j, i);
        const Real &dust_v1 = prim_df(v1_id,  k, j, i);
        const Real &dust_v2 = prim_df(v2_id,  k, j, i);
        const Real &dust_v3 = prim_df(v3_id,  k, j, i);

        // Alias the conserves of dust
        Real &dust_m1  = cons_df(v1_id,  k, j, i);
        Real &dust_m2  = cons_df(v2_id,  k, j, i);
        Real &dust_m3  = cons_df(v3_id,  k, j, i);

        // Calculate the collisional parameters of dust and gas
        Real alpha_dg = 1.0/stopping_time(dust_id,k,j,i);
        Real alpha_gd = dust_d/gas_d*alpha_dg;

        // Update the Momentum of gas and dust
        Real A11   = 1.0 + alpha_gd*dt;
        Real A12   = -alpha_gd*dt;
        Real A21   = -alpha_dg*dt;
        Real A22   = 1.0 + alpha_dg*dt;
        Real deter = A11*A22 - A12*A21;

        Real dust_v1_new = (A11*dust_v1 - A21*gas_v1)/deter;
        Real dust_v2_new = (A11*dust_v2 - A21*gas_v2)/deter;
        Real dust_v3_new = (A11*dust_v3 - A21*gas_v3)/deter;

        Real gas_v1_new = (A22*gas_v1  - A12*dust_v1)/deter;
        Real gas_v2_new = (A22*gas_v2  - A12*dust_v2)/deter;
        Real gas_v3_new = (A22*gas_v3  - A12*dust_v3)/deter;

        Real delta_dust_m1 = dust_d*(dust_v1_new - dust_v1);
        Real delta_dust_m2 = dust_d*(dust_v2_new - dust_v2);
        Real delta_dust_m3 = dust_d*(dust_v3_new - dust_v3);

        Real delta_gas_m1  = gas_d*(gas_v1_new - gas_v1);
        Real delta_gas_m2  = gas_d*(gas_v2_new - gas_v2);
        Real delta_gas_m3  = gas_d*(gas_v3_new - gas_v3);

        dust_m1 += delta_dust_m1;
        dust_m2 += delta_dust_m1;
        dust_m3 += delta_dust_m1;

        gas_m1  += delta_gas_m1;
        gas_m2  += delta_gas_m1;
        gas_m3  += delta_gas_m1;

        // Update the energy of gas if the gas is non barotropic.
        if (NON_BAROTROPIC_EOS)
          gas_e += delta_gas_m1*gas_v1 + delta_gas_m2*gas_v2 + delta_gas_m3*gas_v3;

        }
      }
    }
  return;
}
