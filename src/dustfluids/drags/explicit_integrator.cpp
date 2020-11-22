//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file dustfluids_diffusion.cpp
//  \brief Compute dustfluids fluxes corresponding to diffusion processes.

// C++ headers
#include <algorithm>   // min, max
#include <limits>
#include <cstring>    // strcmp
#include <sstream>
#include <iostream>   // endl

// Athena++ headers
#include "../../defs.hpp"
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../mesh/mesh.hpp"
#include "../dustfluids.hpp"
#include "dust_gas_drag.hpp"
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif


void DustGasDrag::ExplicitFeedback(const int stage,
      const Real dt, const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      AthenaArray<Real> &u, AthenaArray<Real> &cons_df) {

  MeshBlock  *pmb = pmy_dustfluids_->pmy_block;
  DustFluids *pdf = pmy_dustfluids_;
  Hydro      *ph  = pmb->phydro;

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  AthenaArray<Real> force_x1(num_species);
  AthenaArray<Real> force_x2(num_species);
  AthenaArray<Real> force_x3(num_species);

  AthenaArray<Real> delta_m1_explicit(num_species);
  AthenaArray<Real> delta_m2_explicit(num_species);
  AthenaArray<Real> delta_m3_explicit(num_species);

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        force_x1.ZeroClear();
        force_x2.ZeroClear();
        force_x3.ZeroClear();

        delta_m1_explicit.ZeroClear();
        delta_m2_explicit.ZeroClear();
        delta_m3_explicit.ZeroClear();

        // Alias the primitives of gas
        const Real &gas_rho  = w(IDN, k, j, i);
        const Real &gas_v1   = w(IVX, k, j, i);
        const Real &gas_v2   = w(IVY, k, j, i);
        const Real &gas_v3   = w(IVZ, k, j, i);

        // Set the drag force
        for (int index=1; index<=NDUSTFLUIDS; ++index) {
          int dust_id          = index - 1;
          int rho_id           = 4 * dust_id;
          int v1_id            = rho_id + 1;
          int v2_id            = rho_id + 2;
          int v3_id            = rho_id + 3;
          const Real &dust_rho = prim_df(rho_id, k, j, i);
          Real alpha           = 1.0/(stopping_time(dust_id, k, j, i) + TINY_NUMBER);
          Real epsilon         = dust_rho/gas_rho;

          const Real &dust_v1 = prim_df(v1_id, k, j, i);
          const Real &dust_v2 = prim_df(v2_id, k, j, i);
          const Real &dust_v3 = prim_df(v3_id, k, j, i);

          Real gas_mom1 = gas_rho * gas_v1;
          Real gas_mom2 = gas_rho * gas_v2;
          Real gas_mom3 = gas_rho * gas_v3;

          Real dust_mom1 = dust_rho * dust_v1;
          Real dust_mom2 = dust_rho * dust_v2;
          Real dust_mom3 = dust_rho * dust_v3;

          force_x1(index) = epsilon * alpha * gas_mom1 - alpha * dust_mom1;
          force_x2(index) = epsilon * alpha * gas_mom2 - alpha * dust_mom2;
          force_x3(index) = epsilon * alpha * gas_mom3 - alpha * dust_mom3;
        }

        for (int index = 1; index <= NDUSTFLUIDS; ++index) {
          force_x1(0) -= force_x1(index);
          force_x2(0) -= force_x2(index);
          force_x3(0) -= force_x3(index);
        }

        for (int n=0; n<=NDUSTFLUIDS; ++n) {
          delta_m1_explicit(n) = force_x1(n) * dt;
          delta_m2_explicit(n) = force_x2(n) * dt;
          delta_m3_explicit(n) = force_x3(n) * dt;
        }

        // Alias the conserves of gas
        Real &gas_m1 = u(IM1, k, j, i);
        Real &gas_m2 = u(IM2, k, j, i);
        Real &gas_m3 = u(IM3, k, j, i);

        // Add the delta momentum caused by drags on the gas conserves
        gas_m1 += delta_m1_explicit(0);
        gas_m2 += delta_m2_explicit(0);
        gas_m3 += delta_m3_explicit(0);

        // Update the energy of gas if the gas is non barotropic.
        if (NON_BAROTROPIC_EOS) {
          Real &gas_e     = u(IEN, k, j, i);
          Real delta_erg  = delta_m1_explicit(0)*gas_v1 + delta_m2_explicit(0)*gas_v2 + delta_m3_explicit(0)*gas_v3;
          gas_e          += delta_erg;
        }

        for (int n = 1; n <= NDUSTFLUIDS; ++n) {
          int dust_id = n - 1;
          int rho_id  = 4 * dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;

          // Alias the conserves of dust
          Real &dust_m1 = cons_df(v1_id, k, j, i);
          Real &dust_m2 = cons_df(v2_id, k, j, i);
          Real &dust_m3 = cons_df(v3_id, k, j, i);

          // Add the delta momentum caused by drags on the dust conserves
          dust_m1 += delta_m1_explicit(n);
          dust_m2 += delta_m2_explicit(n);
          dust_m3 += delta_m3_explicit(n);
        }

      }
    }
  }
  return;
}


void DustGasDrag::ExplicitNoFeedback(const int stage,
      const Real dt, const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &u, AthenaArray<Real> &cons_df) {

  MeshBlock  *pmb = pmy_dustfluids_->pmy_block;
  DustFluids *pdf = pmy_dustfluids_;
  Hydro      *ph  = pmb->phydro;

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  AthenaArray<Real> force_x1(num_species);
  AthenaArray<Real> force_x2(num_species);
  AthenaArray<Real> force_x3(num_species);

  AthenaArray<Real> delta_m1_explicit(num_species);
  AthenaArray<Real> delta_m2_explicit(num_species);
  AthenaArray<Real> delta_m3_explicit(num_species);

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        force_x1.ZeroClear();
        force_x2.ZeroClear();
        force_x3.ZeroClear();

        delta_m1_explicit.ZeroClear();
        delta_m2_explicit.ZeroClear();
        delta_m3_explicit.ZeroClear();

        // Alias the primitives of gas
        const Real &gas_rho  = w(IDN, k, j, i);
        const Real &gas_v1   = w(IVX, k, j, i);
        const Real &gas_v2   = w(IVY, k, j, i);
        const Real &gas_v3   = w(IVZ, k, j, i);

        // Set the drag force
        for (int index=1; index<=NDUSTFLUIDS; ++index) {
          int dust_id          = index - 1;
          int rho_id           = 4 * dust_id;
          int v1_id            = rho_id + 1;
          int v2_id            = rho_id + 2;
          int v3_id            = rho_id + 3;
          const Real &dust_rho = prim_df(rho_id, k, j, i);
          Real alpha           = 1.0/(stopping_time(dust_id, k, j, i) + TINY_NUMBER);
          Real epsilon         = dust_rho/gas_rho;

          const Real &dust_v1 = prim_df(v1_id, k, j, i);
          const Real &dust_v2 = prim_df(v2_id, k, j, i);
          const Real &dust_v3 = prim_df(v3_id, k, j, i);

          Real gas_mom1 = gas_rho * gas_v1;
          Real gas_mom2 = gas_rho * gas_v2;
          Real gas_mom3 = gas_rho * gas_v3;

          Real dust_mom1 = dust_rho * dust_v1;
          Real dust_mom2 = dust_rho * dust_v2;
          Real dust_mom3 = dust_rho * dust_v3;

          force_x1(index) = epsilon * alpha * gas_mom1 - alpha * dust_mom1;
          force_x2(index) = epsilon * alpha * gas_mom2 - alpha * dust_mom2;
          force_x3(index) = epsilon * alpha * gas_mom3 - alpha * dust_mom3;
        }

        for (int n=1; n<=NDUSTFLUIDS; ++n) {
          delta_m1_explicit(n) = force_x1(n) * dt;
          delta_m2_explicit(n) = force_x2(n) * dt;
          delta_m3_explicit(n) = force_x3(n) * dt;
        }

        for (int n=1; n<=NDUSTFLUIDS; ++n) {
          int dust_id = n - 1;
          int rho_id  = 4 * dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;

          // Alias the conserves of dust
          Real &dust_m1 = cons_df(v1_id, k, j, i);
          Real &dust_m2 = cons_df(v2_id, k, j, i);
          Real &dust_m3 = cons_df(v3_id, k, j, i);

          // Add the delta momentum caused by drags on the dust conserves
          dust_m1 += delta_m1_explicit(n);
          dust_m2 += delta_m2_explicit(n);
          dust_m3 += delta_m3_explicit(n);
        }

      }
    }
  }
  return;
}
