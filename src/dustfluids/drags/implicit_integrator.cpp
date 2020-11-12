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

void DustGasDrag::ImplicitFeedback(const int stage, const Real dt,
    const AthenaArray<Real> &stopping_time,
    const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
    AthenaArray<Real> &u, AthenaArray<Real> &cons_df)
{
  MeshBlock *pmb = pmy_dustfluids_->pmy_block;

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  AthenaArray<Real> old_v1_vector(num_species);
  AthenaArray<Real> old_v2_vector(num_species);
  AthenaArray<Real> old_v3_vector(num_species);

  AthenaArray<Real> new_v1_vector(num_species);
  AthenaArray<Real> new_v2_vector(num_species);
  AthenaArray<Real> new_v3_vector(num_species);

  AthenaArray<Real> delta_v1_implicit(num_species);
  AthenaArray<Real> delta_v2_implicit(num_species);
  AthenaArray<Real> delta_v3_implicit(num_species);

  AthenaArray<Real> jacobi_matrix(num_species, num_species);

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        const Real &gas_rho = w(IDN, k, j, i);
        const Real &gas_v1  = w(IVX, k, j, i);
        const Real &gas_v2  = w(IVY, k, j, i);
        const Real &gas_v3  = w(IVZ, k, j, i);

        drags_matrix.ZeroClear();
        jacobi_matrix.ZeroClear();

        old_v1_vector.ZeroClear();
        old_v2_vector.ZeroClear();
        old_v3_vector.ZeroClear();

        new_v1_vector.ZeroClear();
        new_v2_vector.ZeroClear();
        new_v3_vector.ZeroClear();

        delta_v1_implicit.ZeroClear();
        delta_v2_implicit.ZeroClear();
        delta_v3_implicit.ZeroClear();

        //Set the jacobi_matrix(0, col), except jacobi_matrix(0, 0)
        for (int col = 1; col <= NDUSTFLUIDS; ++col) {
          int dust_id           = col - 1;
          int rho_id            = 4 * dust_id;
          const Real &dust_rho  = prim_df(rho_id, k, j, i);
          jacobi_matrix(0, col) = dust_rho/gas_rho * 1.0/(stopping_time(dust_id, k, j, i) + TINY_NUMBER);
        }

        //Set the jacobi_matrix(row, 0), except jacobi_matrix(0, 0)
        for (int row = 1; row <= NDUSTFLUIDS; ++row) {
          int dust_id           = row - 1;
          jacobi_matrix(row, 0) = 1.0/(stopping_time(dust_id, k, j, i) + TINY_NUMBER);
        }

        //Set the jacobi_matrix(0, 0)
        for (int dust_id = 0; dust_id < NDUSTFLUIDS; ++dust_id) {
          int rho_id            = 4 * dust_id;
          const Real &dust_rho  = prim_df(rho_id, k, j, i);
          jacobi_matrix(0, 0)  -= dust_rho/gas_rho * 1.0/(stopping_time(dust_id, k, j, i) + TINY_NUMBER);
        }

        //Set the other pivots, except jacobi_matrix(0, 0)
        for (int pivot = 1; pivot <= NDUSTFLUIDS; ++pivot) {
          int dust_id                 = pivot - 1;
          jacobi_matrix(pivot, pivot) = -1.0/(stopping_time(dust_id, k, j, i) + TINY_NUMBER);
        }

        //calculate lambda_matrix = I - h * jacobi_matrix
        Multiplication(dt, jacobi_matrix, drags_matrix);
        Addition(1.0, -1.0, drags_matrix);

        // LU decomposition of drags_matrix, save the LU component as lu_matrix
        LUdecompose(drags_matrix);

        old_v1_vector(0) = gas_v1;
        old_v2_vector(0) = gas_v2;
        old_v3_vector(0) = gas_v3;

        for (int n=1; n<=NDUSTFLUIDS; ++n) {
          int dust_id = n-1;
          int rho_id  = 4 * dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;

          old_v1_vector(n) = prim_df(v1_id, k, j, i);
          old_v2_vector(n) = prim_df(v2_id, k, j, i);
          old_v3_vector(n) = prim_df(v3_id, k, j, i);
        }

        SolveLinearEquation(old_v1_vector, new_v1_vector); // b:v^n, x:v^(n+1), along the x1 direction
        SolveLinearEquation(old_v2_vector, new_v2_vector); // b:v^n, x:v^(n+1), along the x2 direction
        SolveLinearEquation(old_v3_vector, new_v3_vector); // b:v^n, x:v^(n+1), along the x3 direction

        Addition(new_v1_vector, -1.0, old_v1_vector, delta_v1_implicit);
        Addition(new_v2_vector, -1.0, old_v2_vector, delta_v2_implicit);
        Addition(new_v3_vector, -1.0, old_v3_vector, delta_v3_implicit);

        // Alias the conserves of gas
        Real &gas_m1 = u(IM1, k, j, i);
        Real &gas_m2 = u(IM2, k, j, i);
        Real &gas_m3 = u(IM3, k, j, i);

        Real delta_gas_m1 = gas_rho * delta_v1_implicit(0);
        Real delta_gas_m2 = gas_rho * delta_v2_implicit(0);
        Real delta_gas_m3 = gas_rho * delta_v3_implicit(0);

        gas_m1 += delta_gas_m1;
        gas_m2 += delta_gas_m2;
        gas_m3 += delta_gas_m3;

        // Update the energy of gas if the gas is non barotropic.
        if (NON_BAROTROPIC_EOS) {
          Real &gas_e     = u(IEN, k, j, i);
          Real delta_erg  = delta_gas_m1 * gas_v1 + delta_gas_m2 * gas_v2 + delta_gas_m3 * gas_v3;
          gas_e          += delta_erg;
        }

        for (int n=1; n<=NDUSTFLUIDS; ++n){
          int dust_id = n-1;
          int rho_id  = 4 * dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;

          // Alias the parameters of dust
          const Real &dust_rho = prim_df(rho_id, k, j, i);

          Real &dust_m1 = cons_df(v1_id, k, j, i);
          Real &dust_m2 = cons_df(v2_id, k, j, i);
          Real &dust_m3 = cons_df(v3_id, k, j, i);

          Real delta_dust_m1 = dust_rho * delta_v1_implicit(n);
          Real delta_dust_m2 = dust_rho * delta_v2_implicit(n);
          Real delta_dust_m3 = dust_rho * delta_v3_implicit(n);

          dust_m1 += delta_dust_m1;
          dust_m2 += delta_dust_m2;
          dust_m3 += delta_dust_m3;
        }

      }
    }
  }
  return;
}


void DustGasDrag::ImplicitNoFeedback(const int stage, const Real dt,
    const AthenaArray<Real> &stopping_time,
    const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
    const AthenaArray<Real> &u, AthenaArray<Real> &cons_df)
{
  MeshBlock *pmb = pmy_dustfluids_->pmy_block;

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  AthenaArray<Real> old_v1_vector(num_species);
  AthenaArray<Real> old_v2_vector(num_species);
  AthenaArray<Real> old_v3_vector(num_species);

  AthenaArray<Real> new_v1_vector(num_species);
  AthenaArray<Real> new_v2_vector(num_species);
  AthenaArray<Real> new_v3_vector(num_species);

  AthenaArray<Real> delta_v1_implicit(num_species);
  AthenaArray<Real> delta_v2_implicit(num_species);
  AthenaArray<Real> delta_v3_implicit(num_species);

  AthenaArray<Real> jacobi_matrix(num_species, num_species);

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        const Real &gas_rho = w(IDN, k, j, i);
        const Real &gas_v1  = w(IVX, k, j, i);
        const Real &gas_v2  = w(IVY, k, j, i);
        const Real &gas_v3  = w(IVZ, k, j, i);

        drags_matrix.ZeroClear();
        jacobi_matrix.ZeroClear();

        old_v1_vector.ZeroClear();
        old_v2_vector.ZeroClear();
        old_v3_vector.ZeroClear();

        new_v1_vector.ZeroClear();
        new_v2_vector.ZeroClear();
        new_v3_vector.ZeroClear();

        delta_v1_implicit.ZeroClear();
        delta_v2_implicit.ZeroClear();
        delta_v3_implicit.ZeroClear();

        //Set the jacobi_matrix(row, 0), except jacobi_matrix(0, 0)
        for (int row = 1; row <= NDUSTFLUIDS; ++row) {
          int dust_id           = row - 1;
          jacobi_matrix(row, 0) = 1.0/(stopping_time(dust_id, k, j, i) + TINY_NUMBER);
        }

        //Set the other pivots, except jacobi_matrix(0, 0)
        for (int pivot = 1; pivot <= NDUSTFLUIDS; ++pivot) {
          int dust_id                 = pivot - 1;
          jacobi_matrix(pivot, pivot) = -1.0/(stopping_time(dust_id, k, j, i) + TINY_NUMBER);
        }

        //calculate lambda_matrix = I - h * jacobi_matrix
        Multiplication(dt, jacobi_matrix, drags_matrix);
        Addition(1.0, -1.0, drags_matrix);

        // LU decomposition of drags_matrix, save the LU component as lu_matrix
        LUdecompose(drags_matrix);

        old_v1_vector(0) = gas_v1;
        old_v2_vector(0) = gas_v2;
        old_v3_vector(0) = gas_v3;

        for (int n=1; n<=NDUSTFLUIDS; ++n) {
          int dust_id = n-1;
          int rho_id  = 4 * dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;

          old_v1_vector(n) = prim_df(v1_id, k, j, i);
          old_v2_vector(n) = prim_df(v2_id, k, j, i);
          old_v3_vector(n) = prim_df(v3_id, k, j, i);
        }

        SolveLinearEquation(old_v1_vector, new_v1_vector); // b:v^n, x:v^(n+1), along the x1 direction
        SolveLinearEquation(old_v2_vector, new_v2_vector); // b:v^n, x:v^(n+1), along the x2 direction
        SolveLinearEquation(old_v3_vector, new_v3_vector); // b:v^n, x:v^(n+1), along the x3 direction

        Addition(new_v1_vector, -1.0, old_v1_vector, delta_v1_implicit);
        Addition(new_v2_vector, -1.0, old_v2_vector, delta_v2_implicit);
        Addition(new_v3_vector, -1.0, old_v3_vector, delta_v3_implicit);

        for (int n=1; n<=NDUSTFLUIDS; ++n){
          int dust_id = n-1;
          int rho_id  = 4 * dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;

          // Alias the parameters of dust
          const Real &dust_rho = prim_df(rho_id, k, j, i);

          Real &dust_m1 = cons_df(v1_id, k, j, i);
          Real &dust_m2 = cons_df(v2_id, k, j, i);
          Real &dust_m3 = cons_df(v3_id, k, j, i);

          Real delta_dust_m1 = dust_rho * delta_v1_implicit(n);
          Real delta_dust_m2 = dust_rho * delta_v2_implicit(n);
          Real delta_dust_m3 = dust_rho * delta_v3_implicit(n);

          dust_m1 += delta_dust_m1;
          dust_m2 += delta_dust_m2;
          dust_m3 += delta_dust_m3;
        }

      }
    }
  }
  return;
}
