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


void DustGasDrag::TRBDF2Feedback(const int stage,
      const Real dt, const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      AthenaArray<Real> &u, AthenaArray<Real> &cons_df) {

  MeshBlock  *pmb = pmy_dustfluids_->pmy_block;
  DustFluids *pdf = pmy_dustfluids_;
  Hydro      *ph  = pmb->phydro;

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  //AthenaArray<Real> &u_n       = ph->u_n;
  //AthenaArray<Real> &cons_df_n = pdf->df_cons_n;
  AthenaArray<Real> &w_n       = ph->w_n;
  AthenaArray<Real> &prim_df_n = pdf->df_prim_n;

  if (stage == 1) {
    AthenaArray<Real> force_x1(num_species);
    AthenaArray<Real> force_x2(num_species);
    AthenaArray<Real> force_x3(num_species);

    AthenaArray<Real> delta_m1_explicit(num_species);
    AthenaArray<Real> delta_m2_explicit(num_species);
    AthenaArray<Real> delta_m3_explicit(num_species);

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
    AthenaArray<Real> lambda_matrix(num_species, num_species);
    AthenaArray<Real> lambda_inv_matrix(num_species, num_species);

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

          jacobi_matrix.ZeroClear();
          lambda_matrix.ZeroClear();
          lambda_inv_matrix.ZeroClear();
          drags_matrix.ZeroClear();

          old_v1_vector.ZeroClear();
          old_v2_vector.ZeroClear();
          old_v3_vector.ZeroClear();

          new_v1_vector.ZeroClear();
          new_v2_vector.ZeroClear();
          new_v3_vector.ZeroClear();

          delta_v1_implicit.ZeroClear();
          delta_v2_implicit.ZeroClear();
          delta_v3_implicit.ZeroClear();

          // Alias the primitives of gas
          const Real &gas_rho = w(IDN, k, j, i);
          const Real &gas_v1  = w(IVX, k, j, i);
          const Real &gas_v2  = w(IVY, k, j, i);
          const Real &gas_v3  = w(IVZ, k, j, i);

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

          for (int dust_index=1; dust_index<=NDUSTFLUIDS; ++dust_index) {
            force_x1(0) -= force_x1(dust_index);
            force_x2(0) -= force_x2(dust_index);
            force_x3(0) -= force_x3(dust_index);
          }

          for (int n=0; n<=NDUSTFLUIDS; ++n) {
            delta_m1_explicit(n) = force_x1(n) * dt;
            delta_m2_explicit(n) = force_x2(n) * dt;
            delta_m3_explicit(n) = force_x3(n) * dt;
          }

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
            int col              = dust_id + 1;
            jacobi_matrix(0, 0) -= jacobi_matrix(0, col);
          }

          //Set the other pivots, except jacobi_matrix(0, 0)
          for (int pivot = 1; pivot <= NDUSTFLUIDS; ++pivot) {
            int row                     = pivot;
            jacobi_matrix(pivot, pivot) = -1.0 * jacobi_matrix(row, 0);
          }

          //calculate lambda_matrix = I - h * jacobi_matrix
          Multiplication(dt, jacobi_matrix, drags_matrix);
          Addition(1.0, -1.0, drags_matrix);

          // LU decomposition of drags_matrix, save the LU component as lu_matrix
          LUdecompose(drags_matrix);

          old_v1_vector(0) = w(IVX, k, j, i);
          old_v2_vector(0) = w(IVY, k, j, i);
          old_v3_vector(0) = w(IVZ, k, j, i);

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

          Real delta_gas_m1 = 0.5*(delta_m1_explicit(0) + gas_rho * delta_v1_implicit(0));
          Real delta_gas_m2 = 0.5*(delta_m2_explicit(0) + gas_rho * delta_v2_implicit(0));
          Real delta_gas_m3 = 0.5*(delta_m3_explicit(0) + gas_rho * delta_v3_implicit(0));

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

            Real delta_dust_m1 = 0.5*(delta_m1_explicit(n) + dust_rho * delta_v1_implicit(n));
            Real delta_dust_m2 = 0.5*(delta_m2_explicit(n) + dust_rho * delta_v2_implicit(n));
            Real delta_dust_m3 = 0.5*(delta_m3_explicit(n) + dust_rho * delta_v3_implicit(n));

            dust_m1 += delta_dust_m1;
            dust_m2 += delta_dust_m2;
            dust_m3 += delta_dust_m3;
          }

        }
      }
    }
  }
  else {
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
    AthenaArray<Real> lambda_matrix(num_species, num_species);
    AthenaArray<Real> lambda_inv_matrix(num_species, num_species);

    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          jacobi_matrix.ZeroClear();
          lambda_matrix.ZeroClear();
          lambda_inv_matrix.ZeroClear();
          drags_matrix.ZeroClear();

          old_v1_vector.ZeroClear();
          old_v2_vector.ZeroClear();
          old_v3_vector.ZeroClear();

          new_v1_vector.ZeroClear();
          new_v2_vector.ZeroClear();
          new_v3_vector.ZeroClear();

          delta_v1_implicit.ZeroClear();
          delta_v2_implicit.ZeroClear();
          delta_v3_implicit.ZeroClear();

          const Real &gas_rho_n = w_n(IDN, k, j, i);
          const Real &gas_v1_n  = w_n(IVX, k, j, i);
          const Real &gas_v2_n  = w_n(IVY, k, j, i);
          const Real &gas_v3_n  = w_n(IVZ, k, j, i);

          const Real &gas_rho_p = w(IDN, k, j, i);
          const Real &gas_v1_p  = w(IVX, k, j, i);
          const Real &gas_v2_p  = w(IVY, k, j, i);
          const Real &gas_v3_p  = w(IVZ, k, j, i);

          //Set the jacobi_matrix(0, col), except jacobi_matrix(0, 0)
          for (int col = 1; col <= NDUSTFLUIDS; ++col) {
            int dust_id            = col - 1;
            int rho_id             = 4 * dust_id;
            const Real &dust_rho_n = prim_df_n(rho_id, k, j, i);
            jacobi_matrix(0, col)  = dust_rho_n/gas_rho_n * 1.0/(stopping_time(dust_id, k, j, i) + TINY_NUMBER);
          }

          //Set the jacobi_matrix(row, 0), except jacobi_matrix(0, 0)
          for (int row = 1; row <= NDUSTFLUIDS; ++row) {
            int dust_id           = row - 1;
            jacobi_matrix(row, 0) = 1.0/(stopping_time(dust_id, k, j, i) + TINY_NUMBER);
          }

          //Set the jacobi_matrix(0, 0)
          for (int dust_id = 0; dust_id < NDUSTFLUIDS; ++dust_id) {
            int col              = dust_id + 1;
            jacobi_matrix(0, 0) -= jacobi_matrix(0, col);
          }

          //Set the other pivots, except jacobi_matrix(0, 0)
          for (int pivot = 1; pivot <= NDUSTFLUIDS; ++pivot) {
            int row                     = pivot;
            jacobi_matrix(pivot, pivot) = -1.0 * jacobi_matrix(row, 0);
          }

          //calculate lambda_matrix = I - h * jacobi_matrix
          Multiplication(dt, jacobi_matrix, drags_matrix);
          Addition(1.0, -1.0, drags_matrix);

          // LU decomposition of drags_matrix, save the LU component as lu_matrix
          LUdecompose(drags_matrix);

          old_v1_vector(0) = w_n(IVX, k, j, i);
          old_v2_vector(0) = w_n(IVY, k, j, i);
          old_v3_vector(0) = w_n(IVZ, k, j, i);

          for (int n=1; n<=NDUSTFLUIDS; ++n) {
            int dust_id = n-1;
            int rho_id  = 4 * dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            old_v1_vector(n) = prim_df_n(v1_id, k, j, i);
            old_v2_vector(n) = prim_df_n(v2_id, k, j, i);
            old_v3_vector(n) = prim_df_n(v3_id, k, j, i);
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

          Real delta_gas_m1 = ONE_3RD * gas_rho_n * delta_v1_implicit(0);
          Real delta_gas_m2 = ONE_3RD * gas_rho_n * delta_v2_implicit(0);
          Real delta_gas_m3 = ONE_3RD * gas_rho_n * delta_v3_implicit(0);

          delta_gas_m1 += (2.0*TWO_3RD*(gas_rho_p * gas_v1_p - gas_rho_n * gas_v1_n));
          delta_gas_m2 += (2.0*TWO_3RD*(gas_rho_p * gas_v2_p - gas_rho_n * gas_v2_n));
          delta_gas_m3 += (2.0*TWO_3RD*(gas_rho_p * gas_v3_p - gas_rho_n * gas_v3_n));

          gas_m1 += delta_gas_m1;
          gas_m2 += delta_gas_m2;
          gas_m3 += delta_gas_m3;

          // Update the energy of gas if the gas is non barotropic.
          if (NON_BAROTROPIC_EOS) {
            Real &gas_e     = u(IEN, k, j, i);
            Real delta_erg  = delta_gas_m1 * gas_v1_n + delta_gas_m2 * gas_v2_n + delta_gas_m3 * gas_v3_n;
            gas_e          += delta_erg;
          }

          for (int n=1; n<=NDUSTFLUIDS; ++n){
            int dust_id = n-1;
            int rho_id  = 4 * dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            // Alias the parameters of dust
            Real &dust_m1 = cons_df(v1_id, k, j, i);
            Real &dust_m2 = cons_df(v2_id, k, j, i);
            Real &dust_m3 = cons_df(v3_id, k, j, i);

            const Real &dust_rho_n = prim_df_n(rho_id, k, j, i);
            const Real &dust_v1_n  = prim_df_n(v1_id,  k, j, i);
            const Real &dust_v2_n  = prim_df_n(v2_id,  k, j, i);
            const Real &dust_v3_n  = prim_df_n(v3_id,  k, j, i);

            const Real &dust_rho_p = prim_df(rho_id, k, j, i);
            const Real &dust_v1_p  = prim_df(v1_id,  k, j, i);
            const Real &dust_v2_p  = prim_df(v2_id,  k, j, i);
            const Real &dust_v3_p  = prim_df(v3_id,  k, j, i);

            Real delta_dust_m1 = ONE_3RD * dust_rho_n * delta_v1_implicit(n);
            Real delta_dust_m2 = ONE_3RD * dust_rho_n * delta_v2_implicit(n);
            Real delta_dust_m3 = ONE_3RD * dust_rho_n * delta_v3_implicit(n);

            delta_dust_m1 += (2.0*TWO_3RD*(dust_rho_p * dust_v1_p - dust_rho_n * dust_v1_n));
            delta_dust_m2 += (2.0*TWO_3RD*(dust_rho_p * dust_v2_p - dust_rho_n * dust_v2_n));
            delta_dust_m3 += (2.0*TWO_3RD*(dust_rho_p * dust_v3_p - dust_rho_n * dust_v3_n));

            dust_m1 += delta_dust_m1;
            dust_m2 += delta_dust_m2;
            dust_m3 += delta_dust_m3;
          }

        }
      }
    }
  }
  return;
}

void DustGasDrag::TRBDF2NoFeedback(const int stage,
      const Real dt, const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &u, AthenaArray<Real> &cons_df) {

  MeshBlock  *pmb = pmy_dustfluids_->pmy_block;
  DustFluids *pdf = pmy_dustfluids_;
  Hydro      *ph  = pmb->phydro;

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  //AthenaArray<Real> &u_n       = ph->u_n;
  //AthenaArray<Real> &cons_df_n = pdf->df_cons_n;
  AthenaArray<Real> &w_n       = ph->w_n;
  AthenaArray<Real> &prim_df_n = pdf->df_prim_n;

  if (stage == 1) {
    AthenaArray<Real> force_x1(num_species);
    AthenaArray<Real> force_x2(num_species);
    AthenaArray<Real> force_x3(num_species);

    AthenaArray<Real> delta_m1_explicit(num_species);
    AthenaArray<Real> delta_m2_explicit(num_species);
    AthenaArray<Real> delta_m3_explicit(num_species);

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
    AthenaArray<Real> lambda_matrix(num_species, num_species);
    AthenaArray<Real> lambda_inv_matrix(num_species, num_species);

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

          jacobi_matrix.ZeroClear();
          lambda_matrix.ZeroClear();
          lambda_inv_matrix.ZeroClear();
          drags_matrix.ZeroClear();

          old_v1_vector.ZeroClear();
          old_v2_vector.ZeroClear();
          old_v3_vector.ZeroClear();

          new_v1_vector.ZeroClear();
          new_v2_vector.ZeroClear();
          new_v3_vector.ZeroClear();

          delta_v1_implicit.ZeroClear();
          delta_v2_implicit.ZeroClear();
          delta_v3_implicit.ZeroClear();

          // Alias the primitives of gas
          const Real &gas_rho = w(IDN, k, j, i);
          const Real &gas_v1  = w(IVX, k, j, i);
          const Real &gas_v2  = w(IVY, k, j, i);
          const Real &gas_v3  = w(IVZ, k, j, i);

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

          for (int n=0; n<=NDUSTFLUIDS; ++n) {
            delta_m1_explicit(n) = force_x1(n) * dt;
            delta_m2_explicit(n) = force_x2(n) * dt;
            delta_m3_explicit(n) = force_x3(n) * dt;
          }

          //Set the jacobi_matrix(row, 0), except jacobi_matrix(0, 0)
          for (int row = 1; row <= NDUSTFLUIDS; ++row) {
            int dust_id           = row - 1;
            jacobi_matrix(row, 0) = 1.0/(stopping_time(dust_id, k, j, i) + TINY_NUMBER);
          }

          //Set the other pivots, except jacobi_matrix(0, 0)
          for (int pivot = 1; pivot <= NDUSTFLUIDS; ++pivot) {
            int row                     = pivot;
            jacobi_matrix(pivot, pivot) = -1.0 * jacobi_matrix(row, 0);
          }

          //calculate lambda_matrix = I - h * jacobi_matrix
          Multiplication(dt, jacobi_matrix, drags_matrix);
          Addition(1.0, -1.0, drags_matrix);

          // LU decomposition of drags_matrix, save the LU component as lu_matrix
          LUdecompose(drags_matrix);

          old_v1_vector(0) = w(IVX, k, j, i);
          old_v2_vector(0) = w(IVY, k, j, i);
          old_v3_vector(0) = w(IVZ, k, j, i);

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

            Real delta_dust_m1 = 0.5*(delta_m1_explicit(n) + dust_rho * delta_v1_implicit(n));
            Real delta_dust_m2 = 0.5*(delta_m2_explicit(n) + dust_rho * delta_v2_implicit(n));
            Real delta_dust_m3 = 0.5*(delta_m3_explicit(n) + dust_rho * delta_v3_implicit(n));

            dust_m1 += delta_dust_m1;
            dust_m2 += delta_dust_m2;
            dust_m3 += delta_dust_m3;
          }

        }
      }
    }
  }
  else {
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
    AthenaArray<Real> lambda_matrix(num_species, num_species);
    AthenaArray<Real> lambda_inv_matrix(num_species, num_species);

    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          jacobi_matrix.ZeroClear();
          lambda_matrix.ZeroClear();
          lambda_inv_matrix.ZeroClear();
          drags_matrix.ZeroClear();

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
            int row                     = pivot;
            jacobi_matrix(pivot, pivot) = -1.0 * jacobi_matrix(row, 0);
          }

          //calculate lambda_matrix = I - h * jacobi_matrix
          Multiplication(dt, jacobi_matrix, drags_matrix);
          Addition(1.0, -1.0, drags_matrix);

          // LU decomposition of drags_matrix, save the LU component as lu_matrix
          LUdecompose(drags_matrix);

          old_v1_vector(0) = w_n(IVX, k, j, i);
          old_v2_vector(0) = w_n(IVY, k, j, i);
          old_v3_vector(0) = w_n(IVZ, k, j, i);

          for (int n=1; n<=NDUSTFLUIDS; ++n) {
            int dust_id = n-1;
            int rho_id  = 4 * dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            old_v1_vector(n) = prim_df_n(v1_id, k, j, i);
            old_v2_vector(n) = prim_df_n(v2_id, k, j, i);
            old_v3_vector(n) = prim_df_n(v3_id, k, j, i);
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
            Real &dust_m1 = cons_df(v1_id, k, j, i);
            Real &dust_m2 = cons_df(v2_id, k, j, i);
            Real &dust_m3 = cons_df(v3_id, k, j, i);

            const Real &dust_rho_n = prim_df_n(rho_id, k, j, i);
            const Real &dust_v1_n  = prim_df_n(v1_id,  k, j, i);
            const Real &dust_v2_n  = prim_df_n(v2_id,  k, j, i);
            const Real &dust_v3_n  = prim_df_n(v3_id,  k, j, i);

            const Real &dust_rho_p = prim_df(rho_id, k, j, i);
            const Real &dust_v1_p  = prim_df(v1_id,  k, j, i);
            const Real &dust_v2_p  = prim_df(v2_id,  k, j, i);
            const Real &dust_v3_p  = prim_df(v3_id,  k, j, i);

            Real delta_dust_m1 = ONE_3RD * dust_rho_n * delta_v1_implicit(n);
            Real delta_dust_m2 = ONE_3RD * dust_rho_n * delta_v2_implicit(n);
            Real delta_dust_m3 = ONE_3RD * dust_rho_n * delta_v3_implicit(n);

            delta_dust_m1 += (2.0*TWO_3RD*(dust_rho_p * dust_v1_p - dust_rho_n * dust_v1_n));
            delta_dust_m2 += (2.0*TWO_3RD*(dust_rho_p * dust_v2_p - dust_rho_n * dust_v2_n));
            delta_dust_m3 += (2.0*TWO_3RD*(dust_rho_p * dust_v3_p - dust_rho_n * dust_v3_n));

            dust_m1 += delta_dust_m1;
            dust_m2 += delta_dust_m2;
            dust_m3 += delta_dust_m3;
          }

        }
      }
    }
  }
  return;
}
