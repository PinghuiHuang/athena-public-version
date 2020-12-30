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


void DustGasDrag::BackwardEulerFeedback(const int stage,
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

  AthenaArray<Real> jacobi_matrix(num_species,     num_species);
  AthenaArray<Real> lambda_matrix(num_species,     num_species);
  AthenaArray<Real> lambda_inv_matrix(num_species, num_species);

  AthenaArray<Real> delta_m1(num_species);
  AthenaArray<Real> delta_m2(num_species);
  AthenaArray<Real> delta_m3(num_species);

  Real inv_dt   = 1.0/dt;
  Real wghts[3] = {0.0, 1.0, -1.0};
  AthenaArray<Real> u_d(NHYDRO,             pmb->ncells3, pmb->ncells2, pmb->ncells1);
  AthenaArray<Real> cons_df_d(num_dust_var, pmb->ncells3, pmb->ncells2, pmb->ncells1);
  pmb->WeightedAve(u_d,       ph->u_as,        ph->u_bs,        wghts);
  pmb->WeightedAve(cons_df_d, pdf->df_cons_as, pdf->df_cons_bs, wghts);

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        force_x1.ZeroClear();
        force_x2.ZeroClear();
        force_x3.ZeroClear();

        jacobi_matrix.ZeroClear();
        lambda_matrix.ZeroClear();
        lambda_inv_matrix.ZeroClear();

        delta_m1.ZeroClear();
        delta_m2.ZeroClear();
        delta_m3.ZeroClear();

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

        //// Add the delta momentum caused by the other explicit source terms, \Delta Gm = (u^(as) - u^(bs))/dt
        //for (int index = 0; index <= NDUSTFLUIDS; index++) {
          //if (index == 0) {
            //force_x1(0) += u_d(IM1, k, j, i) * inv_dt;
            //force_x2(0) += u_d(IM2, k, j, i) * inv_dt;
            //force_x3(0) += u_d(IM3, k, j, i) * inv_dt;
            ////std::cout << "u_d(IM1, k, j, i) is " << u_d(IM1, k, j, i) << std::endl;
            ////std::cout << "u_d(IM2, k, j, i) is " << u_d(IM2, k, j, i) << std::endl;
          //}
          //else {
            //int dust_id = index - 1;
            //int rho_id  = 4*dust_id;
            //int v1_id   = rho_id + 1;
            //int v2_id   = rho_id + 2;
            //int v3_id   = rho_id + 3;

            //force_x1(index) += cons_df_d(v1_id, k, j, i) * inv_dt;
            //force_x2(index) += cons_df_d(v2_id, k, j, i) * inv_dt;
            //force_x3(index) += cons_df_d(v3_id, k, j, i) * inv_dt;
            ////std::cout << "cons_df_d(v1_id, k, j, i) is " << cons_df_d(v1_id, k, j, i) << std::endl;
            ////std::cout << "cons_df_d(v2_id, k, j, i) is " << cons_df_d(v2_id, k, j, i) << std::endl;
          //}
        //}

        // Calculate the jacobi matrix of the drag forces, df/dM
        // Set the jacobi_matrix_n(0, row), except jacobi_matrix_n(0, 0)
        for (int row = 1; row<=NDUSTFLUIDS; ++row) {
          int dust_id           = row - 1;
          int rho_id            = 4*dust_id;
          const Real &dust_rho  = prim_df(rho_id, k, j, i);
          jacobi_matrix(0, row) = dust_rho/gas_rho * 1.0/(stopping_time(dust_id,k,j,i) + TINY_NUMBER);
        }

        // Set the jacobi_matrix(col, 0), except jacobi_matrix(0, 0)
        for (int col = 1; col<=NDUSTFLUIDS; ++col) {
          int dust_id           = col - 1;
          jacobi_matrix(col, 0) = 1.0/(stopping_time(dust_id,k,j,i) + TINY_NUMBER);
        }

        // Set the jacobi_matrix(0,0)
        for (int dust_id = 0; dust_id < NDUSTFLUIDS; ++dust_id) {
          int row              = dust_id + 1;
          jacobi_matrix(0, 0) -= jacobi_matrix(0, row);
        }

        // Set the other pivots, except jacobi_matrix(0, 0)
        for (int pivot = 1; pivot <= NDUSTFLUIDS; ++pivot) {
          int col                     = pivot;
          jacobi_matrix(pivot, pivot) = -1.0*jacobi_matrix(col, 0);
        }

        // calculate lambda_matrix = I - h*jacobi_matrix
        Multiplication(dt, jacobi_matrix, lambda_matrix);
        Addition(1.0, -1.0, lambda_matrix);

        // cauculate the inverse matrix of lambda_matrix
        LUdecompose(lambda_matrix);
        Inverse(lambda_matrix, lambda_inv_matrix);

        // Delta_M = h * (lambda_matrix)^(-1) * f(M)
        Multiplication(dt, lambda_inv_matrix);
        Multiplication(lambda_inv_matrix, force_x1, delta_m1);
        Multiplication(lambda_inv_matrix, force_x2, delta_m2);
        Multiplication(lambda_inv_matrix, force_x3, delta_m3);

        // Alias the conserves of gas
        Real &gas_m1 = u(IM1, k, j, i);
        Real &gas_m2 = u(IM2, k, j, i);
        Real &gas_m3 = u(IM3, k, j, i);

        gas_m1 += delta_m1(0);
        gas_m2 += delta_m2(0);
        gas_m3 += delta_m3(0);

        // Update the energy of gas if the gas is non barotropic. dE = dM * v
        if (NON_BAROTROPIC_EOS) {
          Real &gas_erg   = u(IEN, k, j, i);
          Real work_drag  = delta_m1(0)*gas_v1 + delta_m2(0)*gas_v2 + delta_m3(0)*gas_v3;
          gas_erg        += work_drag;
        }

        for (int n = 1; n <= NDUSTFLUIDS; ++n) {
          int dust_id = n - 1;
          int rho_id  = 4*dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;

          // Alias the conserves of dust
          Real &dust_m1 = cons_df(v1_id, k, j, i);
          Real &dust_m2 = cons_df(v2_id, k, j, i);
          Real &dust_m3 = cons_df(v3_id, k, j, i);

          // Add the delta momentum caused by drags on the dust conserves
          dust_m1 += delta_m1(n);
          dust_m2 += delta_m2(n);
          dust_m3 += delta_m3(n);
        }

        //Real total_m1_implicit = delta_m1(0) + delta_m1(1);
        //Real total_m2_implicit = delta_m2(0) + delta_m2(1);
        //Real total_implicit = std::abs(total_m1_implicit) + std::abs(total_m2_implicit);
        //if (total_m1_implicit != 0.0)
          //std::cout << "Total_m1_implicit is " << total_m1_implicit << std::endl;
        //if (total_m2_implicit != 0.0)
          //std::cout << "Total_m2_implicit is " << total_m2_implicit << std::endl;
        //if (total_implicit > 1e-16)
          //std::cout << "Total_implicit is " << total_implicit << std::endl;

      }
    }
  }
  return;
}


void DustGasDrag::BackwardEulerNoFeedback(const int stage,
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

  AthenaArray<Real> jacobi_matrix(num_species,  num_species);
  AthenaArray<Real> lambda_matrix(num_species, num_species);
  AthenaArray<Real> lambda_inv_matrix(num_species, num_species);

  AthenaArray<Real> delta_m1(num_species);
  AthenaArray<Real> delta_m2(num_species);
  AthenaArray<Real> delta_m3(num_species);

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        force_x1.ZeroClear();
        force_x2.ZeroClear();
        force_x3.ZeroClear();

        jacobi_matrix.ZeroClear();
        lambda_matrix.ZeroClear();
        lambda_inv_matrix.ZeroClear();

        delta_m1.ZeroClear();
        delta_m2.ZeroClear();
        delta_m3.ZeroClear();

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

        // Calculate the jacobi matrix of the drag forces, df/dM
        // Set the jacobi_matrix(col, 0), except jacobi_matrix(0, 0)
        for (int col = 1; col<=NDUSTFLUIDS; ++col) {
          int dust_id           = col - 1;
          jacobi_matrix(col, 0) = 1.0/(stopping_time(dust_id,k,j,i) + TINY_NUMBER);
        }

        // Set the other pivots, except jacobi_matrix(0, 0)
        for (int pivot = 1; pivot <= NDUSTFLUIDS; ++pivot) {
          int col                     = pivot;
          jacobi_matrix(pivot, pivot) = -1.0*jacobi_matrix(col, 0);
        }

        // calculate lambda_matrix = I - h*jacobi_matrix
        Multiplication(dt, jacobi_matrix, lambda_matrix);
        Addition(1.0, -1.0, lambda_matrix);

        // cauculate the inverse matrix of lambda_matrix
        LUdecompose(lambda_matrix);
        Inverse(lambda_matrix, lambda_inv_matrix);

        // Delta_M = h * (lambda_matrix)^(-1) * f(M)
        Multiplication(dt, lambda_inv_matrix);
        Multiplication(lambda_inv_matrix, force_x1, delta_m1);
        Multiplication(lambda_inv_matrix, force_x2, delta_m2);
        Multiplication(lambda_inv_matrix, force_x3, delta_m3);

        for (int n = 1; n <= NDUSTFLUIDS; ++n) {
          int dust_id = n - 1;
          int rho_id  = 4*dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;

          // Alias the conserves of dust
          Real &dust_m1 = cons_df(v1_id, k, j, i);
          Real &dust_m2 = cons_df(v2_id, k, j, i);
          Real &dust_m3 = cons_df(v3_id, k, j, i);

          // Add the delta momentum caused by drags on the dust conserves
          dust_m1 += delta_m1(n);
          dust_m2 += delta_m2(n);
          dust_m3 += delta_m3(n);
        }

      }
    }
  }
  return;
}
