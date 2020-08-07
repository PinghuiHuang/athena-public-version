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


void DustGasDrag::SingleDust_NoFeedback_Implicit(MeshBlock *pmb, const Real dt,
      const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &u, AthenaArray<Real> &cons_df) {

  const int num_dust_var = 4*NDUSTFLUIDS;
  const bool f2          = pmb->pmy_mesh->f2;
  const bool f3          = pmb->pmy_mesh->f3;
  Coordinates *pco       = pmb->pcoord;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  //int il, iu, jl, ju, kl, ku;
  //jl = js, ju = je, kl = ks, ku = ke;
  //if (MAGNETIC_FIELDS_ENABLED) {
    //if (f2) {
      //if (!f3)// 2D
        //jl = js - 1, ju = je + 1, kl = ks, ku = ke;
      //else // 3D
        //jl = js - 1, ju = je + 1, kl = ks - 1, ku = ke + 1;
    //}
  //}

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

void DustGasDrag::SingleDust_Feedback_Implicit(MeshBlock *pmb, const Real dt,
      const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      AthenaArray<Real> &u, AthenaArray<Real> &cons_df) {

  const int num_dust_var = 4*NDUSTFLUIDS;
  const bool f2          = pmb->pmy_mesh->f2;
  const bool f3          = pmb->pmy_mesh->f3;
  Coordinates *pco       = pmb->pcoord;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  //int il, iu, jl, ju, kl, ku;
  //jl = js, ju = je, kl = ks, ku = ke;
  //if (MAGNETIC_FIELDS_ENABLED) {
    //if (f2) {
      //if (!f3)// 2D
        //jl = js - 1, ju = je + 1, kl = ks, ku = ke;
      //else // 3D
        //jl = js - 1, ju = je + 1, kl = ks - 1, ku = ke + 1;
    //}
  //}

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


void DustGasDrag::MultipleDust_NoFeedback_Implicit(MeshBlock *pmb, const Real dt,
      const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &u, AthenaArray<Real> &cons_df) {

  const int num_dust_var = 4*NDUSTFLUIDS;
  const bool f2          = pmb->pmy_mesh->f2;
  const bool f3          = pmb->pmy_mesh->f3;
  Coordinates *pco       = pmb->pcoord;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  //int il, iu, jl, ju, kl, ku;
  //jl = js, ju = je, kl = ks, ku = ke;
  //if (MAGNETIC_FIELDS_ENABLED) {
    //if (f2) {
      //if (!f3)// 2D
        //jl = js - 1, ju = je + 1, kl = ks, ku = ke;
      //else // 3D
        //jl = js - 1, ju = je + 1, kl = ks - 1, ku = ke + 1;
    //}
  //}

  drags_matrix.ZeroClear();

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        // Set the drags_matrix(0,0)
        drags_matrix(0,0) = 1.0;

        // Set the drags_matrix(row,0), except drags_matrix(0,0)
        for (int row=1; row<=NDUSTFLUIDS; row++){
          int dust_id = row -1;
          drags_matrix(row,0) = -dt/stopping_time(dust_id,k,j,i);
        }

        // Set the other pivots, except drags_matrix(0,0)
        for (int row=1; row<=NDUSTFLUIDS; row++){
          int dust_id = row - 1;
          drags_matrix(row,row) = 1.0 + dt/stopping_time(dust_id,k,j,i);
        }

        // LU decomposition of drags_matrix, save the LU component as lu_matrix
        LUdecompose(drags_matrix);
        AthenaArray<Real> b1_vector(num_species);
        AthenaArray<Real> b2_vector(num_species);
        AthenaArray<Real> b3_vector(num_species);
        AthenaArray<Real> x1_vector(num_species);
        AthenaArray<Real> x2_vector(num_species);
        AthenaArray<Real> x3_vector(num_species);

        b1_vector(0) = w(IVX,k,j,i);
        b2_vector(0) = w(IVY,k,j,i);
        b3_vector(0) = w(IVZ,k,j,i);

        for (int n=1; n<=NDUSTFLUIDS; n++) {
          int dust_id  = n-1;
          int rho_id   = 4*dust_id;
          int v1_id    = rho_id + 1;
          int v2_id    = rho_id + 2;
          int v3_id    = rho_id + 3;
          b1_vector(n) = prim_df(v1_id,k,j,i);
          b2_vector(n) = prim_df(v2_id,k,j,i);
          b3_vector(n) = prim_df(v3_id,k,j,i);
        }

        SolveLinearEquation(b1_vector, x1_vector); // b:v^n, x:v^(n+1), along the x1 direction
        SolveLinearEquation(b2_vector, x2_vector); // b:v^n, x:v^(n+1), along the x2 direction
        SolveLinearEquation(b3_vector, x3_vector); // b:v^n, x:v^(n+1), along the x3 direction

        for (int n=1; n<=NDUSTFLUIDS; n++){
          int dust_id = n-1;
          int rho_id  = 4*dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;
          // Alias the parameters of dust
          const Real &dust_d = prim_df(rho_id, k, j, i);

          Real &dust_m1 = cons_df(v1_id,  k, j, i);
          Real &dust_m2 = cons_df(v2_id,  k, j, i);
          Real &dust_m3 = cons_df(v3_id,  k, j, i);

          Real dust_v1 = b1_vector(n);
          Real dust_v2 = b2_vector(n);
          Real dust_v3 = b3_vector(n);

          Real dust_v1_new = x1_vector(n);
          Real dust_v2_new = x2_vector(n);
          Real dust_v3_new = x3_vector(n);

          Real delta_dust_m1 = dust_d*(dust_v1_new - dust_v1);
          Real delta_dust_m2 = dust_d*(dust_v2_new - dust_v2);
          Real delta_dust_m3 = dust_d*(dust_v3_new - dust_v3);

          dust_m1 += delta_dust_m1;
          dust_m2 += delta_dust_m2;
          dust_m3 += delta_dust_m3;
        }

      }
    }
  }
  return;
}


void DustGasDrag::MultipleDust_Feedback_Implicit(MeshBlock *pmb, const Real dt,
      const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      AthenaArray<Real> &u, AthenaArray<Real> &cons_df) {

  const int num_dust_var = 4*NDUSTFLUIDS;
  const bool f2          = pmb->pmy_mesh->f2;
  const bool f3          = pmb->pmy_mesh->f3;
  Coordinates *pco       = pmb->pcoord;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  //int il, iu, jl, ju, kl, ku;
  //jl = js, ju = je, kl = ks, ku = ke;
  //if (MAGNETIC_FIELDS_ENABLED) {
    //if (f2) {
      //if (!f3)// 2D
        //jl = js - 1, ju = je + 1, kl = ks, ku = ke;
      //else // 3D
        //jl = js - 1, ju = je + 1, kl = ks - 1, ku = ke + 1;
    //}
  //}

  drags_matrix.ZeroClear();

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        // Set the drags_matrix(0,0)
        drags_matrix(0,0) = 1.0;
        for (int dust_id=0; dust_id<NDUSTFLUIDS; dust_id++){
          int rho_id = 4*dust_id;
          const Real &dust_d = prim_df(rho_id, k, j, i);
          const Real &gas_d  = w(IDN, k, j, i);
          drags_matrix(0,0) += dust_d/gas_d * dt/stopping_time(dust_id,k,j,i);
        }

        // Set the drags_matrix(0,col), except drags_matrix(0,0)
        for (int col=1; col<=NDUSTFLUIDS; col++){
          int dust_id = col -1;
          int rho_id  = 4*dust_id;
          const Real &dust_d = prim_df(rho_id, k, j, i);
          const Real &gas_d  = w(IDN, k, j, i);
          drags_matrix(0,col) = -dust_d/gas_d * dt/stopping_time(dust_id,k,j,i);
        }

        // Set the drags_matrix(row,0), except drags_matrix(0,0)
        for (int row=1; row<=NDUSTFLUIDS; row++){
          int dust_id = row -1;
          drags_matrix(row,0) = -dt/stopping_time(dust_id,k,j,i);
        }

        // Set the other pivots, except drags_matrix(0,0)
        for (int row=1; row<=NDUSTFLUIDS; row++){
          int dust_id = row - 1;
          drags_matrix(row,row) = 1.0 + dt/stopping_time(dust_id,k,j,i);
        }

        // LU decomposition of drags_matrix, save the LU component as lu_matrix
        LUdecompose(drags_matrix);
        AthenaArray<Real> b1_vector(num_species);
        AthenaArray<Real> b2_vector(num_species);
        AthenaArray<Real> b3_vector(num_species);
        AthenaArray<Real> x1_vector(num_species);
        AthenaArray<Real> x2_vector(num_species);
        AthenaArray<Real> x3_vector(num_species);

        b1_vector(0) = w(IVX,k,j,i);
        b2_vector(0) = w(IVY,k,j,i);
        b3_vector(0) = w(IVZ,k,j,i);

        for (int n=1; n<=NDUSTFLUIDS; n++) {
          int dust_id  = n-1;
          int rho_id   = 4*dust_id;
          int v1_id    = rho_id + 1;
          int v2_id    = rho_id + 2;
          int v3_id    = rho_id + 3;
          b1_vector(n) = prim_df(v1_id,k,j,i);
          b2_vector(n) = prim_df(v2_id,k,j,i);
          b3_vector(n) = prim_df(v3_id,k,j,i);
        }

        SolveLinearEquation(b1_vector, x1_vector); // b:v^n, x:v^(n+1), along the x1 direction
        SolveLinearEquation(b2_vector, x2_vector); // b:v^n, x:v^(n+1), along the x2 direction
        SolveLinearEquation(b3_vector, x3_vector); // b:v^n, x:v^(n+1), along the x3 direction

        // Alias the parameters of gas
        const Real &gas_p = w(IPR, k, j, i);
        const Real &gas_d = w(IDN, k, j, i);
        Real &gas_m1      = u(IM1, k, j, i);
        Real &gas_m2      = u(IM2, k, j, i);
        Real &gas_m3      = u(IM3, k, j, i);
        Real &gas_e       = u(IEN, k, j, i);

        Real gas_v1 = b1_vector(0);
        Real gas_v2 = b2_vector(0);
        Real gas_v3 = b3_vector(0);

        Real gas_v1_new = x1_vector(0);
        Real gas_v2_new = x2_vector(0);
        Real gas_v3_new = x3_vector(0);

        Real delta_gas_m1 = gas_d*(gas_v1_new - gas_v1);
        Real delta_gas_m2 = gas_d*(gas_v2_new - gas_v2);
        Real delta_gas_m3 = gas_d*(gas_v3_new - gas_v3);

        gas_m1 += delta_gas_m1;
        gas_m2 += delta_gas_m2;
        gas_m3 += delta_gas_m3;

        // Update the energy of gas if the gas is non barotropic.
        if (NON_BAROTROPIC_EOS)
          gas_e += delta_gas_m1*gas_v1 + delta_gas_m2*gas_v2 + delta_gas_m3*gas_v3;

        for (int n=1; n<=NDUSTFLUIDS; n++){
          int dust_id = n-1;
          int rho_id  = 4*dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;
          // Alias the parameters of dust
          const Real &dust_d = prim_df(rho_id, k, j, i);

          Real &dust_m1 = cons_df(v1_id, k, j, i);
          Real &dust_m2 = cons_df(v2_id, k, j, i);
          Real &dust_m3 = cons_df(v3_id, k, j, i);

          Real dust_v1 = b1_vector(n);
          Real dust_v2 = b2_vector(n);
          Real dust_v3 = b3_vector(n);

          Real dust_v1_new = x1_vector(n);
          Real dust_v2_new = x2_vector(n);
          Real dust_v3_new = x3_vector(n);

          Real delta_dust_m1 = dust_d*(dust_v1_new - dust_v1);
          Real delta_dust_m2 = dust_d*(dust_v2_new - dust_v2);
          Real delta_dust_m3 = dust_d*(dust_v3_new - dust_v3);

          dust_m1 += delta_dust_m1;
          dust_m2 += delta_dust_m2;
          dust_m3 += delta_dust_m3;
        }

      }
    }
  }
  return;
}

void DustGasDrag::SingleDust_NoFeedback_SemiImplicit(MeshBlock *pmb, const Real dt,
      const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &u, AthenaArray<Real> &cons_df) {

  const int num_dust_var = 4*NDUSTFLUIDS;
  const bool f2          = pmb->pmy_mesh->f2;
  const bool f3          = pmb->pmy_mesh->f3;
  Coordinates *pco       = pmb->pcoord;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  //int il, iu, jl, ju, kl, ku;
  //jl = js, ju = je, kl = ks, ku = ke;
  //if (MAGNETIC_FIELDS_ENABLED) {
    //if (f2) {
      //if (!f3)// 2D
        //jl = js - 1, ju = je + 1, kl = ks, ku = ke;
      //else // 3D
        //jl = js - 1, ju = je + 1, kl = ks - 1, ku = ke + 1;
    //}
  //}

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
        Real deter   = 2.0 + alpha_dg*dt + alpha_gd*dt;
        Real tempA_d = 2.0 * alpha_dg*dt;
        Real tempB_d = 2.0 + alpha_gd*dt - alpha_dg*dt;

        Real dust_v1_new = (tempA_d * gas_v1 + tempB_d * dust_v1)/deter;
        Real dust_v2_new = (tempA_d * gas_v2 + tempB_d * dust_v2)/deter;
        Real dust_v3_new = (tempA_d * gas_v3 + tempB_d * dust_v3)/deter;

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

void DustGasDrag::SingleDust_Feedback_SemiImplicit(MeshBlock *pmb, const Real dt,
      const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      AthenaArray<Real> &u, AthenaArray<Real> &cons_df) {

  const int num_dust_var = 4*NDUSTFLUIDS;
  const bool f2          = pmb->pmy_mesh->f2;
  const bool f3          = pmb->pmy_mesh->f3;
  Coordinates *pco       = pmb->pcoord;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  //int il, iu, jl, ju, kl, ku;
  //jl = js, ju = je, kl = ks, ku = ke;
  //if (MAGNETIC_FIELDS_ENABLED) {
    //if (f2) {
      //if (!f3)// 2D
        //jl = js - 1, ju = je + 1, kl = ks, ku = ke;
      //else // 3D
        //jl = js - 1, ju = je + 1, kl = ks - 1, ku = ke + 1;
    //}
  //}

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
        Real deter   = 2.0 + alpha_dg*dt + alpha_gd*dt;

        Real tempA_g = 2.0 * alpha_gd*dt;
        Real tempB_g = 2.0 + alpha_dg*dt - alpha_gd*dt;

        Real tempA_d = 2.0 * alpha_dg*dt;
        Real tempB_d = 2.0 + alpha_gd*dt - alpha_dg*dt;

        Real dust_v1_new = (tempA_d * gas_v1  + tempB_d * dust_v1)/deter;
        Real dust_v2_new = (tempA_d * gas_v2  + tempB_d * dust_v2)/deter;
        Real dust_v3_new = (tempA_d * gas_v3  + tempB_d * dust_v3)/deter;

        Real delta_dust_m1 = dust_d*(dust_v1_new - dust_v1);
        Real delta_dust_m2 = dust_d*(dust_v2_new - dust_v2);
        Real delta_dust_m3 = dust_d*(dust_v3_new - dust_v3);

        dust_m1 += delta_dust_m1;
        dust_m2 += delta_dust_m2;
        dust_m3 += delta_dust_m3;

        Real gas_v1_new = (tempA_g  * dust_v1 + tempB_g * gas_v1)/deter;
        Real gas_v2_new = (tempA_g  * dust_v2 + tempB_g * gas_v2)/deter;
        Real gas_v3_new = (tempA_g  * dust_v3 + tempB_g * gas_v3)/deter;

        Real delta_gas_m1 = gas_d*(gas_v1_new - gas_v1);
        Real delta_gas_m2 = gas_d*(gas_v2_new - gas_v2);
        Real delta_gas_m3 = gas_d*(gas_v3_new - gas_v3);

        gas_m1 += delta_gas_m1;
        gas_m2 += delta_gas_m2;
        gas_m3 += delta_gas_m3;

        // Update the energy of gas if the gas is non barotropic.
        if (NON_BAROTROPIC_EOS)
          gas_e += delta_gas_m1*gas_v1 + delta_gas_m2*gas_v2 + delta_gas_m3*gas_v3;

        }
      }
    }
  return;
}

void DustGasDrag::SingleDust_NoFeedback_Explicit(MeshBlock *pmb, const Real dt,
      const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &u, AthenaArray<Real> &cons_df) {

  const int num_dust_var = 4*NDUSTFLUIDS;
  const bool f2          = pmb->pmy_mesh->f2;
  const bool f3          = pmb->pmy_mesh->f3;
  Coordinates *pco       = pmb->pcoord;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  //int il, iu, jl, ju, kl, ku;
  //jl = js, ju = je, kl = ks, ku = ke;
  //if (MAGNETIC_FIELDS_ENABLED) {
    //if (f2) {
      //if (!f3)// 2D
        //jl = js - 1, ju = je + 1, kl = ks, ku = ke;
      //else // 3D
        //jl = js - 1, ju = je + 1, kl = ks - 1, ku = ke + 1;
    //}
  //}

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

        Real dust_v1_new = (gas_v1 * alpha_dg *dt + dust_v1 * (1.0 - alpha_dg *dt));
        Real dust_v2_new = (gas_v2 * alpha_dg *dt + dust_v2 * (1.0 - alpha_dg *dt));
        Real dust_v3_new = (gas_v3 * alpha_dg *dt + dust_v3 * (1.0 - alpha_dg *dt));

        // Update the Momentum of gas and dust
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

void DustGasDrag::SingleDust_Feedback_Explicit(MeshBlock *pmb, const Real dt,
      const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      AthenaArray<Real> &u, AthenaArray<Real> &cons_df) {

  const int num_dust_var = 4*NDUSTFLUIDS;
  const bool f2          = pmb->pmy_mesh->f2;
  const bool f3          = pmb->pmy_mesh->f3;
  Coordinates *pco       = pmb->pcoord;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  //int il, iu, jl, ju, kl, ku;
  //jl = js, ju = je, kl = ks, ku = ke;
  //if (MAGNETIC_FIELDS_ENABLED) {
    //if (f2) {
      //if (!f3)// 2D
        //jl = js - 1, ju = je + 1, kl = ks, ku = ke;
      //else // 3D
        //jl = js - 1, ju = je + 1, kl = ks - 1, ku = ke + 1;
    //}
  //}

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

        Real dust_v1_new = (gas_v1 * alpha_dg *dt + dust_v1 * (1.0 - alpha_dg *dt));
        Real dust_v2_new = (gas_v2 * alpha_dg *dt + dust_v2 * (1.0 - alpha_dg *dt));
        Real dust_v3_new = (gas_v3 * alpha_dg *dt + dust_v3 * (1.0 - alpha_dg *dt));

        Real gas_v1_new = (dust_v1 * alpha_gd *dt + gas_v1 * (1.0 - alpha_gd *dt));
        Real gas_v2_new = (dust_v2 * alpha_gd *dt + gas_v2 * (1.0 - alpha_gd *dt));
        Real gas_v3_new = (dust_v3 * alpha_gd *dt + gas_v3 * (1.0 - alpha_gd *dt));

        // Update the Momentum of gas and dust
        Real delta_dust_m1 = dust_d*(dust_v1_new - dust_v1);
        Real delta_dust_m2 = dust_d*(dust_v2_new - dust_v2);
        Real delta_dust_m3 = dust_d*(dust_v3_new - dust_v3);

        Real delta_gas_m1 = gas_d*(gas_v1_new - gas_v1);
        Real delta_gas_m2 = gas_d*(gas_v2_new - gas_v2);
        Real delta_gas_m3 = gas_d*(gas_v3_new - gas_v3);

        // Update the Momentum of gas and dust
        dust_m1 += delta_dust_m1;
        dust_m2 += delta_dust_m2;
        dust_m3 += delta_dust_m3;

        gas_m1  += delta_gas_m1;
        gas_m2  += delta_gas_m2;
        gas_m3  += delta_gas_m3;

        // Update the energy of gas if the gas is non barotropic.
        if (NON_BAROTROPIC_EOS)
          gas_e += delta_gas_m1*gas_v1 + delta_gas_m2*gas_v2 + delta_gas_m3*gas_v3;

        }
      }
    }
  return;
}
