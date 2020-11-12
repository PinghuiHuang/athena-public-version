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


//void DustGasDrag::TRBDF2Feedback(const int stage,
      //const Real dt, const AthenaArray<Real> &stopping_time,
      //const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      //AthenaArray<Real> &u, AthenaArray<Real> &cons_df) {

  //MeshBlock *pmb = pmy_dustfluids_->pmy_block;

  //const bool f2 = pmb->pmy_mesh->f2;
  //const bool f3 = pmb->pmy_mesh->f3;

  //DustFluids  *pdf = pmy_dustfluids_;
  //Hydro       *ph  = pmb->phydro;

  //AthenaArray<Real> &u_n = ph->u_n;
  //AthenaArray<Real> &u_p = ph->u_p;

  //AthenaArray<Real> &cons_df_n = pdf->df_cons_n;
  //AthenaArray<Real> &cons_df_p = pdf->df_cons_p;

  //int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  //int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  //AthenaArray<Real> u_d(NHYDRO,             pmb->ncells3, pmb->ncells2, pmb->ncells1);
  //AthenaArray<Real> cons_df_d(num_dust_var, pmb->ncells3, pmb->ncells2, pmb->ncells1);

  //if ( stage == 1 ) {
    ////std::cout << "Stage 1: The time step is " << dt << std::endl;
    //// Step 1a: All the explicit source terms are added before the drags task
    //// ^(') is marked as "_p", ^(n) is marked as "_n", ^(n+1) is marked as "_n1". prim: w^(n), cons: u^(')
    //AthenaArray<Real> force_x1_n(num_species);
    //AthenaArray<Real> force_x2_n(num_species);
    //AthenaArray<Real> force_x3_n(num_species);

    //AthenaArray<Real> jacobi_matrix_n(num_species,  num_species);

    //AthenaArray<Real> lambda_matrix_n(num_species, num_species);
    //AthenaArray<Real> lambda_inv_matrix_n(num_species, num_species);

    //AthenaArray<Real> delta_m1(num_species);
    //AthenaArray<Real> delta_m2(num_species);
    //AthenaArray<Real> delta_m3(num_species);

    //// Step 1b: Calculate the differences between u^(') and u^(n), u_d = u_p - u_n, "_d" means "differences", dt = h/2
    //Real wghts[3] = {0.0, 1.0, -1.0};
    //pmb->WeightedAve(u_d,       u_p,       u_n,       wghts);
    //pmb->WeightedAve(cons_df_d, cons_df_p, cons_df_n, wghts);
    //Real inv_dt = 1.0/dt;
    ////std::cout << "Stage 0: The time step is " << dt << std::endl;

    //for (int k=ks; k<=ke; ++k) {
      //for (int j=js; j<=je; ++j) {
////#pragma omp simd
        //for (int i=is; i<=ie; ++i) {
          //force_x1_n.ZeroClear();
          //force_x2_n.ZeroClear();
          //force_x3_n.ZeroClear();

          //jacobi_matrix_n.ZeroClear();

          //lambda_matrix_n.ZeroClear();
          //lambda_inv_matrix_n.ZeroClear();

          //delta_m1.ZeroClear();
          //delta_m2.ZeroClear();
          //delta_m3.ZeroClear();

          //// Set the drag force, depends on the stage of u^(n), f(M^(n), V^(n))
          //for (int index = 1; index <= NDUSTFLUIDS; ++index) {
            //int dust_id          = index - 1;
            //int rho_id           = 4*dust_id;
            //int v1_id            = rho_id + 1;
            //int v2_id            = rho_id + 2;
            //int v3_id            = rho_id + 3;
            //const Real &dust_d_n = prim_df(rho_id, k, j, i);
            //const Real &gas_d_n  = w(IDN, k, j, i);
            //Real alpha_n         = 1.0/(stopping_time(dust_id,k,j,i) + TINY_NUMBER);
            //Real epsilon_n       = dust_d_n/gas_d_n;

            //force_x1_n(index) = epsilon_n * alpha_n * u_n(IM1, k, j, i) - alpha_n * cons_df_n(v1_id, k, j, i);
            //force_x2_n(index) = epsilon_n * alpha_n * u_n(IM2, k, j, i) - alpha_n * cons_df_n(v2_id, k, j, i);
            //force_x3_n(index) = epsilon_n * alpha_n * u_n(IM3, k, j, i) - alpha_n * cons_df_n(v3_id, k, j, i);
          //}

          //for (int dust_index = 1; dust_index <= NDUSTFLUIDS; ++dust_index) {
            //force_x1_n(0) -= force_x1_n(dust_index);
            //force_x2_n(0) -= force_x2_n(dust_index);
            //force_x3_n(0) -= force_x3_n(dust_index);
          //}

          ////std::cout << "Stage 1: " << std::endl;
          ////for (int mm = 0; mm <= NDUSTFLUIDS; ++mm)
            ////std::cout << "force_x1_n("<< mm << ") is " << force_x1_n(mm) << std::endl;

          ////// Add the delta momentum caused by the other source terms, \Delta Gm = (u^(') - u^(n))/dt
          ////for (int index = 1; index <= NDUSTFLUIDS; ++index) {
            ////int dust_id = index - 1;
            ////int rho_id  = 4*dust_id;
            ////int v1_id   = rho_id + 1;
            ////int v2_id   = rho_id + 2;
            ////int v3_id   = rho_id + 3;

            ////force_x1_n(index) += cons_df_d(v1_id, k, j, i) * inv_dt;
            ////force_x2_n(index) += cons_df_d(v2_id, k, j, i) * inv_dt;
            ////force_x3_n(index) += cons_df_d(v3_id, k, j, i) * inv_dt;
          ////}

          ////force_x1_n(0) += u_d(IM1, k, j, i) * inv_dt;
          ////force_x2_n(0) += u_d(IM2, k, j, i) * inv_dt;
          ////force_x3_n(0) += u_d(IM3, k, j, i) * inv_dt;

          //// Calculate the jacobi matrix of the drag forces, df/dM|^(n)
          //// Set the jacobi_matrix_n(0, col), except jacobi_matrix_n(0, 0)
          //for (int col = 1; col <= NDUSTFLUIDS; col++) {
            //int dust_id             = col - 1;
            //jacobi_matrix_n(0, col) = 1.0/(stopping_time(dust_id,k,j,i) + TINY_NUMBER);
          //}

          //// Set the jacobi_matrix_n(row, 0), except jacobi_matrix_n(0, 0)
          //for (int row = 1; row <= NDUSTFLUIDS; row++) {
            //int dust_id             = row - 1;
            //int rho_id              = 4*dust_id;
            //const Real &dust_d_n    = prim_df(rho_id, k, j, i);
            //const Real &gas_d_n     = w(IDN, k, j, i);
            //jacobi_matrix_n(row, 0) = dust_d_n/gas_d_n * 1.0/(stopping_time(dust_id,k,j,i) + TINY_NUMBER);
          //}

          //// Set the jacobi_matrix_n(0,0) at stage (n), use the w^(n) and prim^(n)
          //for (int dust_id = 0; dust_id < NDUSTFLUIDS; dust_id++) {
            //int rho_id             = 4*dust_id;
            //int row                = dust_id + 1;
            //jacobi_matrix_n(0, 0) -= jacobi_matrix_n(row, 0);
          //}

          //// Set the other pivots, except jacobi_matrix_n(0, 0)
          //for (int pivot = 1; pivot <= NDUSTFLUIDS; pivot++) {
            //int dust_id                    = pivot - 1;
            //int col                        = pivot;
            //jacobi_matrix_n(pivot, pivot)  = -1.0*jacobi_matrix_n(0, col);
          //}

          ////for (int mm = 0; mm <= NDUSTFLUIDS; ++mm) {
            ////std::cout << "Stage 1: " << std::endl;
            ////for (int nn = 0; nn <= NDUSTFLUIDS; ++nn) {
              ////std::cout << "jacobi_n("<< mm << ',' << nn << ") is " << jacobi_matrix_n(mm, nn) << "; ";
            ////}
            ////std::cout << std::endl;
          ////}

          //// calculate lambda_matrix_n = I - h/2*jacobi_matrix_n, dt = h/2
          //Multiplication(dt, jacobi_matrix_n, lambda_matrix_n);
          //Addition(1.0, -1.0, lambda_matrix_n);

          //// cauculate the inverse matrix of lambda_matrix
          //LUdecompose(lambda_matrix_n);
          //Inverse(lambda_matrix_n, lambda_inv_matrix_n);

          //// Step 1c && 1d: calculate delta momentum (Delta_M) caused by the drags, update the conserves
          //// Delta_M = h/2 * (lambda_matrix_n)^(-1) * f(M^(n), V^(n)), dt = h/2
          //Multiplication(dt, lambda_inv_matrix_n);
          //Multiplication(lambda_inv_matrix_n, force_x1_n, delta_m1);
          //Multiplication(lambda_inv_matrix_n, force_x2_n, delta_m2);
          //Multiplication(lambda_inv_matrix_n, force_x3_n, delta_m3);

          //// Alias the primitives of gas w^(n), "_n"
          //const Real &gas_d_n  = w(IDN, k, j, i);
          //const Real &gas_v1_n = w(IVX, k, j, i);
          //const Real &gas_v2_n = w(IVY, k, j, i);
          //const Real &gas_v3_n = w(IVZ, k, j, i);

          //// Alias the conserves of gas u^('), "_p"
          //Real &gas_m1_p = u(IM1, k, j, i);
          //Real &gas_m2_p = u(IM2, k, j, i);
          //Real &gas_m3_p = u(IM3, k, j, i);

          ////// Add the delta momentum caused by drags on the gas conserves, u^(') -> M^(')
          ////gas_m1_p += delta_m1(0);
          ////gas_m2_p += delta_m2(0);
          ////gas_m3_p += delta_m3(0);

          //pdf->u_s1(IM1, k, j, i) = gas_m1_p + delta_m1(0);
          //pdf->u_s1(IM2, k, j, i) = gas_m2_p + delta_m2(0);
          //pdf->u_s1(IM3, k, j, i) = gas_m3_p + delta_m3(0);

          //// Update the energy of gas if the gas is non barotropic. dE = dM * v^(n)
          //if (NON_BAROTROPIC_EOS) {
            //Real &gas_e_p = u(IEN, k, j, i);
            //gas_e_p += delta_m1(0)*gas_v1_n + delta_m2(0)*gas_v2_n + delta_m3(0)*gas_v3_n;
          //}

          //for (int n = 1; n <= NDUSTFLUIDS; ++n) {
            //int dust_id = n - 1;
            //int rho_id  = 4*dust_id;
            //int v1_id   = rho_id + 1;
            //int v2_id   = rho_id + 2;
            //int v3_id   = rho_id + 3;

            //// Alias the conserves of dust u^('), "_p"
            //Real &dust_m1_p = cons_df(v1_id, k, j, i);
            //Real &dust_m2_p = cons_df(v2_id, k, j, i);
            //Real &dust_m3_p = cons_df(v3_id, k, j, i);

            ////// Add the delta momentum caused by drags on the dust conserves, u^(') -> M^(')
            ////dust_m1_p += delta_m1(n);
            ////dust_m2_p += delta_m2(n);
            ////dust_m3_p += delta_m3(n);

            //pdf->df_cons_s1(v1_id, k, j, i) = dust_m1_p + delta_m1(n);
            //pdf->df_cons_s1(v2_id, k, j, i) = dust_m2_p + delta_m2(n);
            //pdf->df_cons_s1(v3_id, k, j, i) = dust_m3_p + delta_m3(n);
          //}

          ////Real total_mom = 0.0;
          ////for (int mm = 0; mm <=NDUSTFLUIDS; ++mm) {
            ////total_mom += delta_m1(mm);
          ////}
          ////std::cout << "Stage 1: Total_mom = " << total_mom << std::endl;
          ////std::cout << std::endl;
        //}
      //}
    //}
  //}
  //else { // stage == 2
    //// Step 2a: All the explicit source terms are added before the drags task
    //// ^(') is marked as "_p", ^(n) is marked as "_n", ^(n+1) is marked as "_n1". prim: w^('), cons: u^(n+1)
    //AthenaArray<Real> force_x1_p(num_species);
    //AthenaArray<Real> force_x2_p(num_species);
    //AthenaArray<Real> force_x3_p(num_species);

    //AthenaArray<Real> delta_m1(num_species);
    //AthenaArray<Real> delta_m2(num_species);
    //AthenaArray<Real> delta_m3(num_species);

    //AthenaArray<Real> jacobi_matrix_p(num_species, num_species);

    //AthenaArray<Real> temp_A_matrix_p(num_species, num_species);
    //AthenaArray<Real> temp_B_matrix_p(num_species, num_species);

    //AthenaArray<Real> lambda_matrix_p(num_species, num_species);
    //AthenaArray<Real> lambda_inv_matrix_p(num_species, num_species);

    //// Step 2b: Calculate the differences between u^(n+1) and u^(n), u_d = u - u_n
    //Real wghts[3] = {0.0, 1.0, -1.0};
    //pmb->WeightedAve(u_d,       u,       u_n,       wghts);
    //pmb->WeightedAve(cons_df_d, cons_df, cons_df_n, wghts);
    //Real inv_dt = 1.0/dt;
    ////std::cout << "Stage 2: The time step is " << dt << std::endl;

    //for (int k=ks; k<=ke; ++k) {
      //for (int j=js; j<=je; ++j) {
////#pragma omp simd
        //for (int i=is; i<=ie; ++i) {
          //// Set the drag force, depends on the stage of u^('), f(M^('), V^('))
          //force_x1_p.ZeroClear();
          //force_x2_p.ZeroClear();
          //force_x3_p.ZeroClear();

          //delta_m1.ZeroClear();
          //delta_m2.ZeroClear();
          //delta_m3.ZeroClear();

          //jacobi_matrix_p.ZeroClear();

          //temp_A_matrix_p.ZeroClear();
          //temp_B_matrix_p.ZeroClear();

          //lambda_matrix_p.ZeroClear();
          //lambda_inv_matrix_p.ZeroClear();

          //for (int index = 1; index <= NDUSTFLUIDS; ++index) {
            //int dust_id          = index - 1;
            //int rho_id           = 4*dust_id;
            //int v1_id            = rho_id + 1;
            //int v2_id            = rho_id + 2;
            //int v3_id            = rho_id + 3;
            //const Real &dust_d_p = prim_df(rho_id, k, j, i);
            //const Real &gas_d_p  = w(IDN, k, j, i);
            //Real alpha_p         = 1.0/(stopping_time(dust_id,k,j,i) + TINY_NUMBER);
            //Real epsilon_p       = dust_d_p/gas_d_p;

            ////force_x1_p(index) = epsilon_p * alpha_p * u_n(IM1, k, j, i) - alpha_p * cons_df_n(v1_id, k, j, i);
            ////force_x2_p(index) = epsilon_p * alpha_p * u_n(IM2, k, j, i) - alpha_p * cons_df_n(v2_id, k, j, i);
            ////force_x3_p(index) = epsilon_p * alpha_p * u_n(IM3, k, j, i) - alpha_p * cons_df_n(v3_id, k, j, i);

            //force_x1_p(index) = epsilon_p * alpha_p * pdf->u_s1(IM1, k, j, i) - alpha_p * pdf->df_cons_s1(v1_id, k, j, i);
            //force_x2_p(index) = epsilon_p * alpha_p * pdf->u_s1(IM2, k, j, i) - alpha_p * pdf->df_cons_s1(v2_id, k, j, i);
            //force_x3_p(index) = epsilon_p * alpha_p * pdf->u_s1(IM3, k, j, i) - alpha_p * pdf->df_cons_s1(v3_id, k, j, i);

            ////std::cout << "u_n is " << u_n(IM1, k, j, i) << std::endl;
            ////std::cout << "u_p is " << u_p(IM1, k, j, i) << std::endl;
            ////std::cout << "u_s1 is " << pdf->u_s1(IM1, k, j, i) << std::endl;

            ////std::cout << "pdf->df_cons_n is " << cons_df_n(v1_id, k, j, i) << std::endl;
            ////std::cout << "pdf->df_cons_p is " << cons_df_p(v1_id, k, j, i) << std::endl;
            ////std::cout << "pdf->df_cons_s1 is " << pdf->df_cons_s1(v1_id, k, j, i) << std::endl;
          //}

          //for (int dust_index = 1; dust_index <= NDUSTFLUIDS; ++dust_index) {
            //force_x1_p(0) -= force_x1_p(dust_index);
            //force_x2_p(0) -= force_x2_p(dust_index);
            //force_x3_p(0) -= force_x3_p(dust_index);
          //}

          ////std::cout << "Stage 2: " << std::endl;
          ////for (int mm = 0; mm <= NDUSTFLUIDS; ++mm)
            ////std::cout << "force_x1_p("<< mm << ") is " << force_x1_p(mm) << std::endl;

          ////// Add the delta momentum caused by the other source terms, \Delta Gm = (u^(n+1) - u^(n))/h, dt = h
          ////for (int index = 1; index <= NDUSTFLUIDS; ++index) {
            ////int dust_id = index - 1;
            ////int rho_id  = 4*dust_id;
            ////int v1_id   = rho_id + 1;
            ////int v2_id   = rho_id + 2;
            ////int v3_id   = rho_id + 3;

            ////force_x1_p(index) += cons_df_d(v1_id, k, j, i) * inv_dt;
            ////force_x2_p(index) += cons_df_d(v2_id, k, j, i) * inv_dt;
            ////force_x3_p(index) += cons_df_d(v3_id, k, j, i) * inv_dt;
          ////}

          ////force_x1_p(0) += u_d(IM1, k, j, i) * inv_dt;
          ////force_x2_p(0) += u_d(IM2, k, j, i) * inv_dt;
          ////force_x3_p(0) += u_d(IM3, k, j, i) * inv_dt;

          //// Calculate the jacobi matrix of the drag forces, \partial f/\partial M|^(')
          //// Set the jacobi_matrix_p(0, col), except jacobi_matrix_p(0, 0)
          //for (int col = 1; col <= NDUSTFLUIDS; col++) {
            //int dust_id             = col - 1;
            //jacobi_matrix_p(0, col) = 1.0/(stopping_time(dust_id,k,j,i) + TINY_NUMBER);
          //}

          //// Set the jacobi_matrix_p(row, 0), except jacobi_matrix_p(0, 0)
          //for (int row = 1; row <= NDUSTFLUIDS; row++) {
            //int dust_id             = row - 1;
            //int rho_id              = 4*dust_id;
            //const Real &dust_d_p    = prim_df(rho_id, k, j, i);
            //const Real &gas_d_p     = w(IDN, k, j, i);
            //jacobi_matrix_p(row, 0) = dust_d_p/gas_d_p * 1.0/(stopping_time(dust_id,k,j,i) + TINY_NUMBER);
          //}

          //// Set the jacobi_matrix_p(0,0) at stage ('), use the w^(') and prim^(')
          //for (int dust_id = 0; dust_id < NDUSTFLUIDS; dust_id++) {
            //int rho_id             = 4*dust_id;
            //int row                = dust_id + 1;
            //jacobi_matrix_p(0, 0) -= jacobi_matrix_p(row, 0);
          //}

          //// Set the other pivots, except jacobi_matrix_p(0, 0)
          //for (int pivot = 1; pivot <= NDUSTFLUIDS; pivot++) {
            //int dust_id                   = pivot - 1;
            //int col                       = pivot;
            //jacobi_matrix_p(pivot, pivot) = -1.0*jacobi_matrix_p(0, col);
          //}

          ////for (int mm = 0; mm <= NDUSTFLUIDS; ++mm) {
            ////std::cout << "Stage 2: " << std::endl;
            ////for (int nn = 0; nn <= NDUSTFLUIDS; ++nn) {
              ////std::cout << "jacobi_p("<< mm << ',' << nn << ") is " << jacobi_matrix_p(mm,nn) << "; ";
            ////}
            ////std::cout << std::endl;
          ////}

          //// Step 2c && 2d: calculate delta momentum (Delta_M) caused by the drags, update the conserves
          //// calculate temp_A_matrix = I - h*jacobi_matrix_p, dt = h
          //Multiplication(dt, jacobi_matrix_p, temp_A_matrix_p);
          //Addition(1.0, -0.5, temp_A_matrix_p);

          ////for (int mm = 0; mm <= NDUSTFLUIDS; ++mm) {
            ////std::cout << "Stage 2: " << std::endl;
            ////for (int nn = 0; nn <= NDUSTFLUIDS; ++nn) {
              ////std::cout << "temp_A_matrix_p("<< mm << ',' << nn << ") is " << temp_A_matrix_p(mm,nn) << "; ";
            ////}
            ////std::cout << std::endl;
          ////}


          //// calculate the lambda_matrix = I - h*temp_A_matrix*jacobi_matrix_p, dt = h
          //Multiplication(temp_A_matrix_p, jacobi_matrix_p, lambda_matrix_p);
          //Addition(1.0, -1.0*dt, lambda_matrix_p);

          ////for (int mm = 0; mm <= NDUSTFLUIDS; ++mm) {
            ////std::cout << "Stage 2: " << std::endl;
            ////for (int nn = 0; nn <= NDUSTFLUIDS; ++nn) {
              ////std::cout << "lambda_matrix_p("<< mm << ',' << nn << ") is " << lambda_matrix_p(mm,nn) << "; ";
            ////}
            ////std::cout << std::endl;
          ////}


          //// cauculate the inverse matrix of lambda_matrix
          //LUdecompose(lambda_matrix_p);
          //Inverse(lambda_matrix_p, lambda_inv_matrix_p);

          ////for (int mm = 0; mm <= NDUSTFLUIDS; ++mm) {
            ////std::cout << "Stage 2: " << std::endl;
            ////for (int nn = 0; nn <= NDUSTFLUIDS; ++nn) {
              ////std::cout << "lambda_inv_matrix_p("<< mm << ',' << nn << ") is " << lambda_inv_matrix_p(mm,nn) << "; ";
            ////}
            ////std::cout << std::endl;
          ////}

          //// calculate temp_B_matrix = h * lambda^-1 * temp_A_matrix, dt = h
          //Multiplication(lambda_inv_matrix_p, temp_A_matrix_p, temp_B_matrix_p);
          //Multiplication(dt, temp_B_matrix_p);

          ////for (int mm = 0; mm <= NDUSTFLUIDS; ++mm) {
            ////std::cout << "Stage 2: " << std::endl;
            ////for (int nn = 0; nn <= NDUSTFLUIDS; ++nn) {
              ////std::cout << "temp_B_matrix_p("<< mm << ',' << nn << ") is " << temp_B_matrix_p(mm,nn) << "; ";
            ////}
            ////std::cout << std::endl;
          ////}

          //// Delta_M = temp_B_matrix * f(M^(n), V^(n))
          //Multiplication(temp_B_matrix_p, force_x1_p, delta_m1);
          //Multiplication(temp_B_matrix_p, force_x2_p, delta_m2);
          //Multiplication(temp_B_matrix_p, force_x3_p, delta_m3);

          //// Alias the primitives of gas, w^(')
          //const Real &gas_d_p  = w(IDN, k, j, i);
          //const Real &gas_v1_p = w(IVX, k, j, i);
          //const Real &gas_v2_p = w(IVY, k, j, i);
          //const Real &gas_v3_p = w(IVZ, k, j, i);

          //// Alias the conserves of gas, u^(n+1)
          //Real &gas_m1_n1 = u(IM1, k, j, i);
          //Real &gas_m2_n1 = u(IM2, k, j, i);
          //Real &gas_m3_n1 = u(IM3, k, j, i);

          //// Add the delta momentum caused by drags on the gas conserves, u^(n+1) -> M^(n+1)
          //gas_m1_n1 += delta_m1(0);
          //gas_m2_n1 += delta_m2(0);
          //gas_m3_n1 += delta_m3(0);

          //// Update the energy of gas if the gas is non barotropic.
          //if (NON_BAROTROPIC_EOS) {
            //Real &gas_e_n1  = u(IEN, k, j, i);
            //gas_e_n1 += delta_m1(0)*gas_v1_p + delta_m2(0)*gas_v2_p + delta_m3(0)*gas_v3_p;
          //}

          //for (int n = 1; n <= NDUSTFLUIDS; ++n) {
            //int dust_id = n - 1;
            //int rho_id  = 4*dust_id;
            //int v1_id   = rho_id + 1;
            //int v2_id   = rho_id + 2;
            //int v3_id   = rho_id + 3;

            //// Alias the conserves of dust, u^(n+1)
            //Real &dust_m1_n1 = cons_df(v1_id, k, j, i);
            //Real &dust_m2_n1 = cons_df(v2_id, k, j, i);
            //Real &dust_m3_n1 = cons_df(v3_id, k, j, i);

            //// Add the delta momentum caused by drags on the dust conserves, u^(n+1) -> M^(n+1)
            //dust_m1_n1 += delta_m1(n);
            //dust_m2_n1 += delta_m2(n);
            //dust_m3_n1 += delta_m3(n);
          //}

          ////Real total_mom = 0.0;
          ////for (int mm = 0; mm <=NDUSTFLUIDS; ++mm) {
            ////total_mom += delta_m1(mm);
            ////std::cout << "Stage 2: delta_m1(" << mm << ") is " << delta_m1(mm) << std::endl;
          ////}
          ////std::cout << "Stage 2: Total_mom = " << total_mom << std::endl;
          ////std::cout << std::endl;
        //}
      //}
    //}
  //}

  //return;
//}

void DustGasDrag::TRBDF2Feedback(const int stage,
      const Real dt, const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      AthenaArray<Real> &u, AthenaArray<Real> &cons_df) {

  MeshBlock *pmb = pmy_dustfluids_->pmy_block;

  const bool f2 = pmb->pmy_mesh->f2;
  const bool f3 = pmb->pmy_mesh->f3;

  DustFluids  *pdf = pmy_dustfluids_;
  Hydro       *ph  = pmb->phydro;

  //AthenaArray<Real> &u_n = ph->u_n;
  //AthenaArray<Real> &u_p = ph->u_p;

  //AthenaArray<Real> &cons_df_n = pdf->df_cons_n;
  //AthenaArray<Real> &cons_df_p = pdf->df_cons_p;

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  //AthenaArray<Real> u_d(NHYDRO,             pmb->ncells3, pmb->ncells2, pmb->ncells1);
  //AthenaArray<Real> cons_df_d(num_dust_var, pmb->ncells3, pmb->ncells2, pmb->ncells1);

  AthenaArray<Real> force_x1(num_species);
  AthenaArray<Real> force_x2(num_species);
  AthenaArray<Real> force_x3(num_species);

  AthenaArray<Real> jacobi_matrix(num_species, num_species);
  AthenaArray<Real> lambda_matrix_A(num_species, num_species);
  AthenaArray<Real> lambda_inv_matrix_A(num_species, num_species);
  AthenaArray<Real> lambda_matrix_B(num_species, num_species);
  AthenaArray<Real> lambda_inv_matrix_B(num_species, num_species);

  AthenaArray<Real> delta_m1_implicit_A(num_species);
  AthenaArray<Real> delta_m2_implicit_A(num_species);
  AthenaArray<Real> delta_m3_implicit_A(num_species);

  AthenaArray<Real> delta_m1_implicit_B(num_species);
  AthenaArray<Real> delta_m2_implicit_B(num_species);
  AthenaArray<Real> delta_m3_implicit_B(num_species);

  AthenaArray<Real> delta_m1_explicit(num_species);
  AthenaArray<Real> delta_m2_explicit(num_species);
  AthenaArray<Real> delta_m3_explicit(num_species);

  AthenaArray<Real> delta_m1(num_species);
  AthenaArray<Real> delta_m2(num_species);
  AthenaArray<Real> delta_m3(num_species);

  // Step 1b: Calculate the differences between u^(') and u^(n), u_d = u_p - u_n, "_d" means "differences", dt = h/2
  //Real wghts[3] = {0.0, 1.0, -1.0};
  //pmb->WeightedAve(u_d,       u_p,       u_n,       wghts);
  //pmb->WeightedAve(cons_df_d, cons_df_p, cons_df_n, wghts);
  //Real inv_dt = 1.0/dt;
  //std::cout << "Stage 0: The time step is " << dt << std::endl;

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
//#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        force_x1.ZeroClear();
        force_x2.ZeroClear();
        force_x3.ZeroClear();

        jacobi_matrix.ZeroClear();
        lambda_matrix_A.ZeroClear();
        lambda_inv_matrix_A.ZeroClear();
        lambda_matrix_B.ZeroClear();
        lambda_inv_matrix_B.ZeroClear();

        delta_m1_implicit_A.ZeroClear();
        delta_m2_implicit_A.ZeroClear();
        delta_m3_implicit_A.ZeroClear();

        delta_m1_implicit_B.ZeroClear();
        delta_m2_implicit_B.ZeroClear();
        delta_m3_implicit_B.ZeroClear();

        delta_m1_explicit.ZeroClear();
        delta_m2_explicit.ZeroClear();
        delta_m3_explicit.ZeroClear();

        delta_m1.ZeroClear();
        delta_m2.ZeroClear();
        delta_m3.ZeroClear();

        // Alias the primitives of gas
        const Real &gas_rho = w(IDN,k,j,i);
        const Real &gas_v1  = w(IVX,k,j,i);
        const Real &gas_v2  = w(IVY,k,j,i);
        const Real &gas_v3  = w(IVZ,k,j,i);

        // Set the drag force
        for (int index=1; index<=NDUSTFLUIDS; ++index) {
          int dust_id          = index - 1;
          int rho_id           = 4*dust_id;
          int v1_id            = rho_id + 1;
          int v2_id            = rho_id + 2;
          int v3_id            = rho_id + 3;
          const Real &dust_rho = prim_df(rho_id,k,j,i);
          Real alpha           = 1.0/(stopping_time(dust_id,k,j,i) + TINY_NUMBER);
          Real epsilon         = dust_rho/gas_rho;

          const Real &dust_v1 = prim_df(v1_id,k,j,i);
          const Real &dust_v2 = prim_df(v2_id,k,j,i);
          const Real &dust_v3 = prim_df(v3_id,k,j,i);

          Real gas_mom1 = gas_rho*gas_v1;
          Real gas_mom2 = gas_rho*gas_v2;
          Real gas_mom3 = gas_rho*gas_v3;

          Real dust_mom1 = dust_rho*dust_v1;
          Real dust_mom2 = dust_rho*dust_v2;
          Real dust_mom3 = dust_rho*dust_v3;

          force_x1(index) = epsilon * alpha * gas_mom1 - alpha * dust_mom1;
          force_x2(index) = epsilon * alpha * gas_mom2 - alpha * dust_mom2;
          force_x3(index) = epsilon * alpha * gas_mom3 - alpha * dust_mom3;
        }

        for (int dust_index=1; dust_index<=NDUSTFLUIDS; ++dust_index) {
          force_x1(0) -= force_x1(dust_index);
          force_x2(0) -= force_x2(dust_index);
          force_x3(0) -= force_x3(dust_index);
        }

        //std::cout << "Stage 1: " << std::endl;
        //for (int mm = 0; mm <= NDUSTFLUIDS; ++mm)
          //std::cout << "force_x1_n("<< mm << ") is " << force_x1_n(mm) << std::endl;

        //// Add the delta momentum caused by the other source terms, \Delta Gm = (u^(') - u^(n))/dt
        //for (int index = 1; index <= NDUSTFLUIDS; ++index) {
          //int dust_id = index - 1;
          //int rho_id  = 4*dust_id;
          //int v1_id   = rho_id + 1;
          //int v2_id   = rho_id + 2;
          //int v3_id   = rho_id + 3;

          //force_x1_n(index) += cons_df_d(v1_id, k, j, i) * inv_dt;
          //force_x2_n(index) += cons_df_d(v2_id, k, j, i) * inv_dt;
          //force_x3_n(index) += cons_df_d(v3_id, k, j, i) * inv_dt;
        //}

        //force_x1_n(0) += u_d(IM1, k, j, i) * inv_dt;
        //force_x2_n(0) += u_d(IM2, k, j, i) * inv_dt;
        //force_x3_n(0) += u_d(IM3, k, j, i) * inv_dt;

        // Calculate the jacobi matrix of the drag forces, df/dM|^(n)
        // Set the jacobi_matrix(0, col), except jacobi_matrix(0, 0)
        for (int col = 1; col <= NDUSTFLUIDS; ++col) {
          int dust_id           = col - 1;
          jacobi_matrix(0, col) = 1.0/(stopping_time(dust_id,k,j,i) + TINY_NUMBER);
        }

        // Set the jacobi_matrix(row, 0), except jacobi_matrix(0, 0)
        for (int row = 1; row <= NDUSTFLUIDS; ++row) {
          int dust_id           = row - 1;
          int rho_id            = 4*dust_id;
          const Real &dust_rho  = prim_df(rho_id, k, j, i);
          jacobi_matrix(row, 0) = dust_rho/gas_rho * 1.0/(stopping_time(dust_id,k,j,i) + TINY_NUMBER);
        }

        // Set the jacobi_matrix(0,0) at stage (n)
        for (int dust_id = 0; dust_id < NDUSTFLUIDS; ++dust_id) {
          int row              = dust_id + 1;
          jacobi_matrix(0, 0) -= jacobi_matrix(row, 0);
        }

        // Set the other pivots, except jacobi_matrix(0, 0)
        for (int pivot = 1; pivot <= NDUSTFLUIDS; ++pivot) {
          int col                     = pivot;
          jacobi_matrix(pivot, pivot) = -1.0*jacobi_matrix(0, col);
        }

        // calculate lambda_matrix_A = I - h*jacobi_matrix at n+1
        Multiplication(dt, jacobi_matrix, lambda_matrix_A);
        Addition(1.0, -1.0, lambda_matrix_A);
        LUdecompose(lambda_matrix_A);
        Inverse(lambda_matrix_A, lambda_inv_matrix_A);
        Multiplication(dt, lambda_inv_matrix_A);

        Multiplication(lambda_inv_matrix_A, force_x1, delta_m1_implicit_A);
        Multiplication(lambda_inv_matrix_A, force_x2, delta_m2_implicit_A);
        Multiplication(lambda_inv_matrix_A, force_x3, delta_m3_implicit_A);

        // calculate lambda_matrix_B = I - h/2*jacobi_matrix at n+1/2
        Multiplication(dt, jacobi_matrix, lambda_matrix_B);
        Addition(1.0, -0.5, lambda_matrix_B);
        LUdecompose(lambda_matrix_B);
        Inverse(lambda_matrix_B, lambda_inv_matrix_B);
        Multiplication(dt, lambda_inv_matrix_B);

        Multiplication(lambda_inv_matrix_B, force_x1, delta_m1_implicit_B);
        Multiplication(lambda_inv_matrix_B, force_x2, delta_m2_implicit_B);
        Multiplication(lambda_inv_matrix_B, force_x3, delta_m3_implicit_B);

        // calculate the explicit part at n+1/2
        for (int n=0; n<=NDUSTFLUIDS; ++n) {
          delta_m1_explicit(n) = force_x1(n) * dt;
          delta_m2_explicit(n) = force_x2(n) * dt;
          delta_m3_explicit(n) = force_x3(n) * dt;
        }

        for (int n=0; n<=NDUSTFLUIDS; ++n) {
          delta_m1(n) = 1./3.*delta_m1_implicit_A(n) + 1./3.*delta_m1_implicit_B(n) + 1./3.*delta_m1_explicit(n);
          delta_m2(n) = 1./3.*delta_m2_implicit_A(n) + 1./3.*delta_m2_implicit_B(n) + 1./3.*delta_m2_explicit(n);
          delta_m3(n) = 1./3.*delta_m3_implicit_A(n) + 1./3.*delta_m3_implicit_B(n) + 1./3.*delta_m3_explicit(n);
        }


        // Alias the conserves of gas
        Real &gas_m1 = u(IM1, k, j, i);
        Real &gas_m2 = u(IM2, k, j, i);
        Real &gas_m3 = u(IM3, k, j, i);

        gas_m1 += delta_m1(0);
        gas_m2 += delta_m2(0);
        gas_m3 += delta_m3(0);

        // Update the energy of gas if the gas is non barotropic.
        if (NON_BAROTROPIC_EOS) {
          Real &gas_e     = u(IEN, k, j, i);
          Real delta_erg  = delta_m1(0)*gas_v1 + delta_m2(0)*gas_v2 + delta_m3(0)*gas_v3;
          gas_e          += delta_erg;
        }

        for (int n=1; n<=NDUSTFLUIDS; ++n) {
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
