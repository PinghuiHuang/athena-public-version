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

DustGasDrag::DustGasDrag(DustFluids *pdf, ParameterInput *pin) :
  drags_matrix(num_species, num_species), // The drags matrix
  lu_matrix(num_species,    num_species), // The LU decomposition matrix
  aref_matrix(num_species,  num_species), // The matrix for iternative calculation
  indx_array(num_species),                // Stores the permutation
  pmy_dustfluids_(pdf), pmb_(pmy_dustfluids_->pmy_block), pco_(pmb_->pcoord) {

  //hydro_gamma_      = pin->GetReal("hydro", "gamma");
  DustFeedback_Flag = pin->GetBoolean("dust", "DustFeedback_Flag");
}


void DustGasDrag::Aerodynamics_Drag(MeshBlock *pmb, const Real dt, const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      AthenaArray<Real> &u, AthenaArray<Real> &cons_df){
  if ( NDUSTFLUIDS == 1 ) { // If the nudstfluids == 1, update the cons by analytical formulas, see eq 6&7 in Stone (1997).
    if (DustFeedback_Flag) {
      SingleDust_Feedback_Implicit(pmb, dt, stopping_time, w, prim_df, u, cons_df);
      //SingleDust_Feedback_SemiImplicit(pmb, dt, stopping_time, w, prim_df, u, cons_df);
    }
    else {
      SingleDust_NoFeedback_Implicit(pmb, dt, stopping_time, w, prim_df, u, cons_df);
      //SingleDust_NoFeedback_SemiImplicit(pmb, dt, stopping_time, w, prim_df, u, cons_df);
    }
  }
  else { // If NDUSTFLUIDS > 1, then LU decompose the drags matrix, see Benitez-Llambay et al. 2019
    if (DustFeedback_Flag) {
      MultipleDust_Feedback_Implicit(pmb, dt, stopping_time, w, prim_df, u, cons_df);
      //MultipleDust_Feedback_SemiImplicit(pmb, dt, stopping_time, w, prim_df, u, cons_df);
    }
    else {
      MultipleDust_NoFeedback_Implicit(pmb, dt, stopping_time, w, prim_df, u, cons_df);
      //MultipleDust_NoFeedback_SemiImplicit(pmb, dt, stopping_time, w, prim_df, u, cons_df);
    }
  }

  return;
}
