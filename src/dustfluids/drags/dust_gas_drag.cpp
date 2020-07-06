//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file dustfluids_diffusion.cpp
//  \brief Compute dustfluids fluxes corresponding to diffusion processes.

// C headers

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
  hydro_gamma_{pin->GetReal("hydro", "gamma")},
  DustFeedback_Flag_{pin->GetOrAddBoolean("dust", "DustFeedback_Flag", false)},
  drags_matrix(num_species, num_species), // The drags matrix
  lu_matrix(num_species,    num_species), // The LU decomposition matrix
  aref_matrix(num_species,  num_species), // The matrix for iternative calculation
  indx_array(num_species),                // Stores the permutation
  pmy_dustfluids_(pdf), pmb_(pmy_dustfluids_->pmy_block), pco_(pmb_->pcoord) {

  //int nc1 = pmb_->ncells1, nc2 = pmb_->ncells2, nc3 = pmb_->ncells3;
  //int is = pmb_->is; int js = pmb_->js; int ks = pmb_->ks;
  //int ie = pmb_->ie; int je = pmb_->je; int ke = pmb_->ke;

  //int il, iu, jl, ju, kl, ku;
  //jl = js, ju = je, kl = ks, ku = ke;
  //if (pmb->block_size.nx2 > 1) {
    //if (pmb->block_size.nx3 == 1) // 2D
      //jl = js-1, ju = je+1, kl = ks, ku = ke;
    //else // 3D
      //jl = js-1, ju = je+1, kl = ks-1, ku = ke+1;
  //}
  //il = is, iu = ie+1;
}

void DustGasDrag::Aerodynamics_Drag(MeshBlock *pmb, const Real dt, const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      AthenaArray<Real> &u, AthenaArray<Real> &cons_df){
  if ( NDUSTFLUIDS == 1 ) { // If the nudstfluids == 1, update the cons by analytical formulas, see eq 6&7 in Stone (1997).
    if (DustFeedback_Flag_)
      Update_Single_Dust_Feedback(pmb, dt, stopping_time, w, prim_df, u, cons_df);
    else
      Update_Single_Dust_NoFeedback(pmb, dt, stopping_time, w, prim_df, u, cons_df);
  }
  else { // If NDUSTFLUIDS > 1, then LU decompose the drags matrix, see Benitez-Llambay et al. 2019
    if (DustFeedback_Flag_)
      Update_Multiple_Dust_Feedback(pmb, dt, stopping_time, w, prim_df, u, cons_df);
    else
      Update_Multiple_Dust_NoFeedback(pmb, dt, stopping_time, w, prim_df, u, cons_df);
  }

  return;
}
