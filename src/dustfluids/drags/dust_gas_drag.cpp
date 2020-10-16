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
  DustFeedback_Flag = pin->GetBoolean("dust",     "DustFeedback_Flag");
  integrator        = pin->GetOrAddString("time", "integrator", "vl2");
}


void DustGasDrag::AerodynamicDrag(MeshBlock *pmb, const int stage, const Real dt,
      const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      AthenaArray<Real> &u, AthenaArray<Real> &cons_df)
{
  if (integrator == "vl2") {
    if (DustFeedback_Flag) {
      SingleDustFeedbackImplicit(pmb, stage, dt, stopping_time, w, prim_df, u, cons_df);
      //VL2ImplicitFeedback(pmb, stage, dt, stopping_time, w, prim_df, u, cons_df);
    }
    else {
      SingleDustNoFeedbackImplicit(pmb, stage, dt, stopping_time, w, prim_df, u, cons_df);
      //VL2ImplicitFeedback(pmb, stage, dt, stopping_time, w, prim_df, u, cons_df);
    }
  }
  else if ( (integrator == "rk2") || (integrator == "rk1") ) {
    if (DustFeedback_Flag)
      RK2ImplicitFeedback(pmb, stage, dt, stopping_time, w, prim_df, u, cons_df);
    else
      RK2ImplicitFeedback(pmb, stage, dt, stopping_time, w, prim_df, u, cons_df);
      //RK2ImplicitNoFeedback(pmb, stage, dt, stopping_time, w, prim_df, u, cons_df);
  }
  else {
    std::stringstream msg;
    msg << "Right now, the time integrator of dust fluids must be \"RK1\" or \"RK2\" or \"VL2\"!" << std::endl;
    ATHENA_ERROR(msg);
  }

  return;
}
