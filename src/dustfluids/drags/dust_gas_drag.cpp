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
  pmy_dustfluids_(pdf),
  drags_matrix(num_species, num_species), // The drags matrix
  aref_matrix(num_species,  num_species), // The matrix for iternative calculation
  lu_matrix(num_species,    num_species), // The LU decomposition matrix
  indx_array(num_species) {               // Stores the permutation

  integrator        = pin->GetOrAddString("time",  "integrator",      "vl2");

  Implicit_Flag     = pin->GetOrAddBoolean("dust", "Implicit_Flag",   false);
  Explicit_Flag     = pin->GetOrAddBoolean("dust", "Explicit_Flag",   false);
  drag_integrator   = pin->GetOrAddString("dust",  "drag_integrator", "vl2");
  DustFeedback_Flag = pin->GetBoolean("dust",      "DustFeedback_Flag");
}


void DustGasDrag::DragIntegrate(const int stage, const Real t_start, const Real dt,
      const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      AthenaArray<Real> &u, AthenaArray<Real> &cons_df)
{

  if (Implicit_Flag) {
    if (DustFeedback_Flag)
      BackwardEulerFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
    else
      BackwardEulerNoFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
  }
  else if (Explicit_Flag) {
    if (DustFeedback_Flag)
      ExplicitFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
    else
      ExplicitNoFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
  }
  else {
    if (integrator == "vl2") {
      if (DustFeedback_Flag)
        VL2ImplicitFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
        //BDF2Feedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
        //TRBDF2Feedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
      else
        VL2ImplicitNoFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
    }
    else if (integrator == "rk1") {
      if (DustFeedback_Flag)
        BackwardEulerFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
      else
        BackwardEulerNoFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
    }
    else if (integrator == "rk2") {
      if (DustFeedback_Flag)
        TrapezoidFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
      else
        TrapezoidFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
    }
    else {
      std::stringstream msg;
      msg << "Right now, the time integrator of dust-gas drag must be \"VL2\" or \"RK1\" or \"RK2\"!" << std::endl;
      ATHENA_ERROR(msg);
    }
  }
  return;
}
