//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file dust_gas_drag.cpp
//! Contains data and functions that implement physical (not coordinate) drag terms

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
  drags_matrix(NSPECIES, NSPECIES), // The drags matrix
  aref_matrix(NSPECIES,  NSPECIES), // The matrix for iternative calculation
  lu_matrix(NSPECIES,    NSPECIES), // The LU decomposition matrix
  indx_array(NSPECIES) {               // Stores the permutation

  integrator        = pin->GetOrAddString("time", "integrator", "vl2");
  DustFeedback_Flag = pin->GetBoolean("dust", "DustFeedback_Flag");

  drag_method = pin->GetOrAddString("dust", "drag_method", "2nd-implicit");

  if      (drag_method == "2nd-implicit")  drag_method_id = 1;
  else if (drag_method == "1st-implicit")  drag_method_id = 2;
  else if (drag_method == "semi-implicit") drag_method_id = 3;
  else if (drag_method == "explicit")      drag_method_id = 4;
  else                                     drag_method_id = 0;
}


void DustGasDrag::DragIntegrate(const int stage, const Real t_start, const Real dt,
      const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      AthenaArray<Real> &u, AthenaArray<Real> &cons_df)
{
  switch (drag_method_id) {
    case 1:
      if (integrator == "vl2") {
        if (DustFeedback_Flag)
          VL2ImplicitFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
        else
          VL2ImplicitNoFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
      }
      else if (integrator == "rk2") {
        if (DustFeedback_Flag)
          RK2ImplicitFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
        else
          RK2ImplicitNoFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
      }
      else {
        std::stringstream msg;
        msg << "The integrator combined with the 2nd-implicit methods must be \"VL2\" or \"RK2\"!" << std::endl;
        ATHENA_ERROR(msg);
      }
      break;

    case 2:
      if (integrator == "vl2") {
        if (DustFeedback_Flag)
          BDF2Feedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
        else
          BDF2NoFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
      }
      else {
        if (DustFeedback_Flag)
          BackwardEulerFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
        else
          BackwardEulerNoFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
      }
      break;

    case 3:
      if (integrator == "vl2") {
        if (DustFeedback_Flag)
          TRBDF2Feedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
        else
          TRBDF2NoFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
      }
      else if (integrator == "rk2") {
        if (DustFeedback_Flag)
          TrapezoidFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
        else
          TrapezoidNoFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
      }
      else {
        std::stringstream msg;
        msg << "The integrator combined with the semi-implicit methods must be \"VL2\" or \"RK2\"!" << std::endl;
        ATHENA_ERROR(msg);
      }
      break;

    case 4:
      if (DustFeedback_Flag)
        ExplicitFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
      else
        ExplicitNoFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
      break;

    default:
      std::stringstream msg;
      msg << "The drag-integrate method must be \"2nd-implicit\" or \"1st-implicit\" or \"semi-implicit\" or \"explicit\"!" << std::endl;
      ATHENA_ERROR(msg);
      break;
  }
  return;
}
