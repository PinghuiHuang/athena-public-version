//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file sts_task_list.cpp
//  \brief RKL1 Super-Time-Stepping Algorithm
//
// REFERENCE:
// Meyer, C. D., Balsara, D. S., & Aslam, T. D. 2014, J. Comput. Phys.,
//    257A, 594-626

// C headers

// C++ headers
#include <cstring>    // strcmp()
#include <iomanip>    // std::setprecision()
#include <iostream>   // endl
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../bvals/bvals.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../field/field_diffusion/field_diffusion.hpp"
#include "../globals.hpp"
#include "../gravity/gravity.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/hydro_diffusion/hydro_diffusion.hpp"
#include "../hydro/srcterms/hydro_srcterms.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../reconstruct/reconstruction.hpp"
#include "../dustfluids/dustfluids.hpp"
#include "../dustfluids/srcterms/dustfluids_srcterms.hpp"
#include "../dustfluids/diffusions/dustfluids_diffusion.hpp"
#include "../dustfluids/drags/dust_gas_drag.hpp"
#include "task_list.hpp"

//----------------------------------------------------------------------------------------
//  SuperTimeStepTaskList constructor

SuperTimeStepTaskList::SuperTimeStepTaskList(
    ParameterInput *pin, Mesh *pm, TimeIntegratorTaskList *ptlist) :
    sts_max_dt_ratio(pin->GetOrAddReal("time", "sts_max_dt_ratio", -1.0)),
    ptlist_(ptlist) {
  // Check for STS incompatiblities:
  if (!(pm->pblock->phydro->hdif.hydro_diffusion_defined)
      // short-circuit evaluation makes these safe (won't dereference pdustfluids=nullptr):
      && !(MAGNETIC_FIELDS_ENABLED && pm->pblock->pfield->fdif.field_diffusion_defined)
      && !(NDUSTFLUIDS > 0 && pm->pblock->pdustfluids->dfdif.dustfluids_diffusion_defined)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in SuperTimeStepTaskList" << std::endl
        << "Super-time-stepping requires setting parameters for "
        << "at least one diffusive process in the input file." << std::endl;
    ATHENA_ERROR(msg);
  }
  // TODO(pdmullen): time-dep BC's require knowing the time within
  //                 an RKL1 operator-split STS, what is the time?
  if (SHEARING_BOX) {
    std::stringstream msg;
    msg << "### FATAL ERROR in SuperTimeStepTaskList" << std::endl
        << "Super-time-stepping is not yet compatible "
        << "with shearing box BC's." << std::endl;
    ATHENA_ERROR(msg);
  }
  // TODO(pdmullen): how should source terms be handled inside
  //                 operator-split RKL1 STS?
  if (pm->pblock->phydro->hsrc.hydro_sourceterms_defined) {
    std::stringstream msg;
    msg << "### FATAL ERROR in SuperTimeStepTaskList" << std::endl
        << "Super-time-stepping is not yet compatible "
        << "with source terms." << std::endl;
    ATHENA_ERROR(msg);
  }
  // TODO(pdmullen): fix non-Cartesian compatibility; this requires the
  //                 handling of coordinate source terms.
  if (std::strcmp(COORDINATE_SYSTEM, "cartesian") != 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in SuperTimeStepTaskList" << std::endl
        << "Super-time-stepping is not yet compatibile "
        << "with non-Cartesian coordinates." << std::endl;
    ATHENA_ERROR(msg);
  }
  // TODO(pdmullen): add mesh-refinement functionality
  if (pm->multilevel) {
    std::stringstream msg;
    msg << "### FATAL ERROR in SuperTimeStepTaskList" << std::endl
        << "Super-time-stepping is not yet compatibile "
        << "with mesh refinement." << std::endl;
    ATHENA_ERROR(msg);
  }

  // Now assemble list of tasks for each stage of SuperTimeStep integrator
  {using namespace HydroIntegratorTaskNames; // NOLINT (build/namespace)
    // calculate hydro/field diffusive fluxes
    AddTask(DIFFUSE_HYD,NONE);
    if (NDUSTFLUIDS > 0) {
      AddTask(DIFFUSE_DUSTFLUIDS,NONE);
    }
    // compute hydro fluxes, integrate hydro variables
    if (MAGNETIC_FIELDS_ENABLED) {
      AddTask(DIFFUSE_FLD,NONE);
      AddTask(CALC_HYDFLX,(DIFFUSE_HYD|DIFFUSE_FLD));
    } else {
      AddTask(CALC_HYDFLX,DIFFUSE_HYD);
    }
    AddTask(INT_HYD, CALC_HYDFLX);
    AddTask(SEND_HYD,INT_HYD);
    AddTask(RECV_HYD,NONE);
    AddTask(SETB_HYD,(RECV_HYD|INT_HYD));

    // compute dust fluids fluxes and integrate the density of each species:
    if (NDUSTFLUIDS > 0) {
      AddTask(CALC_DUSTFLUIDSFLX,(CALC_HYDFLX|DIFFUSE_DUSTFLUIDS));
      // TODO(felker): uncomment after S/AMR prohibitiion for STS is removed
       if (pm->multilevel) {
          AddTask(SEND_DUSTFLUIDSFLX,CALC_DUSTFLUIDSFLX);
          AddTask(RECV_DUSTFLUIDSFLX,CALC_DUSTFLUIDSFLX);
          AddTask(INT_DUSTFLUIDS,RECV_DUSTFLUIDSFLX);
       } else {
          AddTask(INT_DUSTFLUIDS, CALC_DUSTFLUIDSFLX);
        }
      AddTask(SEND_DUSTFLUIDS,INT_DUSTFLUIDS);
      AddTask(RECV_DUSTFLUIDS,NONE);
      AddTask(SETB_DUSTFLUIDS,(RECV_DUSTFLUIDS|INT_DUSTFLUIDS));
    }

    // compute MHD fluxes, integrate field
    if (MAGNETIC_FIELDS_ENABLED) { // MHD
      AddTask(CALC_FLDFLX,CALC_HYDFLX);
      AddTask(SEND_FLDFLX,CALC_FLDFLX);
      AddTask(RECV_FLDFLX,SEND_FLDFLX);
      AddTask(INT_FLD,RECV_FLDFLX);
      AddTask(SEND_FLD,INT_FLD);
      AddTask(RECV_FLD,NONE);
      AddTask(SETB_FLD,(RECV_FLD|INT_FLD));
      // compute new primitives
      if (NDUSTFLUIDS > 0) {
        AddTask(CONS2PRIM,(SETB_HYD|SETB_FLD|SETB_DUSTFLUIDS));
      } else {
        AddTask(CONS2PRIM,(SETB_HYD|SETB_FLD));
      }
    } else {  // Hydro
      if (NDUSTFLUIDS > 0) {
        AddTask(CONS2PRIM,(SETB_HYD|SETB_DUSTFLUIDS));
      } else {
        AddTask(CONS2PRIM,SETB_HYD);
      }
    }

    // everything else
    AddTask(PHY_BVAL,CONS2PRIM);
    AddTask(CLEAR_ALLBND,PHY_BVAL);
  } // end of using namespace block
}

//---------------------------------------------------------------------------------------
//  Sets id and dependency for "ntask" member of task_list_ array, then iterates value of
//  ntask.

void SuperTimeStepTaskList::AddTask(const TaskID& id, const TaskID& dep) {
  task_list_[ntasks].task_id = id;
  task_list_[ntasks].dependency = dep;

  using namespace HydroIntegratorTaskNames; // NOLINT (build/namespace)
  if (id == CLEAR_ALLBND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&TimeIntegratorTaskList::ClearAllBoundary);
    task_list_[ntasks].lb_time = false;
  } else if (id == CALC_HYDFLX) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&SuperTimeStepTaskList::CalculateHydroFlux_STS);
    task_list_[ntasks].lb_time = true;
  } else if (id == CALC_FLDFLX) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&SuperTimeStepTaskList::CalculateEMF_STS);
    task_list_[ntasks].lb_time = true;
  } else if (id == SEND_FLDFLX) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&TimeIntegratorTaskList::SendEMF);
    task_list_[ntasks].lb_time = true;
  } else if (id == RECV_FLDFLX) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&TimeIntegratorTaskList::ReceiveAndCorrectEMF);
    task_list_[ntasks].lb_time = false;
  } else if (id == INT_HYD) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&SuperTimeStepTaskList::IntegrateHydro_STS);
    task_list_[ntasks].lb_time = true;
  } else if (id == INT_FLD) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&SuperTimeStepTaskList::IntegrateField_STS);
    task_list_[ntasks].lb_time = true;
  } else if (id == SEND_HYD) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&TimeIntegratorTaskList::SendHydro);
    task_list_[ntasks].lb_time = true;
  } else if (id == SEND_FLD) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&TimeIntegratorTaskList::SendField);
    task_list_[ntasks].lb_time = true;
  } else if (id == RECV_HYD) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&TimeIntegratorTaskList::ReceiveHydro);
    task_list_[ntasks].lb_time = false;
  } else if (id == RECV_FLD) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&TimeIntegratorTaskList::ReceiveField);
    task_list_[ntasks].lb_time = false;
  } else if (id == SETB_HYD) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&TimeIntegratorTaskList::SetBoundariesHydro);
    task_list_[ntasks].lb_time = true;
  } else if (id == SETB_FLD) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&TimeIntegratorTaskList::SetBoundariesField);
    task_list_[ntasks].lb_time = true;
  } else if (id == CALC_DUSTFLUIDSFLX) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&SuperTimeStepTaskList::CalculateDustFluidsFlux_STS);
    task_list_[ntasks].lb_time = true;
  } else if (id == INT_DUSTFLUIDS) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&SuperTimeStepTaskList::IntegrateDustFluids_STS);
    task_list_[ntasks].lb_time = true;
  } else if (id == SEND_DUSTFLUIDS) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&TimeIntegratorTaskList::SendDustFluids);
    task_list_[ntasks].lb_time = true;
  } else if (id == RECV_DUSTFLUIDS) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&TimeIntegratorTaskList::ReceiveDustFluids);
    task_list_[ntasks].lb_time = false;
  } else if (id == SETB_DUSTFLUIDS) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&TimeIntegratorTaskList::SetBoundariesDustFluids);
    task_list_[ntasks].lb_time = true;
  } else if (id == CONS2PRIM) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&TimeIntegratorTaskList::Primitives);
    task_list_[ntasks].lb_time = true;
  } else if (id == PHY_BVAL) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&SuperTimeStepTaskList::PhysicalBoundary_STS);
    task_list_[ntasks].lb_time = true;
  } else if (id == DIFFUSE_HYD) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&TimeIntegratorTaskList::DiffuseHydro);
    task_list_[ntasks].lb_time = true;
  } else if (id == DIFFUSE_FLD) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&TimeIntegratorTaskList::DiffuseField);
    task_list_[ntasks].lb_time = true;
  } else if (id == DIFFUSE_DUSTFLUIDS) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&TimeIntegratorTaskList::DiffuseDustFluids);
    task_list_[ntasks].lb_time = true;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in AddTask" << std::endl
        << "Invalid Task is specified" << std::endl;
    ATHENA_ERROR(msg);
  }
  ntasks++;
  return;
}


void SuperTimeStepTaskList::StartupTaskList(MeshBlock *pmb, int stage) {
  Mesh *pm =  pmb->pmy_mesh;
#pragma omp single
  {
    // Set RKL1 params
    pm->muj = (2.*stage - 1.)/stage;
    pm->nuj = (1. - stage)/stage;
    pm->muj_tilde = pm->muj*2./(std::pow(nstages, 2.) + nstages);

    Real dt_ratio = pm->dt / pm->dt_parabolic;
    Real nstages_time_int = ptlist_->nstages;
    Real stage_ratio = nstages_time_int*dt_ratio/(nstages + nstages_time_int);

    if (Globals::my_rank == 0) {
      // check sanity of timestep ratio dt_hyperbolic/dt_parabolic (and hence STS nstages)
      if (sts_max_dt_ratio > 0 && dt_ratio > sts_max_dt_ratio) {
        std::stringstream msg;
        msg << "### FATAL ERROR in SuperTimeStepTaskList::StartupTaskList" << std::endl
            << "Ratio of dt/dt_parabolic = "<< dt_ratio << " exceeds the\n"
            << "specified limit for STS = " <<  sts_max_dt_ratio << std::endl;
        ATHENA_ERROR(msg);
      }
      // output additional diagnostics indiciating progress through STS stages:
      if (pm->dt_diagnostics != -1 && pm->ncycle_out != 0
          && pm->ncycle % pm->ncycle_out == 0) {
        const int ratio_precision = 3;
        const int dt_precision = std::numeric_limits<Real>::max_digits10 - 1;
        if (pm->dt_diagnostics == 0) {
          if (stage == nstages) {
            std::cout << "stage=" << stage << "/" << nstages
                      << " dt_parabolic=" << pm->dt_parabolic
                      << " ratio=" << std::setprecision(ratio_precision) <<  dt_ratio
                      << " stage_ratio=" << stage_ratio
                      << std::setprecision(dt_precision)
                      << std::endl;
          }
        } else {
          if (stage % pm->dt_diagnostics == 0) {
            std::cout << "stage=" << stage << "/" << nstages
                      << " dt_parabolic=" << pm->dt_parabolic
                      << " ratio=" << std::setprecision(ratio_precision) << dt_ratio
                      << " stage_ratio=" << stage_ratio
                      << std::setprecision(dt_precision)
                      << std::endl;
          }
        }
      }
    }
  }

  // Clear flux arrays from previous stage
  pmb->phydro->hdif.ClearFlux(pmb->phydro->flux);
  if (MAGNETIC_FIELDS_ENABLED)
    pmb->pfield->fdif.ClearEMF(pmb->pfield->e);
  if (NDUSTFLUIDS > 0) {
    DustFluids *pdf = pmb->pdustfluids;
    pdf->df_flux[X1DIR].ZeroClear();
    pdf->df_flux[X2DIR].ZeroClear();
    pdf->df_flux[X3DIR].ZeroClear();
  }

  // Real time = pm->time;
  pmb->pbval->StartReceiving(BoundaryCommSubset::all);

  return;
}

//----------------------------------------------------------------------------------------
// Functions to calculates fluxes

TaskStatus SuperTimeStepTaskList::CalculateHydroFlux_STS(MeshBlock *pmb, int stage) {
  Hydro *ph = pmb->phydro;
  if (stage <= nstages) {
    ph->CalculateFluxes_STS();
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}


TaskStatus SuperTimeStepTaskList::CalculateDustFluidsFlux_STS(MeshBlock *pmb, int stage) {
  DustFluids *pdf = pmb->pdustfluids;
  if (stage <= nstages) {
    pdf->CalculateDustFluidsFluxes_STS();
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}


TaskStatus SuperTimeStepTaskList::CalculateEMF_STS(MeshBlock *pmb, int stage) {
  Field *pf = pmb->pfield;
  if (stage <= nstages) {
    pf->ComputeCornerE_STS();
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

//----------------------------------------------------------------------------------------
// Functions to integrate conserved variables

TaskStatus SuperTimeStepTaskList::IntegrateHydro_STS(MeshBlock *pmb, int stage) {
  Hydro *ph = pmb->phydro;
  Field *pf = pmb->pfield;

  if (pmb->pmy_mesh->fluid_setup != FluidFormulation::evolve) return TaskStatus::next;

  // set registers
  ph->u2.SwapAthenaArray(ph->u1);
  ph->u1.SwapAthenaArray(ph->u);
  if (MAGNETIC_FIELDS_ENABLED) {
    pf->b2.x1f.SwapAthenaArray(pf->b1.x1f);
    pf->b2.x2f.SwapAthenaArray(pf->b1.x2f);
    pf->b2.x3f.SwapAthenaArray(pf->b1.x3f);

    pf->b1.x1f.SwapAthenaArray(pf->b.x1f);
    pf->b1.x2f.SwapAthenaArray(pf->b.x2f);
    pf->b1.x3f.SwapAthenaArray(pf->b.x3f);
  }

  // update u
  if (stage <= nstages) {
    Real ave_wghts[3];
    ave_wghts[0] = 0.0;
    ave_wghts[1] = pmb->pmy_mesh->muj;
    ave_wghts[2] = pmb->pmy_mesh->nuj;
    pmb->WeightedAve(ph->u, ph->u1, ph->u2, ave_wghts);
    const Real wght = pmb->pmy_mesh->muj_tilde*pmb->pmy_mesh->dt;
    ph->AddFluxDivergence(wght, ph->u);
    // TODO(pdmullen): check this after disabling ATHENA_ERROR for src terms
    pmb->pcoord->AddCoordTermsDivergence(wght, ph->flux, ph->w, pf->bcc, ph->u);
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}


TaskStatus SuperTimeStepTaskList::IntegrateDustFluids_STS(MeshBlock *pmb, int stage) {
  DustFluids *pdf = pmb->pdustfluids;
  // set registers
  pdf->df_cons2.SwapAthenaArray(pdf->df_cons1);
  pdf->df_cons1.SwapAthenaArray(pdf->df_cons);

  // update s
  if (stage <= nstages) {
    Real ave_wghts[3];
    ave_wghts[0] = 0.0;
    ave_wghts[1] = pmb->pmy_mesh->muj;
    ave_wghts[2] = pmb->pmy_mesh->nuj;
    pmb->WeightedAve(pdf->df_cons, pdf->df_cons1, pdf->df_cons2, ave_wghts);
    const Real wght = pmb->pmy_mesh->muj_tilde*pmb->pmy_mesh->dt;
    pdf->AddDustFluidsFluxDivergence(wght, pdf->df_cons);

    return TaskStatus::next;
  }
  return TaskStatus::fail;
}


TaskStatus SuperTimeStepTaskList::IntegrateField_STS(MeshBlock *pmb, int stage) {
  Field *pf = pmb->pfield;

  if (pmb->pmy_mesh->fluid_setup != FluidFormulation::evolve) return TaskStatus::next;

  if (stage <= nstages) {
    Real ave_wghts[3];
    ave_wghts[0] = 0.0;
    ave_wghts[1] = pmb->pmy_mesh->muj;
    ave_wghts[2] = pmb->pmy_mesh->nuj;
    pmb->WeightedAve(pf->b, pf->b1, pf->b2, ave_wghts);
    pf->CT(pmb->pmy_mesh->muj_tilde*pmb->pmy_mesh->dt, pf->b);
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

TaskStatus SuperTimeStepTaskList::PhysicalBoundary_STS(MeshBlock *pmb, int stage) {
  BoundaryValues *pbval=pmb->pbval;
  if (stage <= nstages) {
    // TODO(pdmullen): for time-dependent BC's, what is the time inside of an
    //                 operator-split RKL1 STS? For now, disable time-dep BCs.
    // Real t_end_stage = pmb->pmy_mesh->time;
    // Real dt = pmb->pmy_mesh->dt;
    pmb->phydro->hbvar.SwapHydroQuantity(pmb->phydro->w, HydroBoundaryQuantity::prim);
    pbval->ApplyPhysicalBoundaries(pmb->pmy_mesh->time, pmb->pmy_mesh->dt);
  } else {
    return TaskStatus::fail;
  }

  return TaskStatus::success;
}
