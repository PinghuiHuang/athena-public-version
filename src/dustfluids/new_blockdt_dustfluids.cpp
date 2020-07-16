//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file new_blockdt.cpp
//  \brief computes timestep using CFL condition on a MEshBlock

// C headers

// C++ headers
#include <algorithm>  // min()
#include <cmath>      // fabs(), sqrt()
#include <limits>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../mesh/mesh.hpp"
#include "dustfluids.hpp"
#include "diffusions/dustfluids_diffusion.hpp"

// MPI/OpenMP header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

//----------------------------------------------------------------------------------------
// \!fn void DusstFluids::NewBlockTimeStep_Hyperbolic()
// \brief calculate the minimum timestep within a MeshBlock

Real DustFluids::NewAdvectionDt() {
  MeshBlock *pmb = pmy_block;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  AthenaArray<Real> &df_prim = pmb->pdustfluids->df_prim;
  // hyperbolic timestep constraint in each (x1-slice) cell along coordinate direction:
  AthenaArray<Real> &dt1 = dt1_, &dt2 = dt2_, &dt3 = dt3_;  // (x1 slices)
  const int num_dust_var = 4*NDUSTFLUIDS;
  Real df_prim_i[num_dust_var];

  Real real_max = std::numeric_limits<Real>::max();
  // Note, "dt_hyperbolic" currently refers to the dt limit imposed by evoluiton of the
  // ideal hydro or MHD fluid by the main integrator (even if not strictly hyperbolic)
  Real min_dt_hyperbolic_df  = real_max;

  FluidFormulation fluid_status = pmb->pmy_mesh->fluid_setup;
  for (int n=0; n<num_dust_var; n+=4){
    int dust_id = n/4;
    int rho_id  = 4*dust_id;
    int v1_id   = rho_id + 1;
    int v2_id   = rho_id + 2;
    int v3_id   = rho_id + 3;
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        pmb->pcoord->CenterWidth1(k, j, is, ie, dt1);
        pmb->pcoord->CenterWidth2(k, j, is, ie, dt2);
        pmb->pcoord->CenterWidth3(k, j, is, ie, dt3);
#pragma ivdep
          for (int i=is; i<=ie; ++i) {
            df_prim_i[rho_id] = df_prim(rho_id,k,j,i);
            df_prim_i[v1_id]  = df_prim(v1_id,k,j,i);
            df_prim_i[v2_id]  = df_prim(v2_id,k,j,i);
            df_prim_i[v3_id]  = df_prim(v3_id,k,j,i);

            //dt1(i) /= (std::abs(df_prim_i[v1_id]));
            //dt2(i) /= (std::abs(df_prim_i[v2_id]));
            //dt3(i) /= (std::abs(df_prim_i[v3_id]));
            if ((fluid_status == FluidFormulation::evolve) && SoundSpeed_Flag) {
              dt1(i) /= (std::abs(df_prim_i[v1_id]) + cs_dustfluids_array(dust_id,k,j,i));
              dt2(i) /= (std::abs(df_prim_i[v2_id]) + cs_dustfluids_array(dust_id,k,j,i));
              dt3(i) /= (std::abs(df_prim_i[v3_id]) + cs_dustfluids_array(dust_id,k,j,i));
            } else { // FluidFormulation::background or disabled. Assume scalar advection:
              dt1(i) /= (std::abs(df_prim_i[v1_id]));
              dt2(i) /= (std::abs(df_prim_i[v2_id]));
              dt3(i) /= (std::abs(df_prim_i[v3_id]));
            }
          }

        // compute minimum of (v1 +/- C)
        for (int i=is; i<=ie; ++i) {
          Real& dt_1 = dt1(i);
          min_dt_hyperbolic_df = std::min(min_dt_hyperbolic_df, dt_1);
        }

        // if grid is 2D/3D, compute minimum of (v2 +/- C)
        if (pmb->block_size.nx2 > 1) {
          for (int i=is; i<=ie; ++i) {
            Real& dt_2 = dt2(i);
            min_dt_hyperbolic_df = std::min(min_dt_hyperbolic_df, dt_2);
          }
        }

        // if grid is 3D, compute minimum of (v3 +/- C)
        if (pmb->block_size.nx3 > 1) {
          for (int i=is; i<=ie; ++i) {
            Real& dt_3 = dt3(i);
            min_dt_hyperbolic_df = std::min(min_dt_hyperbolic_df, dt_3);
          }
        }

      }
    }
  }

  min_dt_hyperbolic_df *= pmb->pmy_mesh->cfl_number;
  // scale the theoretical stability limit by a safety factor = the hyperbolic CFL limit
  // (user-selected or automaticlaly enforced). May add independent parameter "cfl_diff"
  // in the future (with default = cfl_number).

  // set main integrator timestep as the minimum of the appropriate timestep constraints:
  // hyperbolic: (skip if fluid is nonexistent or frozen)

  return min_dt_hyperbolic_df;
}
