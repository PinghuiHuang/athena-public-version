//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//  PRIVATE FUNCTION PROTOTYPES:
//  - ran2() - random number generator from NR
//
//  REFERENCE: NSH equilibrium, Nakagawa-Sekiya-Hayashi, 1986
//
//======================================================================================

// C headers

// C++ headers
#include <cmath>      // sqrt()
#include <iostream>   // cout, endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../eos/eos.hpp"
#include "../hydro/hydro.hpp"
#include "../dustfluids/dustfluids.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../utils/utils.hpp" // ran2()


namespace {
Real amp, nwx, nwy, nwz, rhog0; // amplitude, Wavenumbers
Real etaVk; // The amplitude of pressure gradient force
int ShBoxCoord, ipert, ifield; // initial pattern
Real gm1, iso_cs;
Real x1size, x2size, x3size;
Real Omega_0, qshear;
Real pslope;
Real user_dt;
AthenaArray<Real> volume; // 1D array of volumes
Real initial_D2G[NDUSTFLUIDS];
Real Stokes[NDUSTFLUIDS];
Real kappa, kappap, kappap2, AN(0.0), BN(0.0), Psi(0.0), Kai0;
Real Kpar;

// User Sources
void PressureGradient(MeshBlock *pmb, const Real time, const Real dt,
    const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_df,
    const AthenaArray<Real> &bcc, AthenaArray<Real> &cons, AthenaArray<Real> &cons_df);
Real MyTimeStep(MeshBlock *pmb);
} // namespace

//======================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Init the Mesh properties //======================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // initialize global variables
  user_dt    = pin->GetOrAddReal("time", "user_dt", 0.0);
  amp        = pin->GetReal("problem", "amp");
  rhog0      = pin->GetOrAddReal("problem", "rhog0", 1.0);
  nwx        = pin->GetOrAddInteger("problem", "nwx",   1);
  nwy        = pin->GetOrAddInteger("problem", "nwy",   1);
  nwz        = pin->GetOrAddInteger("problem", "nwz",   1);
  ipert      = pin->GetOrAddInteger("problem", "ipert", 1);
  etaVk      = pin->GetOrAddReal("problem", "etaVk", 0.05);
  iso_cs     = pin->GetReal("hydro", "iso_sound_speed");
  //etaVk      *= iso_cs; // switch to code unit
  Kpar       = pin->GetOrAddReal("problem", "Kpar",    30.0);

  ShBoxCoord = pin->GetOrAddInteger("orbital_advection", "shboxcoord", 1);
  Omega_0    = pin->GetOrAddReal("orbital_advection",    "Omega0",     1.0);
  qshear     = pin->GetOrAddReal("orbital_advection",    "qshear",     1.5);

  if (NDUSTFLUIDS > 0) {
    for (int n=0; n<NDUSTFLUIDS; n++) {
      // Dust to gas ratio && dust stopping time
      initial_D2G[n] = pin->GetReal("dust", "initial_D2G_"      + std::to_string(n+1));
      Stokes[n]      = pin->GetReal("dust", "internal_density_" + std::to_string(n+1));
    }
  }

  kappap  = 2.0*(2.0 - qshear);
  kappap2 = SQR(kappap);
  Kai0    = 2.0*etaVk*iso_cs;

  if (NDUSTFLUIDS > 0) {
    for (int n=0; n<NDUSTFLUIDS; n++) {
      AN += (initial_D2G[n] * Stokes[n])/(1.0 + kappap2 * SQR(Stokes[n]));
      BN += (initial_D2G[n])/(1.0 + kappap2 * SQR(Stokes[n]));
    }
    AN *= kappap2;
    BN += 1.0;
    Psi = 1.0/(SQR(AN) + kappap2*SQR(BN));
  }

  EnrollUserExplicitSourceFunction(PressureGradient);
  if (user_dt > 0.0)
    EnrollUserTimeStepFunction(MyTimeStep);

  return;
}

//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  volume.NewAthenaArray(ncells1);

  x1size = pmy_mesh->mesh_size.x1max - pmy_mesh->mesh_size.x1min;
  x2size = pmy_mesh->mesh_size.x2max - pmy_mesh->mesh_size.x2min;
  x3size = pmy_mesh->mesh_size.x3max - pmy_mesh->mesh_size.x3min;

  Real x_dis, y_dis, z_dis;

  if (ShBoxCoord == 1) {
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          x_dis = pcoord->x1v(i);
          y_dis = pcoord->x2v(j);
          z_dis = pcoord->x3v(k);

          Real K_vel    = qshear*Omega_0*x_dis;
          Real gas_vel1 = AN*Kai0*Psi;
          Real gas_vel2 = 0.0;
          if(!porb->orbital_advection_defined)
            gas_vel2 = -1.0*K_vel - 0.5*kappap2*BN*Kai0*Psi;
          else
            gas_vel2 = -0.5*kappap2*BN*Kai0*Psi;
          Real gas_vel3 = 0.0;

          Real &gas_den  = phydro->u(IDN, k, j, i);
          Real &gas_mom1 = phydro->u(IM1, k, j, i);
          Real &gas_mom2 = phydro->u(IM2, k, j, i);
          Real &gas_mom3 = phydro->u(IM3, k, j, i);

          gas_den  = rhog0;
          gas_mom1 = gas_den * gas_vel1;
          gas_mom2 = gas_den * gas_vel2;
          gas_mom3 = gas_den * gas_vel3;

          if (NDUSTFLUIDS > 0) {
            for (int n=0; n<NDUSTFLUIDS; n++) {
              int dust_id = n;
              int rho_id  = 4*dust_id;
              int v1_id   = rho_id + 1;
              int v2_id   = rho_id + 2;
              int v3_id   = rho_id + 3;

              Real dust_vel1 = 0.0;
              Real dust_vel2 = 0.0;
              Real dust_vel3 = 0.0;

              if(!porb->orbital_advection_defined) { // orbital advection turns off
                dust_vel1 = (gas_vel1 + 2.0*Stokes[dust_id]*(gas_vel2 + K_vel))/(1.0 + kappap2*SQR(Stokes[dust_id]));
                dust_vel2 = -1.0 * K_vel + ((gas_vel2 + K_vel) - (2.0 - qshear)*Stokes[dust_id]*gas_vel1)/(1.0 + kappap2*SQR(Stokes[dust_id]));
                dust_vel3 = 0.0;
              } else { // orbital advection truns on
                dust_vel1 = (gas_vel1 + 2.0*Stokes[dust_id]*gas_vel2)/(1.0 + kappap2*SQR(Stokes[dust_id]));
                dust_vel2 = (gas_vel2 - (2.0 - qshear)*Stokes[dust_id]*gas_vel1)/(1.0 + kappap2*SQR(Stokes[dust_id]));
                dust_vel3 = 0.0;
              }

              Real &dust_den  = pdustfluids->df_cons(rho_id, k, j, i);
              Real &dust_mom1 = pdustfluids->df_cons(v1_id,  k, j, i);
              Real &dust_mom2 = pdustfluids->df_cons(v2_id,  k, j, i);
              Real &dust_mom3 = pdustfluids->df_cons(v3_id,  k, j, i);

              dust_den   = initial_D2G[dust_id] * rhog0;
              dust_mom1  = dust_den * dust_vel1;
              dust_mom2  = dust_den * dust_vel2;
              dust_mom3  = dust_den * dust_vel3;
            }
          }
        }
      }
    }
  } else {
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          x_dis = pcoord->x1v(i);
          z_dis = pcoord->x2v(j);
          y_dis = pcoord->x3v(k);

          Real K_vel    = qshear*Omega_0*x_dis;
          Real gas_vel1 = AN*Kai0*Psi;
          Real gas_vel2 = 0.0;
          Real gas_vel3 = 0.0;
          if(!porb->orbital_advection_defined)
            gas_vel3 = -1.0*K_vel - 0.5*kappap2*BN*Kai0*Psi;
          else
            gas_vel3 = -0.5*kappap2*BN*Kai0*Psi;

          Real &gas_den  = phydro->u(IDN, k, j, i);
          Real &gas_mom1 = phydro->u(IM1, k, j, i);
          Real &gas_mom2 = phydro->u(IM2, k, j, i);
          Real &gas_mom3 = phydro->u(IM3, k, j, i);

          gas_den   = rhog0;
          gas_mom1  = gas_den * gas_vel1;
          gas_mom2  = gas_den * gas_vel2;
          gas_mom3  = gas_den * gas_vel3;

          if (NDUSTFLUIDS > 0) {
            for (int n=0; n<NDUSTFLUIDS; n++) {
              int dust_id = n;
              int rho_id  = 4*dust_id;
              int v1_id   = rho_id + 1;
              int v2_id   = rho_id + 2;
              int v3_id   = rho_id + 3;

              Real dust_vel1 = 0.0;
              Real dust_vel2 = 0.0;
              Real dust_vel3 = 0.0;

              if(!porb->orbital_advection_defined) { // orbital advection turns off
                dust_vel1 = (gas_vel1 + 2.0*Stokes[dust_id]*(gas_vel3 + K_vel))/(1.0 + kappap2*SQR(Stokes[dust_id]));
                dust_vel2 = 0.0;
                dust_vel3 = -1.0 * K_vel + ((gas_vel3 + K_vel) - (2.0 - qshear)*Stokes[dust_id]*gas_vel1)/(1.0 + kappap2*SQR(Stokes[dust_id]));
              } else { // orbital advection truns on
                dust_vel1 = (gas_vel1 + 2.0*Stokes[dust_id]*gas_vel3)/(1.0 + kappap2*SQR(Stokes[dust_id]));
                dust_vel2 = 0.0;
                dust_vel3 = (gas_vel3 - (2.0 - qshear)*Stokes[dust_id]*gas_vel1)/(1.0 + kappap2*SQR(Stokes[dust_id]));
              }

              Real &dust_den  = pdustfluids->df_cons(rho_id, k, j, i);
              Real &dust_mom1 = pdustfluids->df_cons(v1_id,  k, j, i);
              Real &dust_mom2 = pdustfluids->df_cons(v2_id,  k, j, i);
              Real &dust_mom3 = pdustfluids->df_cons(v3_id,  k, j, i);

              dust_den   = initial_D2G[dust_id] * rhog0;
              dust_mom1  = dust_den * dust_vel1;
              dust_mom2  = dust_den * dust_vel2;
              dust_mom3  = dust_den * dust_vel3;
            }
          }

        }
      }
    }
  }
  return;
}


namespace {
void PressureGradient(MeshBlock *pmb, const Real time, const Real dt,
    const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_df,
    const AthenaArray<Real> &bcc, AthenaArray<Real> &cons, AthenaArray<Real> &cons_df) {

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real press_gra       = rhog0*Kai0*Omega_0*dt;
        Real &m1_gas         = cons(IM1, k, j, i);
        m1_gas              += press_gra;
      }
    }
  }
  return;
}

Real MyTimeStep(MeshBlock *pmb)
{
  Real min_user_dt = user_dt;
  return min_user_dt;
}

}
