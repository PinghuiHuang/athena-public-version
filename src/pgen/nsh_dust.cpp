//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//  PRIVATE FUNCTION PROTOTYPES:
//  - ran2() - random number generator from NR
//
//  REFERENCE: Hawley, J. F. & Balbus, S. A., ApJ 400, 595-609 (1992).*/
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
#include "../eos/eos.hpp"
#include "../hydro/hydro.hpp"
#include "../dustfluids/dustfluids.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../utils/utils.hpp" // ran2()


#if !SHEARING_BOX
#error "This problem generator requires shearing box"
#endif

namespace {
Real amp, nwx, nwy, d0; // amplitude, Wavenumbers
Real etaVk; // The amplitude of pressure gradient force
int ShBoxCoord, ipert, ifield; // initial pattern
Real gm1, iso_cs;
Real x1size, x2size, x3size;
Real Omega_0, qshear;
Real pslope;
Real user_dt;
AthenaArray<Real> volume; // 1D array of volumes
Real HistoryBxBy(MeshBlock *pmb, int iout);
Real initial_D2G[NDUSTFLUIDS];
Real Stokes[NDUSTFLUIDS];
Real kappa, kappap, kappap2, AN(0.0), BN(0.0), Psi(0.0), Kai0;

// Perturbations
Real rho_gas_real,  rho_gas_imag,  vel1_gas_real, vel1_gas_imag;
Real vel2_gas_real, vel2_gas_imag, vel3_gas_real, vel3_gas_imag;
Real rho_dust_real[NDUSTFLUIDS],  rho_dust_imag[NDUSTFLUIDS];
Real vel1_dust_real[NDUSTFLUIDS], vel1_dust_imag[NDUSTFLUIDS];
Real vel2_dust_real[NDUSTFLUIDS], vel2_dust_imag[NDUSTFLUIDS];
Real vel3_dust_real[NDUSTFLUIDS], vel3_dust_imag[NDUSTFLUIDS];

// User Sources
void PressureGradient(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
Real MyTimeStep(MeshBlock *pmb);
} // namespace

//======================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Init the Mesh properties //======================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // initialize global variables
  amp        = pin->GetReal("problem",         "amp");
  d0         = pin->GetOrAddReal("problem",    "d0",         1.0);
  nwx        = pin->GetOrAddInteger("problem", "nwx",        1);
  nwy        = pin->GetOrAddInteger("problem", "nwy",        1);
  ShBoxCoord = pin->GetOrAddInteger("problem", "shboxcoord", 1);
  ipert      = pin->GetOrAddInteger("problem", "ipert",      1);
  Omega_0    = pin->GetOrAddReal("problem",    "Omega0",     0.001);
  qshear     = pin->GetOrAddReal("problem",    "qshear",     1.5);
  etaVk      = pin->GetOrAddReal("problem",    "etaVk",      0.05);
  iso_cs     = pin->GetReal("hydro", "iso_sound_speed");
  //etaVk      *= iso_cs; // switch to code unit
  user_dt    = pin->GetOrAddReal("time", "user_dt", 0.0);

  rho_gas_real  = pin->GetOrAddReal("problem", "rho_real_gas",  0.0);
  rho_gas_imag  = pin->GetOrAddReal("problem", "rho_imag_gas",  0.0);
  vel1_gas_real = pin->GetOrAddReal("problem", "vel1_real_gas", 0.0);
  vel1_gas_imag = pin->GetOrAddReal("problem", "vel1_imag_gas", 0.0);
  vel2_gas_real = pin->GetOrAddReal("problem", "vel2_real_gas", 0.0);
  vel2_gas_imag = pin->GetOrAddReal("problem", "vel2_imag_gas", 0.0);
  vel3_gas_real = pin->GetOrAddReal("problem", "vel3_real_gas", 0.0);
  vel3_gas_imag = pin->GetOrAddReal("problem", "vel3_imag_gas", 0.0);

  if (NDUSTFLUIDS > 0) {
    for (int n=0; n<NDUSTFLUIDS; n++) {
      rho_dust_real[n]  = pin->GetOrAddReal("problem", "rho_real_dust_"  + std::to_string(n+1), 0.0);
      rho_dust_imag[n]  = pin->GetOrAddReal("problem", "rho_imag_dust_"  + std::to_string(n+1), 0.0);
      vel1_dust_real[n] = pin->GetOrAddReal("problem", "vel1_real_dust_" + std::to_string(n+1), 0.0);
      vel1_dust_imag[n] = pin->GetOrAddReal("problem", "vel1_imag_dust_" + std::to_string(n+1), 0.0);
      vel2_dust_real[n] = pin->GetOrAddReal("problem", "vel2_real_dust_" + std::to_string(n+1), 0.0);
      vel2_dust_imag[n] = pin->GetOrAddReal("problem", "vel2_imag_dust_" + std::to_string(n+1), 0.0);
      vel3_dust_real[n] = pin->GetOrAddReal("problem", "vel3_real_dust_" + std::to_string(n+1), 0.0);
      vel3_dust_imag[n] = pin->GetOrAddReal("problem", "vel3_imag_dust_" + std::to_string(n+1), 0.0);
    }
  }

  // Dust to gas ratio && dust stopping time
  if (NDUSTFLUIDS > 0) {
    for (int n=0; n<NDUSTFLUIDS; n++) {
      initial_D2G[n] = pin->GetReal("dust", "initial_D2G_"      + std::to_string(n+1));
      Stokes[n]      = pin->GetReal("dust", "internal_density_" + std::to_string(n+1));
    }
  }

  kappap  = 2.0*(2.0 - qshear);
  kappap2 = SQR(kappap);
  Kai0    = 2.0*etaVk*iso_cs;

  // Dust to gas ratio && dust stopping time
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
//  \brief

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  volume.NewAthenaArray(ncells1);

  x1size = pmy_mesh->mesh_size.x1max - pmy_mesh->mesh_size.x1min;
  x2size = pmy_mesh->mesh_size.x2max - pmy_mesh->mesh_size.x2min;
  x3size = pmy_mesh->mesh_size.x3max - pmy_mesh->mesh_size.x3min;

  Real kx = (TWO_PI/x1size)*(static_cast<Real>(nwx));
  // Real kz = (TWO_PI/x2size)*(static_cast<Real>(nwy));
  Real x1, x2; //, x3;
  Real rd, rp, rval;
  Real rvx, rvy, rvz;
  std::int64_t iseed = -1-gid; // Initialize on the first call to ran2
  // Initialize perturbations
  //   ipert = 1 - isentropic perturbations to P & d [default]
  //   ipert = 2 - uniform Vx=amp, sinusoidal density

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        x1 = pcoord->x1v(i);
        x2 = pcoord->x2v(j);
        rvx = 0.0;
        rvy = 0.0;
        rvz = 0.0;

        Real K_vel = qshear*Omega_0*x1;

        Real gas_vel1 = AN*Kai0*Psi;
        Real gas_vel2 = -1.0*K_vel - 0.5*kappap2*BN*Kai0*Psi;
        Real gas_vel3 = 0.0;

        Real &gas_den  = phydro->u(IDN, k, j, i);
        Real &gas_mom1 = phydro->u(IM1, k, j, i);
        Real &gas_mom2 = phydro->u(IM2, k, j, i);
        Real &gas_mom3 = phydro->u(IM3, k, j, i);

        gas_den   = d0;
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

            Real dust_vel1  = (gas_vel1 + 2.0*Stokes[n]*(gas_vel2 + K_vel))/(1.0 + kappap2*SQR(Stokes[n]));
            Real dust_vel2  = -1.0 * K_vel;
            dust_vel2      += ((gas_vel2 + K_vel) - (2.0 - qshear)*Stokes[n]*gas_vel1)/(1.0 + kappap2*SQR(Stokes[n]));
            Real dust_vel3  = 0.0;

            Real &dust_den  = pdustfluids->df_cons(rho_id, k, j, i);
            Real &dust_mom1 = pdustfluids->df_cons(v1_id,  k, j, i);
            Real &dust_mom2 = pdustfluids->df_cons(v2_id,  k, j, i);
            Real &dust_mom3 = pdustfluids->df_cons(v3_id,  k, j, i);

            dust_den  = initial_D2G[dust_id] * gas_den;
            dust_mom1 = dust_den * dust_vel1;
            dust_mom2 = dust_den * dust_vel2;
            dust_mom3 = dust_den * dust_vel3;
          }
        }

      }
    }
  }
  return;
}

namespace {
void PressureGradient(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc, AthenaArray<Real> &cons)
{
  //DustFluids *pdf = pmb->pdustfluids;
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        const Real &gas_rho  = prim(IDN, k, j, i);
        Real press_gra       = gas_rho*Kai0*Omega_0*dt;
        Real &m1_gas         = cons(IM1, k, j, i);
        m1_gas              += press_gra;
      }
    }
  }

  //for (int n=0; n<NDUSTFLUIDS; n++) {
    //int dust_id = n;
    //int rho_id  = 4*dust_id;
    //int v1_id   = rho_id + 1;
    //int v2_id   = rho_id + 2;
    //int v3_id   = rho_id + 3;
    //for (int k=pmb->ks; k<=pmb->ke; ++k) {
      //for (int j=pmb->js; j<=pmb->je; ++j) {
//#pragma omp simd
        //for (int i=pmb->is; i<=pmb->ie; ++i) {
          //Real &dust_rho             = pdf->df_cons(rho_id,k,j,i);
          //Real press_gra             = dust_rho*Kai0*Omega_0*dt;
          //pdf->df_cons(v1_id,k,j,i) -= press_gra;
        //}
      //}
    //}
  //}

  return;
}

Real MyTimeStep(MeshBlock *pmb)
{
  Real min_user_dt = user_dt;
  return min_user_dt;
}
}
