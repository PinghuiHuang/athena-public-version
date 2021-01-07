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

// Perturbations
Real rho_gas_real,  rho_gas_imag,  velx_gas_real, velx_gas_imag;
Real vely_gas_real, vely_gas_imag, velz_gas_real, velz_gas_imag;
Real rho_dust_real[NDUSTFLUIDS],  rho_dust_imag[NDUSTFLUIDS];
Real velx_dust_real[NDUSTFLUIDS], velx_dust_imag[NDUSTFLUIDS];
Real vely_dust_real[NDUSTFLUIDS], vely_dust_imag[NDUSTFLUIDS];
Real velz_dust_real[NDUSTFLUIDS], velz_dust_imag[NDUSTFLUIDS];

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
  rhog0      = pin->GetOrAddReal("problem",    "rhog0",      1.0);
  nwx        = pin->GetOrAddInteger("problem", "nwx",        1);
  nwy        = pin->GetOrAddInteger("problem", "nwy",        1);
  nwz        = pin->GetOrAddInteger("problem", "nwz",        1);
  ShBoxCoord = pin->GetOrAddInteger("problem", "shboxcoord", 2);
  ipert      = pin->GetOrAddInteger("problem", "ipert",      1);
  Omega_0    = pin->GetOrAddReal("problem",    "Omega0",     1.0);
  qshear     = pin->GetOrAddReal("problem",    "qshear",     1.5);
  etaVk      = pin->GetOrAddReal("problem",    "etaVk",      0.05);
  iso_cs     = pin->GetReal("hydro",           "iso_sound_speed");
  //etaVk      *= iso_cs; // switch to code unit
  user_dt    = pin->GetOrAddReal("time",       "user_dt",    0.0);
  Kpar       = pin->GetOrAddReal("problem",    "Kpar",       30.0);

  kappap     = 2.0*(2.0 - qshear);
  kappap2    = SQR(kappap);
  Kai0       = 2.0*etaVk*iso_cs;

  //rho_gas_real  = pin->GetReal("hydro", "rho_real_gas");
  //rho_gas_imag  = pin->GetReal("hydro", "rho_imag_gas");
  //velx_gas_real = pin->GetReal("hydro", "velx_real_gas");
  //velx_gas_imag = pin->GetReal("hydro", "velx_imag_gas");
  //vely_gas_real = pin->GetReal("hydro", "vely_real_gas");
  //vely_gas_imag = pin->GetReal("hydro", "vely_imag_gas");
  //velz_gas_real = pin->GetReal("hydro", "velz_real_gas");
  //velz_gas_imag = pin->GetReal("hydro", "velz_imag_gas");

  if (NDUSTFLUIDS > 0) {
    for (int n=0; n<NDUSTFLUIDS; n++) {
      // Dust to gas ratio && dust stopping time
      initial_D2G[n] = pin->GetReal("dust", "initial_D2G_"      + std::to_string(n+1));
      Stokes[n]      = pin->GetReal("dust", "internal_density_" + std::to_string(n+1));

      //// Eigenvalues, Eigenvectors
      //rho_dust_real[n]  = pin->GetReal("dust", "rho_real_dust_"  + std::to_string(n+1));
      //rho_dust_imag[n]  = pin->GetReal("dust", "rho_imag_dust_"  + std::to_string(n+1));
      //velx_dust_real[n] = pin->GetReal("dust", "velx_real_dust_" + std::to_string(n+1));
      //velx_dust_imag[n] = pin->GetReal("dust", "velx_imag_dust_" + std::to_string(n+1));
      //vely_dust_real[n] = pin->GetReal("dust", "vely_real_dust_" + std::to_string(n+1));
      //vely_dust_imag[n] = pin->GetReal("dust", "vely_imag_dust_" + std::to_string(n+1));
      //velz_dust_real[n] = pin->GetReal("dust", "velz_real_dust_" + std::to_string(n+1));
      //velz_dust_imag[n] = pin->GetReal("dust", "velz_imag_dust_" + std::to_string(n+1));
    }
  }

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
  std::int64_t iseed = -1 - gid;
  volume.NewAthenaArray(ncells1);

  x1size = pmy_mesh->mesh_size.x1max - pmy_mesh->mesh_size.x1min;
  x2size = pmy_mesh->mesh_size.x2max - pmy_mesh->mesh_size.x2min;
  x3size = pmy_mesh->mesh_size.x3max - pmy_mesh->mesh_size.x3min;

  //Real kx = (TWO_PI/x1size)*(static_cast<Real>(nwx));
  //Real ky = (TWO_PI/x2size)*(static_cast<Real>(nwy));
  //Real kz = (TWO_PI/x3size)*(static_cast<Real>(nwz));
  //
  Real kx = Kpar * Omega_0/(etaVk * iso_cs);
  Real ky = 0.0;
  Real kz = Kpar * Omega_0/(etaVk * iso_cs);
  Real x_dis, y_dis, z_dis;
  //Real rd, rp, rval;
  //Real rvx, rvy, rvz;
  //std::int64_t iseed = -1-gid; // Initialize on the first call to ran2
  // Initialize perturbations
  // ipert = 1 - isentropic perturbations to P & d [default]
  // ipert = 2 - uniform Vx=amp, sinusoidal density

  if (block_size.nx3 > 1 || ShBoxCoord == 1) {
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          x_dis = pcoord->x1v(i);
          y_dis = pcoord->x2v(j);
          z_dis = pcoord->x3v(k);

          Real K_vel    = qshear*Omega_0*x_dis;
          Real gas_vel1 = AN*Kai0*Psi;
          Real gas_vel2 = -1.0*K_vel - 0.5*kappap2*BN*Kai0*Psi;
          Real gas_vel3 = 0.0;

          Real delta_gas_rho  = amp*rhog0*amp*(ran2(&iseed) - 0.5);
          Real delta_gas_vel1 = amp*etaVk*iso_cs*(ran2(&iseed) - 0.5);
          Real delta_gas_vel2 = amp*etaVk*iso_cs*(ran2(&iseed) - 0.5);
          Real delta_gas_vel3 = amp*etaVk*iso_cs*(ran2(&iseed) - 0.5);

          Real &gas_den  = phydro->u(IDN, k, j, i);
          Real &gas_mom1 = phydro->u(IM1, k, j, i);
          Real &gas_mom2 = phydro->u(IM2, k, j, i);
          Real &gas_mom3 = phydro->u(IM3, k, j, i);

          gas_den   = rhog0;
          gas_mom1  = gas_den * (gas_vel1 + delta_gas_vel1);
          gas_mom2  = gas_den * (gas_vel2 + delta_gas_vel2);
          gas_mom3  = gas_den * (gas_vel3 + delta_gas_vel3);
          gas_den  += delta_gas_rho;

          if (NDUSTFLUIDS > 0) {
            for (int n=0; n<NDUSTFLUIDS; n++) {
              int dust_id = n;
              int rho_id  = 4*dust_id;
              int v1_id   = rho_id + 1;
              int v2_id   = rho_id + 2;
              int v3_id   = rho_id + 3;

              Real delta_dust_rho  = amp*rhog0*amp*(ran2(&iseed) - 0.5);
              Real delta_dust_vel1 = amp*etaVk*iso_cs*(ran2(&iseed) - 0.5);
              Real delta_dust_vel2 = amp*etaVk*iso_cs*(ran2(&iseed) - 0.5);
              Real delta_dust_vel3 = amp*etaVk*iso_cs*(ran2(&iseed) - 0.5);

              Real dust_vel1  = (gas_vel1 + 2.0*Stokes[dust_id]*(gas_vel2 + K_vel))/(1.0 + kappap2*SQR(Stokes[dust_id]));
              Real dust_vel2  = -1.0 * K_vel;
              dust_vel2      += ((gas_vel2 + K_vel) - (2.0 - qshear)*Stokes[dust_id]*gas_vel1)/(1.0 + kappap2*SQR(Stokes[dust_id]));
              Real dust_vel3  = 0.0;

              Real &dust_den  = pdustfluids->df_cons(rho_id, k, j, i);
              Real &dust_mom1 = pdustfluids->df_cons(v1_id,  k, j, i);
              Real &dust_mom2 = pdustfluids->df_cons(v2_id,  k, j, i);
              Real &dust_mom3 = pdustfluids->df_cons(v3_id,  k, j, i);

              dust_den   = initial_D2G[dust_id] * rhog0;
              dust_mom1  = dust_den * (dust_vel1 + delta_dust_vel1);
              dust_mom2  = dust_den * (dust_vel2 + delta_dust_vel2);
              dust_mom3  = dust_den * (dust_vel3 + delta_dust_vel3);
              dust_den  += delta_dust_rho;
            }
          }

        }
      }
    }
  } else if (block_size.nx3 == 1 && ShBoxCoord == 2) {
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          x_dis = pcoord->x1v(i);
          z_dis = pcoord->x2v(j);
          y_dis = pcoord->x3v(k);

          //Real delta_gas_rho  = amp*rhog0*amp*(ran2(&iseed) - 0.5);
          //Real delta_gas_vel1 = amp*etaVk*iso_cs*(ran2(&iseed) - 0.5);
          //Real delta_gas_vel2 = amp*etaVk*iso_cs*(ran2(&iseed) - 0.5);
          //Real delta_gas_vel3 = amp*etaVk*iso_cs*(ran2(&iseed) - 0.5);

          Real delta_gas_rho  = 0.0;
          Real delta_gas_vel1 = 0.0;
          Real delta_gas_vel2 = 0.0;
          Real delta_gas_vel3 = 0.0;

          Real K_vel    = qshear*Omega_0*x_dis;
          Real gas_vel1 = AN*Kai0*Psi;
          Real gas_vel2 = 0.0;
          Real gas_vel3 = -1.0*K_vel - 0.5*kappap2*BN*Kai0*Psi;

          Real &gas_den  = phydro->u(IDN, k, j, i);
          Real &gas_mom1 = phydro->u(IM1, k, j, i);
          Real &gas_mom2 = phydro->u(IM2, k, j, i);
          Real &gas_mom3 = phydro->u(IM3, k, j, i);

          gas_den   = rhog0;
          gas_mom1  = gas_den * (gas_vel1 + delta_gas_vel1);
          gas_mom2  = gas_den * (gas_vel2 + delta_gas_vel2);
          gas_mom3  = gas_den * (gas_vel3 + delta_gas_vel3);
          gas_den  += delta_gas_rho;


          if (NDUSTFLUIDS > 0) {
            for (int n=0; n<NDUSTFLUIDS; n++) {
              int dust_id = n;
              int rho_id  = 4*dust_id;
              int v1_id   = rho_id + 1;
              int v2_id   = rho_id + 2;
              int v3_id   = rho_id + 3;

              Real delta_dust_rho  = amp*rhog0*amp*(ran2(&iseed) - 0.5);
              Real delta_dust_vel1 = amp*etaVk*iso_cs*(ran2(&iseed) - 0.5);
              Real delta_dust_vel2 = amp*etaVk*iso_cs*(ran2(&iseed) - 0.5);
              Real delta_dust_vel3 = amp*etaVk*iso_cs*(ran2(&iseed) - 0.5);

              Real dust_vel1  = (gas_vel1 + 2.0*Stokes[dust_id]*(gas_vel3 + K_vel))/(1.0 + kappap2*SQR(Stokes[dust_id]));
              Real dust_vel2  = 0.0;
              Real dust_vel3  = -1.0 * K_vel;
              dust_vel3      += ((gas_vel3 + K_vel) - (2.0 - qshear)*Stokes[dust_id]*gas_vel1)/(1.0 + kappap2*SQR(Stokes[dust_id]));

              Real &dust_den  = pdustfluids->df_cons(rho_id, k, j, i);
              Real &dust_mom1 = pdustfluids->df_cons(v1_id,  k, j, i);
              Real &dust_mom2 = pdustfluids->df_cons(v2_id,  k, j, i);
              Real &dust_mom3 = pdustfluids->df_cons(v3_id,  k, j, i);

              dust_den   = initial_D2G[dust_id] * rhog0;
              dust_mom1  = dust_den * (dust_vel1 + delta_dust_vel1);
              dust_mom2  = dust_den * (dust_vel2 + delta_dust_vel2);
              dust_mom3  = dust_den * (dust_vel3 + delta_dust_vel3);
              dust_den  += delta_dust_rho;
            }
          }

        }
      }
    }
  } else {
    std::cout << "[ShearingBoxSourceTerms]: not compatible to 1D !!" << std::endl;
  }
  return;
}

namespace {
void PressureGradient(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc, AthenaArray<Real> &cons)
{
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
  return;
}

Real MyTimeStep(MeshBlock *pmb)
{
  Real min_user_dt = user_dt;
  return min_user_dt;
}
}
