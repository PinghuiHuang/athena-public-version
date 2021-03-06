//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file visc.cpp
//  iprob = 0 - test viscous shear flow density column in various coordinate systems
//  iprob = 1 - test viscous spreading of Keplerain ring

// C headers

// C++ headers
#include <algorithm>  // min
#include <cmath>      // sqrt()
#include <cstdlib>    // srand
#include <cstring>    // strcmp()
#include <fstream>
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../dustfluids/dustfluids.hpp"

#if NON_BAROTROPIC_EOS
#error "This problem generator requires isothermal equation of state!"
#endif

// problem parameters which are useful to make global to this file
namespace {
Real v0, t0, x0, d0, v1, v2, v3;
Real nuiso, gm0, iso_cs;
Real A0, sig_x1, sig_x2, cen1, cen2;
int iprob;
Real TotalMomentum1(MeshBlock *pmb, int iout);
Real TotalMomentum2(MeshBlock *pmb, int iout);
Real TotalMomentum3(MeshBlock *pmb, int iout);
} // namespace

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Get parameters for gravitatonal potential of central point mass
  iso_cs = pin->GetOrAddReal("hydro",   "iso_sound_speed", 1e-1);
  iprob  = pin->GetOrAddInteger("problem", "iprob",    0);
  v1     = pin->GetOrAddReal("problem",    "v1",       0.001);
  v2     = pin->GetOrAddReal("problem",    "v2",       0.001);
  v3     = pin->GetOrAddReal("problem",    "v3",       0.001);
  x0     = pin->GetOrAddReal("problem",    "x0",       0.0);
  d0     = pin->GetOrAddReal("problem",    "d0",       1.0);
  cen1   = pin->GetOrAddReal("problem",    "cen1",     0.0);
  cen2   = pin->GetOrAddReal("problem",    "cen2",     0.0);
  t0     = pin->GetOrAddReal("problem",    "t0",       0.5);
  nuiso  = pin->GetOrAddReal("problem",    "nu_iso",   0.0);
  gm0    = pin->GetOrAddReal("problem",    "GM",       0.0);
  A0     = pin->GetOrAddReal("problem",    "A0",       1.0);
  sig_x1 = pin->GetOrAddReal("problem",    "sigma_x1", 0.2);
  sig_x2 = pin->GetOrAddReal("problem",    "sigma_x2", 0.2);

  if (NDUSTFLUIDS == 1) {
    AllocateUserHistoryOutput(3);
    EnrollUserHistoryOutput(0, TotalMomentum1, "tot-mom1");
    EnrollUserHistoryOutput(1, TotalMomentum2, "tot-mom2");
    EnrollUserHistoryOutput(2, TotalMomentum3, "tot-mom3");
  }
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Initializes viscous shear flow.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  //Real rad, phi, z;
  //Real v1 = 0.0, v2 = 0.0, v3 = 0.0;
  Real x1, x2, x3; // x2 and x3 are set but unused
  Real rad,z,phi,theta;

  //  Initialize density and momenta in Cartesian grids
  if (iprob == 0) { //visc column
    if (nuiso == 0) nuiso = 0.03;
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
            x1 = pcoord->x1v(i);
            x2 = pcoord->x2v(j);
            x3 = pcoord->x3v(k);
            //v1 = v0;
            //v2 = v0;
            Real &gas_den  = phydro->u(IDN, k, j, i);
            Real &gas_mom1 = phydro->u(IM1, k, j, i);
            Real &gas_mom2 = phydro->u(IM2, k, j, i);
            Real &gas_mom3 = phydro->u(IM3, k, j, i);

            gas_den  = d0;
            gas_mom1 = gas_den*v1;
            gas_mom2 = gas_den*v2;
            gas_mom3 = gas_den*v3;

            if (NDUSTFLUIDS > 0) {
              Real concentration = 0.0;
              if (sig_x2 > 0.0)
                concentration = A0*std::exp(-SQR(x1-x0)/(2.0*SQR(sig_x1)) -SQR(x2-x0)/(2.0*SQR(sig_x2)));
              else
                concentration = A0*std::exp(-SQR(x1-x0)/(2.0*SQR(sig_x1)));
              //Real concentration = (x1 <= 0.2 && x1 >= -0.2 && x2 <=0.2 && x2 >=-0.2) ? A0 : 0.0;
              //Real concentration = 1.0;
              for (int n=0; n<NDUSTFLUIDS; n++) {
                int dust_id = n;
                int rho_id  = 4*dust_id;
                int v1_id   = rho_id + 1;
                int v2_id   = rho_id + 2;
                int v3_id   = rho_id + 3;

                Real &dust_den  = pdustfluids->df_cons(rho_id, k, j, i);
                Real &dust_mom1 = pdustfluids->df_cons(v1_id, k, j, i);
                Real &dust_mom2 = pdustfluids->df_cons(v2_id, k, j, i);
                Real &dust_mom3 = pdustfluids->df_cons(v3_id, k, j, i);

                dust_den  = concentration*gas_den;
                dust_mom1 = dust_den * v1;
                dust_mom2 = dust_den * v2;
                dust_mom3 = dust_den * v3;
              }
            }
          } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
            rad = pcoord->x1v(i);
            phi = pcoord->x2v(j);
            z   = pcoord->x3v(k);
            x1  = rad*std::cos(phi);
            x2  = rad*std::sin(phi);
            x3  = z;
            //v2 = v0/std::sqrt(4.0*PI*nuiso*t0)*std::exp(-SQR(x1-x0)/(4.0*nuiso*t0));
            v2 = 0.0;

            Real &gas_den  = phydro->u(IDN, k, j, i);
            Real &gas_mom1 = phydro->u(IM1, k, j, i);
            Real &gas_mom2 = phydro->u(IM2, k, j, i);
            Real &gas_mom3 = phydro->u(IM3, k, j, i);

            gas_den  = A0*std::exp(-SQR(x1-cen1)/(2.0*SQR(sig_x1)) -SQR(x2-cen2)/(2.0*SQR(sig_x2)));
            gas_mom1 = gas_den*(v1*std::cos(phi) + v2*std::sin(phi));
            gas_mom2 = gas_den*(-v1*std::sin(phi) + v2*std::cos(phi));
            gas_mom3 = gas_den*v3;

            if (NDUSTFLUIDS > 0) {
              //Real concentration = A0*std::exp(-SQR(x1-cen1)/(2.0*SQR(sig_x1)) -SQR(x2-cen2)/(2.0*SQR(sig_x2)));
              //Real concentration = (x1 <= 0.2 && x1 >= -0.2 && x2 <=0.2 && x2 >=-0.2) ? A0 : 0.0;
              Real concentration = 0.5;
              for (int n=0; n<NDUSTFLUIDS; n++) {
                int dust_id = n;
                int rho_id  = 4*dust_id;
                int v1_id   = rho_id + 1;
                int v2_id   = rho_id + 2;
                int v3_id   = rho_id + 3;

                Real &dust_den  = pdustfluids->df_cons(rho_id, k, j, i);
                Real &dust_mom1 = pdustfluids->df_cons(v1_id, k, j, i);
                Real &dust_mom2 = pdustfluids->df_cons(v2_id, k, j, i);
                Real &dust_mom3 = pdustfluids->df_cons(v3_id, k, j, i);

                dust_den  = concentration*gas_den;
                dust_mom1 = dust_den * v1;
                dust_mom2 = dust_den * v2;
                dust_mom3 = dust_den * v3;
              }
            }
          } else { // (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
            rad   = std::abs(pcoord->x1v(i)*std::sin(pcoord->x2v(j)));
            phi   = pcoord->x3v(k);
            theta = pcoord->x2v(j);
            z     = pcoord->x1v(i)*std::cos(pcoord->x2v(j));
            x1    = rad*std::cos(phi);
            x2    = rad*std::sin(phi);
            x3    = z;
            Real vel2    = v1/std::sqrt(4.0*PI*nuiso*t0)*std::exp(-SQR(x1-x0)/(4.0*nuiso*t0));
            phydro->u(IDN, k, j, i) = d0;
            phydro->u(IM1, k, j, i) = phydro->u(IDN, k, j, i)*(v1*std::cos(phi)*std::sin(theta)
                                                              +vel2*std::sin(phi)*std::sin(theta)
                                                              +v3*std::cos(theta));
            phydro->u(IM2, k, j, i) = phydro->u(IDN, k, j, i)*(v1*std::cos(phi)*std::cos(theta)
                                                              +vel2*std::sin(phi)*std::cos(theta)
                                                              +v3*std::sin(theta));
            phydro->u(IM3, k, j, i) = phydro->u(IDN, k, j, i)*(-v1*std::sin(phi) + vel2*std::cos(phi));
          }
        }
      }
    }
  } else if (iprob == 1) { // visc ring gaussian profile at t=0
    if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") != 0 && gm0 == 0.0) {
      std::stringstream msg;
      msg << "### FATAL ERROR in visc.cpp ProblemGenerator" << std::endl
          << "viscous ring test only compatible with cylindrical coord"
          << std::endl << "with point mass in center" << std::endl;
      ATHENA_ERROR(msg);
    }
    Real width = 0.1;
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          rad = pcoord->x1v(i);
          Real dens0 = std::exp(-SQR(rad-x0)/2.0/SQR(width));
          if (dens0 < 1e-6) dens0 += 1e-6;
          Real vel1 = -3.0*nuiso*(0.5/rad - (rad-x0)/SQR(width));
          Real vel2 = std::sqrt(gm0/rad); // set it to be Mach=100
          phydro->u(IDN, k, j, i) = dens0;
          phydro->u(IM1, k, j, i) = phydro->u(IDN, k, j, i)*vel1;
          phydro->u(IM2, k, j, i) = phydro->u(IDN, k, j, i)*vel2;
          phydro->u(IM3, k, j, i) = phydro->u(IDN, k, j, i)*v3;
        }
      }
    }
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in visc.cpp ProblemGenerator" << std::endl
        << "viscous iprob has to be either 0 or 1" << std::endl;
    ATHENA_ERROR(msg);
  }
  return;
}


namespace {
Real TotalMomentum1(MeshBlock *pmb, int iout) {
  Real total_mom = 0;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> &s = pmb->pdustfluids->df_cons;
  AthenaArray<Real> &u = pmb->phydro->u;
  AthenaArray<Real> volume; // 1D array of volumes
  // allocate 1D array for cell volume used in usr def history
  volume.NewAthenaArray(pmb->ncells1);

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, volume);
      for (int i=is; i<=ie; i++) {
        total_mom += volume(i)*u(IM1,k,j,i);
      }
    }
  }

  for (int n=0; n<NDUSTFLUIDS; n++) {
    int dust_id = n;
    int rho_id  = 4*dust_id;
    int v1_id   = rho_id + 1;
    int v2_id   = rho_id + 2;
    int v3_id   = rho_id + 3;
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, volume);
        for (int i=is; i<=ie; i++) {
          total_mom += volume(i)*s(v1_id,k,j,i);
        }
      }
    }
  }
  return total_mom;
}


Real TotalMomentum2(MeshBlock *pmb, int iout) {
  Real total_mom = 0;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> &s = pmb->pdustfluids->df_cons;
  AthenaArray<Real> &u = pmb->phydro->u;
  AthenaArray<Real> volume; // 1D array of volumes
  // allocate 1D array for cell volume used in usr def history
  volume.NewAthenaArray(pmb->ncells1);

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, volume);
      for (int i=is; i<=ie; i++) {
        total_mom += volume(i)*u(IM2,k,j,i);
      }
    }
  }

  for (int n=0; n<NDUSTFLUIDS; n++) {
    int dust_id = n;
    int rho_id  = 4*dust_id;
    int v1_id   = rho_id + 1;
    int v2_id   = rho_id + 2;
    int v3_id   = rho_id + 3;
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, volume);
        for (int i=is; i<=ie; i++) {
          total_mom += volume(i)*s(v2_id,k,j,i);
        }
      }
    }
  }
  return total_mom;
}


Real TotalMomentum3(MeshBlock *pmb, int iout) {
  Real total_mom = 0;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> &s = pmb->pdustfluids->df_cons;
  AthenaArray<Real> &u = pmb->phydro->u;
  AthenaArray<Real> volume; // 1D array of volumes
  // allocate 1D array for cell volume used in usr def history
  volume.NewAthenaArray(pmb->ncells1);

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, volume);
      for (int i=is; i<=ie; i++) {
        total_mom += volume(i)*u(IM3,k,j,i);
      }
    }
  }

  for (int n=0; n<NDUSTFLUIDS; n++) {
    int dust_id = n;
    int rho_id  = 4*dust_id;
    int v1_id   = rho_id + 1;
    int v2_id   = rho_id + 2;
    int v3_id   = rho_id + 3;
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, volume);
        for (int i=is; i<=ie; i++) {
          total_mom += volume(i)*s(v3_id,k,j,i);
        }
      }
    }
  }
  return total_mom;
}
} // namespace
