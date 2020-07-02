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

// problem parameters which are useful to make global to this file
namespace {
Real v0, t0, x0;
Real nuiso, gm0;
Real A0, sig_x1, sig_x2, cen1, cen2;
int iprob;
} // namespace

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Get parameters for gravitatonal potential of central point mass
  v0     = pin->GetOrAddReal("problem",    "v0",       0.001);
  x0     = pin->GetOrAddReal("problem",    "x0",       0.0);
  cen1   = pin->GetOrAddReal("problem",    "cen1",     0.0);
  cen2   = pin->GetOrAddReal("problem",    "cen2",     0.0);
  t0     = pin->GetOrAddReal("problem",    "t0",       0.5);
  nuiso  = pin->GetOrAddReal("problem",    "nu_iso",   0.0);
  iprob  = pin->GetOrAddInteger("problem", "iprob",    0);
  gm0    = pin->GetOrAddReal("problem",    "GM",       0.0);
  A0     = pin->GetOrAddReal("problem",    "A0",       1.0);
  sig_x1 = pin->GetOrAddReal("problem",    "sigma_x1", 0.2);
  sig_x2 = pin->GetOrAddReal("problem",    "sigma_x2", 0.2);
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Initializes viscous shear flow.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  //Real rad, phi, z;
  Real v1=0.0, v2=0.0, v3=0.0;
  Real d0 = 1.0; // p0=1.0;
  Real x1, x2, x3; // x2 and x3 are set but unused
  Real rad,z,phi,theta;

  const int num_dust_var = 4*NDUSTFLUIDS;
  //  Initialize density and momenta in Cartesian grids
  if (iprob == 0) { //visc column
    if (nuiso == 0) {
      nuiso = 0.03;
    }
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
            x1=pcoord->x1v(i);
            x2=pcoord->x2v(j);
            x3=pcoord->x3v(k);
            phydro->u(IDN,k,j,i) = A0;
            //phydro->u(IDN,k,j,i) = A0*std::exp(-SQR(x1-x0)/(2.0*SQR(sig_x1)) -SQR(x2-x0)/(2.0*SQR(sig_x2)))+1.0;
            phydro->u(IM1,k,j,i) = phydro->u(IDN,k,j,i)*v1;
            //v2 = v0/std::sqrt(4.0*PI*nuiso*t0)*std::exp(-SQR(x1-x0)/(4.0*nuiso*t0));
            //v1 = v0;
            v2 = v0;
            phydro->u(IM2,k,j,i) = phydro->u(IDN,k,j,i)*v2;
            phydro->u(IM3,k,j,i) = phydro->u(IDN,k,j,i)*v3;
            if (NDUSTFLUIDS > 0) {
              Real concentration = A0*std::exp(-SQR(x1-x0)/(2.0*SQR(sig_x1)) -SQR(x2-x0)/(2.0*SQR(sig_x2)));
              //Real concentration = (x1 <= 0.2 && x1 >= -0.2 && x2 <=0.2 && x2 >=-0.2) ? A0 : 0.0;
              //Real concentration = 0.5;
              for (int n=0; n<num_dust_var; n+=4) {
                int dust_id = n/4;
                int rho_id  = 4*dust_id;
                int v1_id   = rho_id + 1;
                int v2_id   = rho_id + 2;
                int v3_id   = rho_id + 3;
                pdustfluids->df_cons(rho_id,k,j,i) = concentration*phydro->u(IDN,k,j,i);
                //pdustfluids->df_cons(v1_id,k,j,i)  = pdustfluids->df_cons(rho_id,k,j,i)*v0;
                //pdustfluids->df_cons(v2_id,k,j,i)  = pdustfluids->df_cons(rho_id,k,j,i)*v2;
                pdustfluids->df_cons(v1_id,k,j,i)  = 0.0;
                pdustfluids->df_cons(v2_id,k,j,i)  = 0.0;
                pdustfluids->df_cons(v3_id,k,j,i)  = 0.0;
              }
            }
          } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
            rad=pcoord->x1v(i);
            phi=pcoord->x2v(j);
            z=pcoord->x3v(k);
            x1=rad*std::cos(phi);
            x2=rad*std::sin(phi);
            x3=z;
            //v2 = v0/std::sqrt(4.0*PI*nuiso*t0)*std::exp(-SQR(x1-x0)/(4.0*nuiso*t0));
            v2 = 0.0;
            phydro->u(IDN,k,j,i) = A0*std::exp(-SQR(x1-cen1)/(2.0*SQR(sig_x1)) -SQR(x2-cen2)/(2.0*SQR(sig_x2)));
            phydro->u(IM1,k,j,i) = phydro->u(IDN,k,j,i)*(
                v1*std::cos(phi) + v2*std::sin(phi));
            phydro->u(IM2,k,j,i) = phydro->u(IDN,k,j,i)*(
                -v1*std::sin(phi) + v2*std::cos(phi));
            phydro->u(IM3,k,j,i) = phydro->u(IDN,k,j,i)*v3;
            if (NDUSTFLUIDS > 0) {
              //Real concentration = A0*std::exp(-SQR(x1-cen1)/(2.0*SQR(sig_x1)) -SQR(x2-cen2)/(2.0*SQR(sig_x2)));
              //Real concentration = (x1 <= 0.2 && x1 >= -0.2 && x2 <=0.2 && x2 >=-0.2) ? A0 : 0.0;
              Real concentration = 0.5;
              for (int n=0; n<num_dust_var; n+=4) {
                //pdustfluids->df_cons(n,k,j,i)   = concentration*phydro->u(IDN,k,j,i);
                //pdustfluids->df_cons(n+1,k,j,i) = concentration*phydro->u(IM1,k,j,i);
                //pdustfluids->df_cons(n+2,k,j,i) = concentration*phydro->u(IM2,k,j,i);
                //pdustfluids->df_cons(n+3,k,j,i) = concentration*phydro->u(IM3,k,j,i);
                pdustfluids->df_cons(n,k,j,i)   = concentration*phydro->u(IDN,k,j,i);
                pdustfluids->df_cons(n+1,k,j,i) = 0.0;
                pdustfluids->df_cons(n+2,k,j,i) = 0.0;
                pdustfluids->df_cons(n+3,k,j,i) = 0.0;
              }
            }
          } else { // (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
            rad=std::abs(pcoord->x1v(i)*std::sin(pcoord->x2v(j)));
            phi=pcoord->x3v(k);
            theta=pcoord->x2v(j);
            z=pcoord->x1v(i)*std::cos(pcoord->x2v(j));
            x1=rad*std::cos(phi);
            x2=rad*std::sin(phi);
            x3=z;
            v2 = v0/std::sqrt(4.0*PI*nuiso*t0)*std::exp(-SQR(x1-x0)/(4.0*nuiso*t0));
            phydro->u(IDN,k,j,i) = d0;
            phydro->u(IM1,k,j,i) = phydro->u(IDN,k,j,i)*(v1*std::cos(phi)*std::sin(theta)
                                                         +v2*std::sin(phi)*std::sin(theta)
                                                         +v3*std::cos(theta));
            phydro->u(IM2,k,j,i) = phydro->u(IDN,k,j,i)*(v1*std::cos(phi)*std::cos(theta)
                                                         +v2*std::sin(phi)*std::cos(theta)
                                                         +v3*std::sin(theta));
            phydro->u(IM3,k,j,i) = phydro->u(IDN,k,j,i)*(
                -v1*std::sin(phi) + v2*std::cos(phi));
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
          rad=pcoord->x1v(i);
          d0 = std::exp(-SQR(rad-x0)/2.0/SQR(width));
          if (d0 < 1e-6) d0 += 1e-6;
          v1 = -3.0*nuiso*(0.5/rad - (rad-x0)/SQR(width));
          v2 = std::sqrt(gm0/rad); // set it to be Mach=100
          phydro->u(IDN,k,j,i) = d0;
          phydro->u(IM1,k,j,i) = phydro->u(IDN,k,j,i)*v1;
          phydro->u(IM2,k,j,i) = phydro->u(IDN,k,j,i)*v2;
          phydro->u(IM3,k,j,i) = phydro->u(IDN,k,j,i)*v3;
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
