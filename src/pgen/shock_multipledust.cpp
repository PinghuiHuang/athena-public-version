//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file shock_tube.cpp
//  \brief Problem generator for shock tube problems.
//
// Problem generator for shock tube (1-D Riemann) problems. Initializes plane-parallel
// shock along x1 (in 1D, 2D, 3D), along x2 (in 2D, 3D), and along x3 (in 3D).
//========================================================================================

// C headers

// C++ headers
#include <cmath>      // sqrt()
#include <cstdio>     // fopen(), freopen(), fprintf(), fclose()
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../dustfluids/dustfluids.hpp"

// problem parameters which are useful to make global to this file
namespace {
Real user_dt, iso_cs, xshock;
Real MyTimeStep(MeshBlock *pmb);
Real wl[NHYDRO];
Real wr[NHYDRO];
Real wl_d[4];
Real wr_d[4];
} // namespace

//========================================================================================
//! \fn Real press(Real rho, Real T)
//  \brief Calculate pressure as a function of density and temperature for H EOS.
//========================================================================================

// User-defined boundary conditions for disk simulations
void InnerX1_Fixed(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);

void OuterX1_Fixed(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);

Real press(Real rho, Real T) {
  // Ionization fraction
  Real x = 2. /(1 + std::sqrt(1 + 4. * rho * std::exp(1. / T) * std::pow(T, -1.5)));
  return rho * T * (1. + x);
}

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Get parameters for gravitatonal potential of central point mass
  user_dt = pin->GetOrAddReal("problem", "user_dt", 1e-1);
  iso_cs  = pin->GetOrAddReal("hydro", "iso_sound_speed", 1e-1);
  xshock  = pin->GetReal("problem","xshock");
  wl[IDN] = pin->GetReal("problem", "dl");
  wl[IVX] = pin->GetReal("problem", "ul");
  wl[IVY] = pin->GetReal("problem", "vl");
  wl[IVZ] = pin->GetReal("problem", "wl");

  wl_d[0] = pin->GetReal("dust", "dl_d");
  wl_d[1] = pin->GetReal("dust", "ul_d");
  wl_d[2] = pin->GetReal("dust", "vl_d");
  wl_d[3] = pin->GetReal("dust", "wl_d");

  wr[IDN] = pin->GetReal("problem","dr");
  wr[IVX] = pin->GetReal("problem","ur");
  wr[IVY] = pin->GetReal("problem","vr");
  wr[IVZ] = pin->GetReal("problem","wr");

  wr_d[0] = pin->GetReal("dust", "dr_d");
  wr_d[1] = pin->GetReal("dust", "ur_d");
  wr_d[2] = pin->GetReal("dust", "vr_d");
  wr_d[3] = pin->GetReal("dust", "wr_d");
  EnrollUserTimeStepFunction(MyTimeStep);
  //
  // enroll user-defined boundary condition
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, InnerX1_Fixed);
  }
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, OuterX1_Fixed);
  }

  return;
}

//namespace {
//Real MyTimeStep(MeshBlock *pmb)
//{
  //Real min_user_dt = user_dt;
  //return min_user_dt;
//}
//}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Problem Generator for the shock tube tests
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  std::stringstream msg;

  // parse shock direction: {1, 2, 3} -> {x1, x2, x3}
  int shk_dir = pin->GetInteger("problem","shock_dir");

  // parse shock location (must be inside grid)
  Real xshock = pin->GetReal("problem","xshock");
  if (shk_dir == 1 && (xshock < pmy_mesh->mesh_size.x1min ||
                       xshock > pmy_mesh->mesh_size.x1max)) {
    msg << "### FATAL ERROR in Problem Generator" << std::endl << "xshock="
        << xshock << " lies outside x1 domain for shkdir=" << shk_dir << std::endl;
    ATHENA_ERROR(msg);
  }
  if (shk_dir == 2 && (xshock < pmy_mesh->mesh_size.x2min ||
                       xshock > pmy_mesh->mesh_size.x2max)) {
    msg << "### FATAL ERROR in Problem Generator" << std::endl << "xshock="
        << xshock << " lies outside x2 domain for shkdir=" << shk_dir << std::endl;
    ATHENA_ERROR(msg);
  }
  if (shk_dir == 3 && (xshock < pmy_mesh->mesh_size.x3min ||
                       xshock > pmy_mesh->mesh_size.x3max)) {
    msg << "### FATAL ERROR in Problem Generator" << std::endl << "xshock="
        << xshock << " lies outside x3 domain for shkdir=" << shk_dir << std::endl;
    ATHENA_ERROR(msg);
  }

  // Parse left state read from input file: dl, ul, vl, wl,[pl]
  Real wl[NHYDRO];
  Real wl_d[4];
  wl[IDN] = pin->GetReal("problem", "dl");
  wl[IVX] = pin->GetReal("problem", "ul");
  wl[IVY] = pin->GetReal("problem", "vl");
  wl[IVZ] = pin->GetReal("problem", "wl");

  wl_d[0] = pin->GetReal("dust", "dl_d");
  wl_d[1] = pin->GetReal("dust", "ul_d");
  wl_d[2] = pin->GetReal("dust", "vl_d");
  wl_d[3] = pin->GetReal("dust", "wl_d");
  if (NON_BAROTROPIC_EOS) {
    if (pin->DoesParameterExist("problem","Tl"))
      wl[IPR] = press(wl[IDN], pin->GetReal("problem","Tl"));
    else
      wl[IPR] = pin->GetReal("problem","pl");
  }
  // Parse right state read from input file: dr, ur, vr, wr,[pr]
  Real wr[NHYDRO];
  Real wr_d[4];
  wr[IDN] = pin->GetReal("problem","dr");
  wr[IVX] = pin->GetReal("problem","ur");
  wr[IVY] = pin->GetReal("problem","vr");
  wr[IVZ] = pin->GetReal("problem","wr");

  wr_d[0] = pin->GetReal("dust", "dr_d");
  wr_d[1] = pin->GetReal("dust", "ur_d");
  wr_d[2] = pin->GetReal("dust", "vr_d");
  wr_d[3] = pin->GetReal("dust", "wr_d");
  if (NON_BAROTROPIC_EOS) {
    if (pin->DoesParameterExist("problem","Tr"))
      wr[IPR] = press(wr[IDN], pin->GetReal("problem","Tr"));
    else
      wr[IPR] = pin->GetReal("problem","pr");
  }
  // Initialize the discontinuity in the Hydro variables ---------------------------------

  switch(shk_dir) {
    //--- shock in 1-direction
    case 1:
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie; ++i) {
            if (pcoord->x1v(i) < xshock) {
              phydro->u(IDN, k, j, i) = wl[IDN];
              phydro->u(IM1, k, j, i) = wl[IVX]*wl[IDN];
              phydro->u(IM2, k, j, i) = wl[IVY]*wl[IDN];
              phydro->u(IM3, k, j, i) = wl[IVZ]*wl[IDN];

              if (NDUSTFLUIDS > 0) {
                for (int n = 0; n<NDUSTFLUIDS; ++n){
                  int dust_id = n;
                  int rho_id  = 4 * dust_id;
                  int v1_id   = rho_id + 1;
                  int v2_id   = rho_id + 2;
                  int v3_id   = rho_id + 3;
                  pdustfluids->df_cons(rho_id, k, j, i) = wl_d[0];
                  pdustfluids->df_cons(v1_id, k, j, i)  = wl_d[1]*wl_d[0];
                  pdustfluids->df_cons(v2_id, k, j, i)  = wl_d[2]*wl_d[0];
                  pdustfluids->df_cons(v3_id, k, j, i)  = wl_d[3]*wl_d[0];
                }
              }
              if (NON_BAROTROPIC_EOS) {
                if (GENERAL_EOS) {
                  phydro->u(IEN, k, j, i) = peos->EgasFromRhoP(wl[IDN], wl[IPR]);
                } else {
                  phydro->u(IEN, k, j, i) = wl[IPR]/(peos->GetGamma() - 1.0);
                }
                phydro->u(IEN, k, j, i) += 0.5 * wl[IDN]*(wl[IVX]*wl[IVX] + wl[IVY]*wl[IVY] + wl[IVZ]*wl[IVZ]);
              }
            } else {
              phydro->u(IDN, k, j, i) = wr[IDN];
              phydro->u(IM1, k, j, i) = wr[IVX]*wr[IDN];
              phydro->u(IM2, k, j, i) = wr[IVY]*wr[IDN];
              phydro->u(IM3, k, j, i) = wr[IVZ]*wr[IDN];

              if (NDUSTFLUIDS > 0) {
                for (int n = 0; n<NDUSTFLUIDS; ++n){
                  int dust_id = n;
                  int rho_id  = 4 * dust_id;
                  int v1_id   = rho_id + 1;
                  int v2_id   = rho_id + 2;
                  int v3_id   = rho_id + 3;
                  pdustfluids->df_cons(rho_id, k, j, i) = wr_d[0];
                  pdustfluids->df_cons(v1_id,  k, j, i) = wr_d[1]*wr_d[0];
                  pdustfluids->df_cons(v2_id,  k, j, i) = wr_d[2]*wr_d[0];
                  pdustfluids->df_cons(v3_id,  k, j, i) = wr_d[3]*wr_d[0];
                }
              }

              if (NON_BAROTROPIC_EOS) {
                if (GENERAL_EOS) {
                  phydro->u(IEN, k, j, i) = peos->EgasFromRhoP(wr[IDN], wr[IPR]);
                } else {
                  phydro->u(IEN, k, j, i) = wr[IPR]/(peos->GetGamma() - 1.0);
                }
                phydro->u(IEN, k, j, i) += 0.5 * wr[IDN]*(wr[IVX]*wr[IVX] + wr[IVY]*wr[IVY] + wr[IVZ]*wr[IVZ]);
              }
            }
          }
        }
      }
      break;
      //--- shock in 2-direction
    case 2:
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
          if (pcoord->x2v(j) < xshock) {
            for (int i=is; i<=ie; ++i) {
              phydro->u(IDN, k, j, i) = wl[IDN];
              phydro->u(IM2, k, j, i) = wl[IVX]*wl[IDN];
              phydro->u(IM3, k, j, i) = wl[IVY]*wl[IDN];
              phydro->u(IM1, k, j, i) = wl[IVZ]*wl[IDN];
              if (NON_BAROTROPIC_EOS) {
                if (GENERAL_EOS) {
                  phydro->u(IEN, k, j, i) = peos->EgasFromRhoP(wl[IDN], wl[IPR]);
                } else {
                  phydro->u(IEN, k, j, i) = wl[IPR]/(peos->GetGamma() - 1.0);
                }
                phydro->u(IEN, k, j, i) += 0.5 * wl[IDN]*(wl[IVX]*wl[IVX] + wl[IVY]*wl[IVY] + wl[IVZ]*wl[IVZ]);
              }
            }
          } else {
            for (int i=is; i<=ie; ++i) {
              phydro->u(IDN, k, j, i) = wr[IDN];
              phydro->u(IM2, k, j, i) = wr[IVX]*wr[IDN];
              phydro->u(IM3, k, j, i) = wr[IVY]*wr[IDN];
              phydro->u(IM1, k, j, i) = wr[IVZ]*wr[IDN];
              if (NON_BAROTROPIC_EOS) {
                if (GENERAL_EOS) {
                  phydro->u(IEN, k, j, i) = peos->EgasFromRhoP(wr[IDN], wr[IPR]);
                } else {
                  phydro->u(IEN, k, j, i) = wr[IPR]/(peos->GetGamma() - 1.0);
                }
                phydro->u(IEN, k, j, i) += 0.5 * wr[IDN]*(wr[IVX]*wr[IVX] + wr[IVY]*wr[IVY] + wr[IVZ]*wr[IVZ]);
              }
            }
          }
        }
      }
      break;

      //--- shock in 3-direction
    case 3:
      for (int k=ks; k<=ke; ++k) {
        if (pcoord->x3v(k) < xshock) {
          for (int j=js; j<=je; ++j) {
            for (int i=is; i<=ie; ++i) {
              phydro->u(IDN, k, j, i) = wl[IDN];
              phydro->u(IM3, k, j, i) = wl[IVX]*wl[IDN];
              phydro->u(IM1, k, j, i) = wl[IVY]*wl[IDN];
              phydro->u(IM2, k, j, i) = wl[IVZ]*wl[IDN];
              if (NON_BAROTROPIC_EOS) {
                if (GENERAL_EOS) {
                  phydro->u(IEN, k, j, i) = peos->EgasFromRhoP(wl[IDN], wl[IPR]);
                } else {
                  phydro->u(IEN, k, j, i) = wl[IPR]/(peos->GetGamma() - 1.0);
                }
                phydro->u(IEN, k, j, i) += 0.5 * wl[IDN]*(wl[IVX]*wl[IVX] + wl[IVY]*wl[IVY] + wl[IVZ]*wl[IVZ]);
              }
            }
          }
        } else {
          for (int j=js; j<=je; ++j) {
            for (int i=is; i<=ie; ++i) {
              phydro->u(IDN, k, j, i) = wr[IDN];
              phydro->u(IM3, k, j, i) = wr[IVX]*wr[IDN];
              phydro->u(IM1, k, j, i) = wr[IVY]*wr[IDN];
              phydro->u(IM2, k, j, i) = wr[IVZ]*wr[IDN];
              if (NON_BAROTROPIC_EOS) {
                if (GENERAL_EOS) {
                  phydro->u(IEN, k, j, i) = peos->EgasFromRhoP(wr[IDN], wr[IPR]);
                } else {
                  phydro->u(IEN, k, j, i) = wr[IPR]/(peos->GetGamma() - 1.0);
                }
                phydro->u(IEN, k, j, i) += 0.5 * wr[IDN]*(wr[IVX]*wr[IVX] + wr[IVY]*wr[IVY] + wr[IVZ]*wr[IVZ]);
              }
            }
          }
        }
      }
      break;

    default:
      msg << "### FATAL ERROR in Problem Generator" << std::endl
          << "shock_dir=" << shk_dir << " must be either 1, 2, or 3" << std::endl;
      ATHENA_ERROR(msg);
  }

  // now set face-centered (interface) magnetic fields -----------------------------------


  // uniformly fill all scalars to have equal concentration
  // mass fraction? or concentration?
  return;
}

void InnerX1_Fixed(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        prim(IDN, k, j, il-i) = wl[IDN];
        prim(IVX, k, j, il-i) = wl[IVX];
        prim(IVY, k, j, il-i) = wl[IVY];
        prim(IVZ, k, j, il-i) = wl[IVZ];
        if (NON_BAROTROPIC_EOS)
          prim(IPR, k, j, il-i) = SQR(iso_cs)*prim(IDN, k, j, il-i);

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; n++) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;
            pmb->pdustfluids->df_prim(rho_id, k, j, il-i) = wl_d[rho_id];
            pmb->pdustfluids->df_prim(v1_id,  k, j, il-i) = wl_d[v1_id];
            pmb->pdustfluids->df_prim(v2_id,  k, j, il-i) = wl_d[v2_id];
            pmb->pdustfluids->df_prim(v3_id,  k, j, il-i) = wl_d[v3_id];
          }
        }

      }
    }
  }
  return;
}

void OuterX1_Fixed(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        prim(IDN, k, j, iu+i) = wl[IDN];
        prim(IVX, k, j, iu+i) = wl[IVX];
        prim(IVY, k, j, iu+i) = wl[IVY];
        prim(IVZ, k, j, iu+i) = wl[IVZ];
        if (NON_BAROTROPIC_EOS)
          prim(IPR, k, j, iu+i) = SQR(iso_cs)*prim(IDN, k, j, iu+i);

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; n++) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;
            pmb->pdustfluids->df_prim(rho_id, k, j, iu+i) = wl_d[rho_id];
            pmb->pdustfluids->df_prim(v1_id,  k, j, iu+i) = wl_d[v1_id];
            pmb->pdustfluids->df_prim(v2_id,  k, j, iu+i) = wl_d[v2_id];
            pmb->pdustfluids->df_prim(v3_id,  k, j, iu+i) = wl_d[v3_id];
          }
        }

      }
    }
  }
  return;
}

void MeshBlock::UserWorkInLoop() {
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        if (NON_BAROTROPIC_EOS) {
          //phydro->w(IPR, k, j, i) = SQR(iso_cs)*phydro->w(IDN,k,j,i);
          if (pcoord->x1v(i) <= xshock)
            phydro->w(IPR, k, j, i) = SQR(iso_cs)*wl[IDN];
          else
            phydro->w(IPR, k, j, i) = SQR(iso_cs)*wr[IDN];
        }
      }
    }
  }
  return;
}
