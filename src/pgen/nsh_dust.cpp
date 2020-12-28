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
int ShBoxCoord, ipert,ifield; // initial pattern
Real gm1, iso_cs;
Real x1size, x2size, x3size;
Real Omega_0, qshear;
Real pslope;
Real user_dt;
AthenaArray<Real> volume; // 1D array of volumes
Real HistoryBxBy(MeshBlock *pmb, int iout);
AthenaArray<Real> initial_D2G(NDUSTFLUIDS);
//
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
  //etaVk      *= iso_cs; // switch to code unit
  user_dt    = pin->GetOrAddReal("time", "user_dt", 1e-3);

  if (NDUSTFLUIDS > 0) {
    for (int n=0; n<NDUSTFLUIDS; n++)
      initial_D2G(n) = pin->GetOrAddReal("dust", "Initial_D2G_" + std::to_string(n+1), 0.01);
  }

  EnrollUserExplicitSourceFunction(PressureGradient);
  EnrollUserTimeStepFunction(MyTimeStep);

  // enroll new history variables
  //AllocateUserHistoryOutput(1);
  //EnrollUserHistoryOutput(0, HistoryBxBy, "<-BxBy>");
  return;
}

//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  volume.NewAthenaArray(ncells1);

  //Real d0 = 1.0;
  Real p0 = 1e-5;

  if (NON_BAROTROPIC_EOS) {
    gm1 = (peos->GetGamma() - 1.0);
    iso_cs = std::sqrt((gm1+1.0)*p0/d0);
    //std::cout << "gamma  = " << peos->GetGamma() << std::endl;
  } else {
    iso_cs = peos->GetIsoSoundSpeed();
    p0 = d0*SQR(iso_cs);
    //std::cout << "iso_cs = " << iso_cs << std::endl;
  }

  x1size = pmy_mesh->mesh_size.x1max - pmy_mesh->mesh_size.x1min;
  x2size = pmy_mesh->mesh_size.x2max - pmy_mesh->mesh_size.x2min;
  x3size = pmy_mesh->mesh_size.x3max - pmy_mesh->mesh_size.x3min;
  //std::cout << "[nsh_dust.cpp]: [Lx,Lz,Ly] = [" <<x1size <<","<<x2size<<","<<x3size<<"]"
            //<< std::endl;

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
        rd = d0;
        rp = p0;
        rvx = 0.0;
        rvy = 0.0;
        rvz = 0.0;

        Real &gas_den         = phydro->u(IDN,k,j,i);
        gas_den               = rd;
        phydro->u(IM1,k,j,i)  = gas_den*rvx;
        phydro->u(IM2,k,j,i)  = gas_den*rvy;
        phydro->u(IM3,k,j,i)  = gas_den*rvz;
        phydro->u(IM2,k,j,i) -= gas_den*qshear*Omega_0*x1; // Keplerian Shear
        phydro->u(IM2,k,j,i) -= gas_den*etaVk*iso_cs;      // Pressure Gradient

        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) = rp/gm1 + 0.5*(SQR(phydro->u(IM1,k,j,i)) +
                                               SQR(phydro->u(IM2,k,j,i)) +
                                               SQR(phydro->u(IM3,k,j,i)))/gas_den;
        }

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; n++) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real &dust_den                     = pdustfluids->df_cons(rho_id, k, j, i);
            dust_den                           = initial_D2G(dust_id)*gas_den;
            pdustfluids->df_cons(v1_id,k,j,i)  = dust_den*rvx;
            pdustfluids->df_cons(v2_id,k,j,i)  = dust_den*rvy;
            pdustfluids->df_cons(v3_id,k,j,i)  = dust_den*rvz;
            pdustfluids->df_cons(v2_id,k,j,i) -= dust_den*qshear*Omega_0*x1; // Keplerian Shear
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
  Hydro      *phy = pmb->phydro;
  DustFluids *pdf = pmb->pdustfluids;

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        const Real &gas_rho  = prim(IDN,k,j,i);
        Real press_gra       = 2.0*gas_rho*etaVk*iso_cs*Omega_0*dt;
        Real &m1_gas         = cons(IM1,k,j,i);
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
          //Real press_gra             = 2.0*dust_rho*etaVk*iso_cs*Omega_0*dt;
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



//======================================================================================
//! \fn void MeshBlock::UserWorkInLoop()
//  \brief User-defined work function for every time step
//======================================================================================
//void MeshBlock::UserWorkInLoop() {
  //// nothing to do
  //return;
//}

//namespace {
//Real HistoryBxBy(MeshBlock *pmb, int iout) {
  //Real bxby=0;
  //int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  //AthenaArray<Real> &b = pmb->pfield->bcc;

  //for (int k=ks; k<=ke; k++) {
    //for (int j=js; j<=je; j++) {
      //pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,volume);
      //for (int i=is; i<=ie; i++) {
        //bxby-=volume(i)*b(IB1,k,j,i)*b(IB3,k,j,i);
      //}
    //}
  //}

  //return bxby;
//}
//} // namespace

