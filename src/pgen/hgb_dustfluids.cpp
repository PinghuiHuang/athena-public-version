//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file hgb.cpp
//
//  \brief Problem generator for 3D shearing sheet.
//
// PURPOSE:  Problem generator for 3D shearing sheet.  Based on the initial
//   conditions described in "Local Three-dimensional Magnetohydrodynamic
//   Simulations of Accretion Disks" by Hawley, Gammie & Balbus, or HGB.
//
// Several different field configurations and perturbations are possible:
//
//- ifield = 0 - uses field set by choice of ipert flag
//- ifield = 1 - Bz=B0std::sin(kx*x1) field with zero-net-flux [default] (kx input)
//- ifield = 2 - uniform Bz
//- ifield = 3 - B=(0,B0std::cos(kx*x1),B0std::sin(kx*x1))= zero-net flux w helicity
//- ifield = 4 - B=(0,B0/std::sqrt(2),B0/std::sqrt(2))= net toroidal+vertical field
//- ifield = 5 - uniform By
//
//- ipert = 1 - random perturbations to P and V [default, used by HGB]
//- ipert = 2 - uniform Vx=amp (epicyclic wave test)
//- ipert = 3 - J&G vortical shwave (hydro test)
//- ipert = 4 - nonlinear density wave test of Fromang & Papaloizou
//- ipert = 5 - 2nd MHD shwave test of JGG (2008) -- their figure 9
//- ipert = 6 - 3rd MHD shwave test of JGG (2008) -- their figure 11
//- ipert = 7 - nonlinear shearing wave test of Heinemann & Papaloizou (2008)
//
// To run simulations of stratified disks (including vertical gravity), use the
// strat.c problem generator.
//
// Code must be configured using -shear
//
// REFERENCE: Hawley, J. F. & Balbus, S. A., ApJ 400, 595-609 (1992).
//            Johnson, Guan, & Gammie, ApJSupp, (2008)
//============================================================================

// C headers

// C++ headers
#include <cmath>      // sqrt()
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../utils/utils.hpp" // ran2()
#include "../dustfluids/dustfluids.hpp"

#if !SHEARING_BOX
#error "This problem generator requires shearing box"
#endif

namespace {
Real HistoryBxBy(MeshBlock *pmb, int iout);
Real HistorydVxVy(MeshBlock *pmb, int iout);
Real HistorydBy(MeshBlock *pmb, int iout);
Real Lx, Ly, Lz; // root grid size, global to share with output functions
Real Omega_0, qshear;
} // namespace

// ===================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {
  AllocateUserHistoryOutput(3);
  EnrollUserHistoryOutput(0, HistoryBxBy, "-BxBy");
  EnrollUserHistoryOutput(1, HistorydVxVy, "dVxVy");
  EnrollUserHistoryOutput(2, HistorydBy, "dBy");
  // Read problem parameters
  Omega_0 = pin->GetOrAddReal("problem","Omega0",1.0);
  qshear  = pin->GetOrAddReal("problem","qshear",1.5);
  return;
}

//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief mhd shearing waves and unstratified disk problem generator for
//  3D problems.
//======================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real SumRvx=0.0, SumRvy=0.0, SumRvz=0.0;
  if (pmy_mesh->mesh_size.nx2 == 1) {
    std::cout << "[hgb.cpp]: HGB only works on a 2D or 3D grid" << std::endl;
  }

  // Read problem parameters for initial conditions
  Real amp = pin->GetReal("problem","amp");
  int ipert = pin->GetOrAddInteger("problem","ipert", 1);

  Real beta, dir_sgn;
  int ifield, Bdir;

  // Compute pressure based on the EOS.
  Real den = 1.0, pres =1.0, gamma=1.0, iso_cs=1.0;
  if (NON_BAROTROPIC_EOS) {
    gamma = peos->GetGamma();
    pres = pin->GetReal("problem","pres");
  } else {
    iso_cs =peos->GetIsoSoundSpeed();
    pres = den*SQR(iso_cs);
  }
  // Compute field strength based on beta.
  Real B0  = 0.0;

  // Ensure a different initial random seed for each meshblock.
  std::int64_t iseed = -1 - gid;

  // Initialize boxsize
  Lx = pmy_mesh->mesh_size.x1max - pmy_mesh->mesh_size.x1min;
  Ly = pmy_mesh->mesh_size.x2max - pmy_mesh->mesh_size.x2min;
  Lz = pmy_mesh->mesh_size.x3max - pmy_mesh->mesh_size.x3min;

  // initialize wavenumbers
  int nwx = pin->GetOrAddInteger("problem","nwx",1);
  int nwy = pin->GetOrAddInteger("problem","nwy",1);
  int nwz = pin->GetOrAddInteger("problem","nwz",1);
  Real kx = (TWO_PI/Lx)*(static_cast<Real>(nwx));// nxw=-ve for leading wave
  Real ky = (TWO_PI/Ly)*(static_cast<Real>(nwy));
  Real kz = (TWO_PI/Lz)*(static_cast<Real>(nwz));

  // For PF density wave test, read data from file: not implemented yet.


  // Rescale amp to sound speed for ipert 2,3
  if (NON_BAROTROPIC_EOS) {
    if (ipert == 2 || ipert == 3)
      amp *= std::sqrt(gamma*pres/den);
  } else {
    if (ipert == 2 || ipert == 3)
      amp *= iso_cs;
  }

  Real x1, x2, x3;  // xmin, xmax;
  Real x1f, x2f, x3f;
  Real rd(0.0), rp(0.0), rvx(0.0), rvy(0.0), rvz(0.0), rbx(0.0), rby(0.0), rbz(0.0);
  Real rval;

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        x1 = pcoord->x1v(i);
        x2 = pcoord->x2v(j);
        x3 = pcoord->x3v(k);
        x1f = pcoord->x1f(i);
        x2f = pcoord->x2f(j);
        x3f = pcoord->x3f(k);

        //Initialize perturbations
        // ipert = 1 - random perturbations to P and V [default, used by HGB]
        // ipert = 2 - uniform Vx=amp (epicyclic wave test)
        // ipert = 3 - vortical shwave (hydro test)
        // ipert = 4 - Fromang & Papaloizou nonlinear density wave (hydro test)
        // ipert = 5 & 6 - JGG MHD shwave tests
        // ipert = 7 - Heinemann & Papaloizou (2008) nonlinear shwave (hydro test)
        if (ipert == 1) {
          rval = amp*(ran2(&iseed) - 0.5);
          if (NON_BAROTROPIC_EOS) {
            rp = pres*(1.0 + 2.0*rval);
            rd = den;
          } else {
            rd = den; //den*(1.0 + 2.0*rval);
          }
          // Follow HGB: the perturbations to V/Cs are
          // (1/5)amp/std::sqrt(gamma)
          rval = amp*(ran2(&iseed) - 0.5);
          rvx = (0.4/std::sqrt(3.0)) *rval*1e-3/std::sqrt(gamma);
          //rvx = 0.4*rval*std::sqrt(pres/den);
          SumRvx += rvx;

          rval = amp*(ran2(&iseed) - 0.5);
          rvy = (0.4/std::sqrt(3.0)) *rval*1e-3/std::sqrt(gamma);
          //rvy = 0.4*rval*std::sqrt(pres/den);
          SumRvy += rvy;

          rval = amp*(ran2(&iseed) - 0.5);
          rvz = (0.4/std::sqrt(3.0)) *rval*1e-3/std::sqrt(gamma);
          //rvz = 0.4*rval*std::sqrt(pres/den);
          SumRvz += rvz;
        }
        if (ipert == 2) {
          rp = pres;
          rd = den;
          rvx = amp;
          rvy = 0.0;
          rvz = 0.0;
        }
        if (ipert == 3) {
          rp = pres;
          rd = den;
          rvx = amp*std::sin(static_cast<Real>(kx*x1 + ky*x2));
          rvy = -amp*(kx/ky)*std::sin(static_cast<Real>(kx*x1 + ky*x2));
          rvz = 0.0;
        }
        if (ipert == 4) {
          std::stringstream msg;
          msg << "### FATAL ERROR in hgb.cpp ProblemGenerator" << std::endl
              << "ipert=4 (nonlinear density wave test of Fromang & Papaloizou)"
              << " not implemented yet!" << std::endl;
          ATHENA_ERROR(msg);
        }
        // Note: ICs in JGG for this test are incorrect.
        if (ipert == 5) {
          ifield = 0;
          rd =
              den + 8.9525e-10*std::cos(static_cast<Real>(kx*x1 + ky*x2 + kz*x3 - PI/4.));
          rvx = 8.16589e-8*std::cos(static_cast<Real>(kx*x1 + ky*x2 + kz*x3 + PI/4.));
          rvy = 8.70641e-8*std::cos(static_cast<Real>(kx*x1 + ky*x2 + kz*x3 + PI/4.));
          rvz = 0.762537e-8*std::cos(static_cast<Real>(kx*x1 + ky*x2 + kz*x3 + PI/4.));
          rbx = -1.08076e-7;
          rbx *= std::cos(static_cast<Real>(kx*(x1-0.5*pcoord->dx1f(i)) +
                                            ky*x2 + kz*x3 - PI/4.));
          rby = 1.04172e-7;
          rby *= std::cos(static_cast<Real>(kx*x1 + ky*(x2-0.5*pcoord->dx2f(j)) +
                                            kz*x3 - PI/4.));
          rbz = -0.320324e-7;
          rbz *= std::cos(static_cast<Real>(kx*x1 + ky*x2 +
                                            kz*(x3-0.5*pcoord->dx3f(k)) - PI/4.));
          rbz += (std::sqrt(15.0)/16.0)*(Omega_0/kz);
        }
        if (ipert == 6) {
          ifield = 0;
          rd = den + 5.48082e-6*std::cos(static_cast<Real>(kx*x1 + ky*x2 + kz*x3));
          rvx = -4.5856e-6*std::cos(static_cast<Real>(kx*x1 + ky*x2 + kz*x3));
          rvy = 2.29279e-6*std::cos(static_cast<Real>(kx*x1 + ky*x2 + kz*x3));
          rvz = 2.29279e-6*std::cos(static_cast<Real>(kx*x1 + ky*x2 + kz*x3));
          rbx = 5.48082e-7;
          rbx *= std::cos(static_cast<Real>(kx*x1f + ky*x2 + kz*x3));
          rbx += (0.1);
          rby = 1.0962e-6;
          rby *= std::cos(static_cast<Real>(kx*x1 + ky*x2f + kz*x3));
          rby += (0.2);
          rbz = 0.0;
        }
        if (ipert == 7) {
          if (!NON_BAROTROPIC_EOS) {
            Real kappa2 = 2.0*(2.0 - qshear)*Omega_0*Omega_0;
            Real aa = (kx*kx + ky*ky)*SQR(iso_cs) + kappa2;
            Real bb = 2.0*qshear*Omega_0*ky*iso_cs;
            Real denom = aa*aa + bb*bb;
            Real rd_hat =         (ky*iso_cs*bb -2.0*Omega_0*aa)*amp/denom;
            Real px_hat = -iso_cs*(ky*iso_cs*aa +2.0*Omega_0*bb)*amp/denom;
            Real py_hat = (amp + ky*px_hat + (2.0-qshear)*Omega_0*rd_hat)/kx;
            rd  = 1.0 + rd_hat*std::cos(static_cast<Real>(kx*x1 + ky*x2));
            rvx = px_hat*std::sin(static_cast<Real>(kx*x1 + ky*x2))/rd;
            rvy = py_hat*std::sin(static_cast<Real>(kx*x1 + ky*x2))/rd;
          }
          rvz = 0.0;
        }

        // Initialize (d, M, P)
        // for_the_future: if FARGO do not initialize the bg shear
        phydro->u(IDN,k,j,i) = rd;
        phydro->u(IM1,k,j,i) = rd*rvx;
        phydro->u(IM2,k,j,i) = rd*rvy;
        phydro->u(IM2,k,j,i) -= rd*(qshear*Omega_0*x1);
        phydro->u(IM3,k,j,i) = rd*rvz;
        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) = rp/(gamma-1.0)
                                 + 0.5*(SQR(phydro->u(IM1,k,j,i))
                                        + SQR(phydro->u(IM2,k,j,i))
                                        + SQR(phydro->u(IM3,k,j,i)))/rd;
        } // Hydro

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; n++) {
            int rho_id  = 4*n;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            pdustfluids->df_cons(rho_id, ks, j, i)  = d0;
            pdustfluids->df_cons(v1_id,  ks, j, i)  = d0*rvx;
            pdustfluids->df_cons(v2_id,  ks, j, i)  = d0*rvy;
            pdustfluids->df_cons(v3_id,  ks, j, i)  = d0*rvz;
            pdustfluids->df_cons(v3_id,  ks, j, i) -= d0*qshear*Omega_0*x1;
          }
        }

        // Initialize b.  For 3D shearing box B1=Bx, B2=By, B3=Bz
        // ifield = 0 - used with ipert=5 or 6
        // ifield = 1 - Bz=B0std::sin(x1) field with zero-net-flux[default]
        // ifield = 2 - uniform Bz
        // ifield = 3 - B=(0,B0std::cos(kx*x1),B0std::sin(kx*x1))=zero-net flux w helicity
        // ifield = 4 - B=(0,B0/std::sqrt(2),B0/std::sqrt(2))= net toroidal+vertical field
      }
    }
  }


  // For random perturbations as in HGB, ensure net momentum is zero by
  // subtracting off mean of perturbations
  if (ipert == 1) {
    int cell_num = block_size.nx1*block_size.nx2*block_size.nx3;
    SumRvx /= cell_num;
    SumRvy /= cell_num;
    SumRvz /= cell_num;
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          phydro->u(IM1,k,j,i) -= rd*SumRvx;
          phydro->u(IM2,k,j,i) -= rd*SumRvy;
          phydro->u(IM3,k,j,i) -= rd*SumRvz;
        }
      }
    }
  }


  return;
}

namespace {

Real HistoryBxBy(MeshBlock *pmb, int iout) {
  Real bxby = 0;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> &b = pmb->pfield->bcc;
  AthenaArray<Real> volume; // 1D array of volumes
  // allocate 1D array for cell volume used in usr def history
  volume.NewAthenaArray(pmb->ncells1);

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k, j, is, ie, volume);
      for (int i=is; i<=ie; i++) {
        bxby-=volume(i)*b(IB1,k,j,i)*b(IB2,k,j,i);
      }
    }
  }
  return bxby;
}

Real HistorydVxVy(MeshBlock *pmb, int iout) {
  Real dvxvy = 0.0;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> &w = pmb->phydro->w;
  Real vshear = 0.0;
  AthenaArray<Real> volume; // 1D array of volumes
  // allocate 1D array for cell volume used in usr def history
  volume.NewAthenaArray(pmb->ncells1);

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k, j, is, ie, volume);
      for (int i=is; i<=ie; i++) {
        vshear = qshear*Omega_0*pmb->pcoord->x1v(i);
        dvxvy += volume(i)*w(IDN,k,j,i)*w(IVX,k,j,i)*(w(IVY,k,j,i)+vshear);
      }
    }
  }
  return dvxvy;
}

Real HistorydBy(MeshBlock *pmb, int iout) {
  Real dby = 0;
  Real fkx, fky, fkz; // Fourier kx, ky
  Real x1, x2, x3;
  AthenaArray<Real> volume; // 1D array of volumes
  volume.NewAthenaArray(pmb->ncells1);
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> &b = pmb->pfield->bcc;

  fky = TWO_PI/Ly;
  fkx = -4.0*PI/Lx + qshear*Omega_0*fky*pmb->pmy_mesh->time;
  fkz = TWO_PI/Lz;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k, j, is, ie, volume);
      for (int i=is; i<=ie; i++) {
        x1 = pmb->pcoord->x1v(i);
        x2 = pmb->pcoord->x2v(j);
        x3 = pmb->pcoord->x3v(k);
        dby += (2.0
                * volume(i)
                * (b(IB2, k, j, i) - (0.2-0.15*Omega_0*pmb->pmy_mesh->time))
                * std::cos(fkx*x1 + fky*x2 + fkz*x3));
      }
    }
  }
  return dby;
}
} // namespace
