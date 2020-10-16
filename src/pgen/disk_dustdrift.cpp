//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file disk.cpp
//  \brief Initializes stratified Keplerian accretion disk in both cylindrical and
//  spherical polar coordinates.  Initial conditions are in vertical hydrostatic eqm.

// C headers

// C++ headers
#include <algorithm>  // min
#include <cmath>      // sqrt
#include <cstdlib>    // srand
#include <cstring>    // strcmp()
#include <fstream>
#include <iostream>   // endl
#include <limits>
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

namespace {
void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k);
Real DenProfileCyl(const Real rad, const Real phi, const Real z);
Real DenProfileCyl_Gaussian(const Real rad, const Real phi, const Real z);
Real RadialD2G(const Real rad, const Real initial_dust2gas, const Real slope);
Real PoverR(const Real rad, const Real phi, const Real z);
void VelProfileCyl(const Real rad, const Real phi, const Real z,
                   Real &v1, Real &v2, Real &v3);
void VelProfileCyl_DustFluids(const Real rad, const Real phi, const Real z,
                   Real &v1, Real &v2, Real &v3);
void Linear_interpolate(const Real x0, const Real x1, const Real y0, const Real y1,
    Real &x, Real &y);
void Keplerian_interpolate(const Real r_active, const Real r_ghost, const Real vphi_active,
    Real &vphi_ghost);
void Density_interpolate(const Real r_active, const Real r_ghost, const Real rho_active,
    const Real slope, Real &rho_ghost);
void Vr_interpolate_inner_powerlaw(const Real r_active, const Real r_ghost, const Real vr_active,
    const Real slope, Real &vr_ghost);
void Vr_interpolate_outer_powerlaw(const Real r_active, const Real r_ghost, const Real vr_active,
    const Real slope, Real &vr_ghost);
// problem parameters which are useful to make global to this file

// User Sources
void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
void cooling(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
void potentialwell(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
void corotate(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
void wavedamping(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);

Real gm0, r0, rho0, dslope, p0_over_r0, pslope, gamma_gas, initial_D2G, dfloor;
Real tau_relax, rs, gmp, rad_planet, phi_planet, t0pot, omega_p, Bump_flag, A0, dwidth, rn, rand_amp, dust_dens_slope;
Real x1min, x1max, tau_damping, damping_rate;
Real radius_inner_damping, radius_outer_damping, inner_ratio_region, outer_ratio_region, inner_width_damping, outer_width_damping;
} // namespace

// User-defined boundary conditions for disk simulations
void InnerX1_NoMatterInput(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void OuterX1_NoMatterInput(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);

//========================================================================================
//! \fn void InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Get parameters for gravitatonal potential of central point mass
  gm0 = pin->GetOrAddReal("problem","GM",0.0);
  r0  = pin->GetOrAddReal("problem","r0",1.0);

  // Get parameters for initial density and velocity
  rho0            = pin->GetReal("problem",      "rho0");
  dslope          = pin->GetOrAddReal("problem", "dslope", 0.0);
  dust_dens_slope = pin->GetOrAddReal("problem", "dust_dens_slope", 0.0);

  // The parameters of the amplitude of random perturbation on the radial velocity
  rand_amp = pin->GetOrAddReal("problem", "random_vel_r_amp", 0.0);

  // The parameters of gaussian bumps
  Bump_flag = pin->GetOrAddInteger("problem", "gaussianbump_flag", 0);
  A0        = pin->GetOrAddReal("problem",    "A0",                0.0);
  rn        = pin->GetOrAddReal("problem",    "rn",                1.0);
  dwidth    = pin->GetOrAddReal("problem",    "dwidth",            0.1);

  // The parameters of damping zones
  x1min = pin->GetReal("mesh", "x1min");
  x1max = pin->GetReal("mesh", "x1max");

  //ratio of the orbital periods between the edge of the wave-killing zone and the corresponding edge of the mesh
  inner_ratio_region   = pin->GetOrAddReal("problem","inner_dampingregion_ratio",2.5);
  outer_ratio_region   = pin->GetOrAddReal("problem","outer_dampingregion_ratio",1.2);

  radius_inner_damping = x1min*pow(inner_ratio_region, 2./3.);
  radius_outer_damping = x1max*pow(outer_ratio_region, -2./3.);

  inner_width_damping = radius_inner_damping - x1min;
  outer_width_damping = x1max - radius_inner_damping;

  damping_rate = pin->GetOrAddReal("problem", "damping_rate", 1.0); // The normalized wave damping timescale, in unit of dynamical timescale.

  // The parameters of one planet
  tau_relax    = pin->GetOrAddReal("hydro",   "tau_relax",    0.01);
  rad_planet   = pin->GetOrAddReal("problem", "rad_planet",   1.0);  // radial position of the planet
  phi_planet   = pin->GetOrAddReal("problem", "phi_planet",   0.0);   // azimuthal position of the planet
  t0pot        = pin->GetOrAddReal("problem", "t0pot",        0.0);  // time to put in the planet
  gmp          = pin->GetOrAddReal("problem", "GMp",          0.0);  // GM of the planet
  rs           = pin->GetOrAddReal("problem", "rs",           0.1);  // softening length of the gravitational potential of planets
  omega_p      = sqrt(gm0/pow(rad_planet,     3));                     // The Omega of planetary orbit

  if (NDUSTFLUIDS > 0)
      initial_D2G = pin->GetOrAddReal("problem", "intial_D2G", 0.01);

  // Get parameters of initial pressure and cooling parameters
  if (NON_BAROTROPIC_EOS) {
    p0_over_r0 = pin->GetOrAddReal("problem", "p0_over_r0", 0.01);
    pslope     = pin->GetOrAddReal("problem", "pslope",     -0.5);
    gamma_gas  = pin->GetReal("hydro",        "gamma");
  } else {
    p0_over_r0 = SQR(pin->GetReal("hydro","iso_sound_speed"));
  }
  Real float_min = std::numeric_limits<float>::min();
  dfloor         = pin->GetOrAddReal("hydro","dfloor",(1024*(float_min)));

  //if (gmp > 0.0)
    EnrollUserExplicitSourceFunction(MySource);

  // enroll user-defined boundary condition
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, InnerX1_NoMatterInput);
  }
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, OuterX1_NoMatterInput);
  }

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Initializes Keplerian accretion disk.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real v1(0.0), v2(0.0), v3(0.0);
  Real df_v1(0.0), df_v2(0.0), df_v3(0.0);
  Real delta(0.0);

  //  Initialize density and momenta
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        GetCylCoord(pcoord,rad,phi,z,i,j,k); // convert to cylindrical coordinates
        // compute initial conditions in cylindrical coordinates
        phydro->u(IDN,k,j,i) = DenProfileCyl(rad,phi,z);
        //phydro->u(IDN,k,j,i) = Bump_flag ? DenProfileCyl_Gaussian(rad,phi,z) : DenProfileCyl(rad,phi,z);
        VelProfileCyl(rad,phi,z,v1,v2,v3);

        phydro->u(IM1,k,j,i) = phydro->u(IDN,k,j,i)*v1;
        phydro->u(IM2,k,j,i) = phydro->u(IDN,k,j,i)*v2;
        phydro->u(IM3,k,j,i) = phydro->u(IDN,k,j,i)*v3;

        if (NON_BAROTROPIC_EOS) {
          Real p_over_r = PoverR(rad,phi,z);
          phydro->u(IEN,k,j,i) = p_over_r*phydro->u(IDN,k,j,i)/(gamma_gas - 1.0);
          phydro->u(IEN,k,j,i) += 0.5*(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))
                                       + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
        }

        if (NDUSTFLUIDS > 0){
          for (int n = 0; n<NDUSTFLUIDS; n++){
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;
            //delta = ((rad > 0.8) && (rad < 1.2)) ? 1.0 : 0.0;
            //pdustfluids->df_cons(rho_id,k,j,i) = 0.1*rho0*std::exp(-SQR(rad-1.0)/(2*SQR(0.1)));
            VelProfileCyl_DustFluids(rad, phi, z, df_v1, df_v2, df_v3);
            pdustfluids->df_cons(rho_id,k,j,i) = initial_D2G* phydro->u(IDN,k,j,i);
            pdustfluids->df_cons(v1_id,k,j,i)  = pdustfluids->df_cons(rho_id,k,j,i) * df_v1;
            pdustfluids->df_cons(v2_id,k,j,i)  = pdustfluids->df_cons(rho_id,k,j,i) * df_v2;
            pdustfluids->df_cons(v3_id,k,j,i)  = pdustfluids->df_cons(rho_id,k,j,i) * df_v3;
          }
        }
      }
    }
  }

  return;
}

namespace {
//----------------------------------------------------------------------------------------
//!\f transform to cylindrical coordinate

void MySource(MeshBlock *pmb, const Real time, const Real dt,
    const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons)
{
  //cooling(pmb,       time, dt, prim, bcc, cons);
  //potentialwell(pmb, time, dt, prim, bcc, cons);
  wavedamping(pmb,   time, dt, prim, bcc, cons);
  //corotate(pmb,      time, dt, prim, bcc, cons);
  return;
}
//----------------------------------------------------------------------------------------
// Cooling Process
void cooling(MeshBlock *pmb, const Real time, const Real dt,
      const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
      AthenaArray<Real> &cons)
{
  Real rad,phi,z;
  for (int k=pmb->ks; k<=pmb->ke; ++k){
    for (int j=pmb->js; j<=pmb->je; ++j){
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i){
        GetCylCoord(pmb->pcoord,rad,phi,z,i,j,k); // can use: pmb->pcoord->x1v.i? check mesh.hpp.
        if (NON_BAROTROPIC_EOS){
          Real p_over_r    = PoverR(rad,phi,z);
          Real temp        = prim(IEN,k,j,i)/prim(IDN,k,j,i);
          cons(IEN,k,j,i) -= dt * prim(IDN,k,j,i)*(temp-p_over_r)/(gamma_gas-1.0)/tau_relax;
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
// Add planet (2D force for now.)
void potentialwell(MeshBlock *pmb, const Real time, const Real dt,
        const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
        AthenaArray<Real> &cons)
{
  Real rad, phi, z, acc_r, acc_phi, dis, dis_r, dis_phi, mom_r, mom_phi, sec_g, forth_g, sixth_g;
  if (time>=t0pot){
    phi_planet = sqrt(gm0/pow(rad_planet,3))*time; // omega*time, only turn on when no call of corotate
    //phi_planet = PI;
    for (int k=pmb->ks; k<=pmb->ke; ++k){
      for (int j=pmb->js; j<=pmb->je; ++j){
#pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i){
          GetCylCoord(pmb->pcoord,rad,phi,z,i,j,k);
          dis     = sqrt(SQR(rad*cos(phi)-rad_planet*cos(phi_planet))+
                    SQR(rad*sin(phi)-rad_planet*sin(phi_planet)));//dist btw cell&planet
          dis_r   = rad - rad_planet*cos(phi-phi_planet);
          dis_phi = rad_planet*sin(phi-phi_planet);

          //second order gravity
          //sec_g   = gmp/pow(SQR(dis)+SQR(rs),1.5);
          //acc_r   = sec_g*dis_r; // radial acceleration
          //acc_phi = sec_g*dis_phi; // asimuthal acceleration

          //fourth order gravity
          //forth_g = gmp*(5*SQR(rs)+2*SQR(dis))/(2*pow(SQR(rs)+SQR(dis), 2.5));
          //acc_r   = forth_g*dis_r; // radial acceleration
          //acc_phi = forth_g*dis_phi; // asimuthal acceleration

          //sixth order gravity
          sixth_g = gmp*(35*SQR(SQR(rs))+28*SQR(rs)*SQR(dis)+8*SQR(SQR(dis)))/(8*pow(SQR(rs)+SQR(dis), 3.5));
          acc_r   = sixth_g*dis_r; // radial acceleration
          acc_phi = sixth_g*dis_phi; // asimuthal acceleration

          mom_r   = dt*prim(IDN,k,j,i)*acc_r;
          mom_phi = dt*prim(IDN,k,j,i)*acc_phi;
          cons(IM1,k,j,i) -= mom_r;
          cons(IM2,k,j,i) -= mom_phi;
          if (NON_BAROTROPIC_EOS)
            cons(IEN,k,j,i) += (mom_r*prim(IM1,k,j,i) + mom_phi*prim(IM2,k,j,i));

          if (NDUSTFLUIDS > 0){
            for (int n=0;n<NDUSTFLUIDS;n++){
              int dust_id = n;
              int rho_id  = 4*dust_id;
              int v1_id   = rho_id + 1;
              int v2_id   = rho_id + 2;
              int v3_id   = rho_id + 3;
              pmb->pdustfluids->df_cons(v1_id,k,j,i) -= dt*pmb->pdustfluids->df_prim(rho_id,k,j,i)*acc_r;
              pmb->pdustfluids->df_cons(v2_id,k,j,i) -= dt*pmb->pdustfluids->df_prim(rho_id,k,j,i)*acc_phi;
            }
          }
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
// Wavedamping function
void wavedamping(MeshBlock *pmb, const Real time, const Real dt,
        const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
        AthenaArray<Real> &cons) {
  DustFluids *pdf = pmb->pdustfluids;
  Real rad, phi, z, vel10, vel20, vel30, df_vel10, df_vel20, df_vel30;
  int is = pmb->is;
  int ie = pmb->ie;
  int js = pmb->js;
  int je = pmb->je;
  int ks = pmb->ks;
  int ke = pmb->ke;

  for (int k=ks; k<=ke; ++k){
    for (int j=js; j<=je; ++j){
#pragma omp simd
      for (int i=is; i<=ie; ++i){
        GetCylCoord(pmb->pcoord,rad,phi,z,i,j,k);
        if (rad >= x1min && rad < radius_inner_damping){
          Real omega_dyn     = sqrt(gm0/pow(rad,3));
          Real rho0 = DenProfileCyl(rad,phi,z);
          VelProfileCyl(rad,            phi, z, vel10,    vel20,    vel30);
          VelProfileCyl_DustFluids(rad, phi, z, df_vel10, df_vel20, df_vel30);

          // See de Val-Borro et al. 2006 & 2007
          Real R_func        = SQR((rad - radius_inner_damping)/inner_width_damping);
          Real damping_speed = damping_rate*omega_dyn*R_func;
          Real alpha1        = 1./(1. + dt*damping_speed);
          Real alpha2        = 1.-alpha1;

          const Real &rho  = prim(IDN,k,j,i);
          const Real &vel1 = prim(IVX,k,j,i);
          const Real &vel2 = prim(IVY,k,j,i);
          const Real &vel3 = prim(IVZ,k,j,i);

          Real &den  = cons(IDN,k,j,i);
          Real &mom1 = cons(IM1,k,j,i);
          Real &mom2 = cons(IM2,k,j,i);
          Real &mom3 = cons(IM3,k,j,i);
          Real &eng  = cons(IEN,k,j,i);

          den  = den*alpha1 + rho0*alpha2;
          mom1 = mom1*alpha1 + (rho0*vel10)*alpha2;
          mom2 = mom2*alpha1 + (rho0*vel20)*alpha2;
          mom3 = mom3*alpha1 + (rho0*vel30)*alpha2;

          if (NON_BAROTROPIC_EOS) {
            Real p_over_r_0  = PoverR(rad,phi,z);
            Real eng_0       = p_over_r_0*rho0/(gamma_gas - 1.0);
            eng_0           += 0.5*(SQR(rho0*vel10) + SQR(rho0*vel20) + SQR(rho0*vel30))/rho0;
            eng              = eng*alpha1 + eng_0*alpha2;
          }

          if ( NDUSTFLUIDS > 0 ) {
            for (int n=0;n<NDUSTFLUIDS;n++){
              int dust_id = n;
              int rho_id  = 4*dust_id;
              int v1_id   = rho_id + 1;
              int v2_id   = rho_id + 2;
              int v3_id   = rho_id + 3;
              Real df_rho0 = initial_D2G*DenProfileCyl(rad,phi,z);

              const Real &df_rho  = pdf->df_prim(rho_id,k,j,i);
              const Real &df_vel1 = pdf->df_prim(v1_id,k,j,i);
              const Real &df_vel2 = pdf->df_prim(v2_id,k,j,i);
              const Real &df_vel3 = pdf->df_prim(v3_id,k,j,i);

              Real &df_den  = pdf->df_cons(rho_id,k,j,i);
              Real &df_mom1 = pdf->df_cons(v1_id,k,j,i);
              Real &df_mom2 = pdf->df_cons(v2_id,k,j,i);
              Real &df_mom3 = pdf->df_cons(v3_id,k,j,i);

              //df_den  = df_den*alpha1 + df_rho0*alpha2;
              df_mom1 = df_mom1*alpha1 + (df_rho0*df_vel10)*alpha2;
              //df_mom2 = df_mom2*alpha1 + (df_rho0*df_vel20)*alpha2;
              //df_mom3 = df_mom3*alpha1 + (df_rho0*df_vel30)*alpha2;
            }
          }

        }

        if (rad <= x1max && rad > radius_outer_damping){
          Real rho0 = DenProfileCyl(rad,phi,z);
          VelProfileCyl(rad,            phi, z, vel10,    vel20,    vel30);
          VelProfileCyl_DustFluids(rad, phi, z, df_vel10, df_vel20, df_vel30);

          Real omega_dyn     = sqrt(gm0/pow(rad,3));
          Real R_func        = SQR((rad - radius_outer_damping)/outer_width_damping); // See de Val-Borro et al. 2006 & 2007
          Real damping_speed = damping_rate*omega_dyn*R_func;
          Real alpha1        = 1./(1. + dt*damping_speed);
          Real alpha2        = 1.-alpha1;

          const Real &rho = prim(IDN,k,j,i);
          const Real &vel1 = prim(IVX,k,j,i);
          const Real &vel2 = prim(IVY,k,j,i);
          const Real &vel3 = prim(IVZ,k,j,i);

          Real &den  = cons(IDN,k,j,i);
          Real &mom1 = cons(IM1,k,j,i);
          Real &mom2 = cons(IM2,k,j,i);
          Real &mom3 = cons(IM3,k,j,i);
          Real &eng = cons(IEN,k,j,i);

          den  = den*alpha1 + rho0*alpha2;
          mom1 = mom1*alpha1 + (rho0*vel10)*alpha2;
          mom2 = mom2*alpha1 + (rho0*vel20)*alpha2;
          mom3 = mom3*alpha1 + (rho0*vel30)*alpha2;

          if (NON_BAROTROPIC_EOS) {
            Real p_over_r_0  = PoverR(rad,phi,z);
            Real eng_0       = p_over_r_0*rho0/(gamma_gas - 1.0);
            eng_0           += 0.5*(SQR(rho0*vel10) + SQR(rho0*vel20) + SQR(rho0*vel30))/rho0;
            eng              = eng*alpha1 + eng_0*alpha2;
          }

          //if ( NDUSTFLUIDS > 0 ) {
            //for (int n=0;n<NDUSTFLUIDS;n++){
              //int dust_id = n;
              //int rho_id  = 4*dust_id;
              //int v1_id   = rho_id + 1;
              //int v2_id   = rho_id + 2;
              //int v3_id   = rho_id + 3;
              //Real df_rho0 = initial_D2G*DenProfileCyl(rad,phi,z);

              //const Real &df_rho  = pdf->df_prim(rho_id,k,j,i);
              //const Real &df_vel1 = pdf->df_prim(v1_id,k,j,i);
              //const Real &df_vel2 = pdf->df_prim(v2_id,k,j,i);
              //const Real &df_vel3 = pdf->df_prim(v3_id,k,j,i);

              //Real &df_den  = pdf->df_cons(rho_id,k,j,i);
              //Real &df_mom1 = pdf->df_cons(v1_id,k,j,i);
              //Real &df_mom2 = pdf->df_cons(v2_id,k,j,i);
              //Real &df_mom3 = pdf->df_cons(v3_id,k,j,i);

              //df_den  = df_den*alpha1  + 0.0*alpha2;
              //df_mom1 = df_mom1*alpha1 + (0.0*df_vel10)*alpha2;
              //df_mom2 = df_mom2*alpha1 + (0.0*df_vel20)*alpha2;
              //df_mom3 = df_mom3*alpha1 + (0.0*df_vel30)*alpha2;
            //}
          //}

        }

      }
    }
  }
  return;
}


//----------------------------------------------------------------------------------------
// Corotating reference
void corotate(MeshBlock *pmb, const Real time, const Real dt,
        const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
        AthenaArray<Real> &cons)
{
  DustFluids *pdf = pmb->pdustfluids;
  Real rad, phi, z;
  for (int k=pmb->ks; k<=pmb->ke; ++k){
    for (int j=pmb->js; j<=pmb->je; ++j){
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i){
        GetCylCoord(pmb->pcoord,rad,phi,z,i,j,k);
        Real work_planet = SQR(omega_p)*rad_planet;
        //Real acen  = 3*SQR(omega_p)*(rad - rad_planet);
        //Real acor1 = 2*omega_p  * prim(IVY,k,j,i); //radial coriolis
        //Real acor2 = -2*omega_p * prim(IVX,k,j,i);
        //cons(IM1,k,j,i) += dt * prim(IDN,k,j,i)* (acen+acor1);
        //cons(IM2,k,j,i) += dt * prim(IDN,k,j,i)* acor2;
        cons(IM2,k,j,i) += dt * prim(IDN,k,j,i)*work_planet;
        if (NDUSTFLUIDS > 0){
          for (int n=0;n<NDUSTFLUIDS;n++){
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;
            //Real acor1_dust =  2*omega_p * pmb->pdustfluids->df_prim(v2_id,k,j,i); //radial coriolis
            //Real acor2_dust = -2*omega_p * pmb->pdustfluids->df_prim(v1_id,k,j,i);
            //pmb->pdustfluids->df_cons(v1_id,k,j,i) += dt*pmb->pdustfluids->df_prim(rho_id,k,j,i)*(acen+acor1_dust);
            //pmb->pdustfluids->df_cons(v2_id,k,j,i) += dt*pmb->pdustfluids->df_prim(rho_id,k,j,i)*acor2_dust;
            pdf->df_cons(v2_id,k,j,i) += dt*pdf->df_prim(rho_id,k,j,i)*work_planet;
          }
        }
      }
    }
  }
  return;
}


void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k) {
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    rad=pco->x1v(i);
    phi=pco->x2v(j);
    z=pco->x3v(k);
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    rad=std::abs(pco->x1v(i)*std::sin(pco->x2v(j)));
    phi=pco->x3v(i);
    z=pco->x1v(i)*std::cos(pco->x2v(j));
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \f  computes density in cylindrical coordinates
Real RadialD2G(const Real rad, const Real initial_dust2gas, const Real slope)
{
  Real dust2gas = initial_dust2gas*std::pow(rad/r0,slope);
  return dust2gas;
}

Real DenProfileCyl(const Real rad, const Real phi, const Real z) {
  Real den;
  Real p_over_r = p0_over_r0;
  if (NON_BAROTROPIC_EOS) p_over_r = PoverR(rad, phi, z);
  Real denmid = rho0*std::pow(rad/r0,dslope);
  Real dentem = denmid*std::exp(gm0/p_over_r*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
  den         = dentem;
  return std::max(den,dfloor);
}

Real DenProfileCyl_Gaussian(const Real rad, const Real phi, const Real z) {
  Real den;
  Real p_over_r = p0_over_r0;
  if (NON_BAROTROPIC_EOS) p_over_r = PoverR(rad, phi, z);
  Real denmid = rho0*std::pow(rad/r0,dslope);
  denmid      = denmid * (1 + A0 * std::exp(-0.5 * std::pow((rad - rn)/(r0 * dwidth),2.0)));
  Real dentem = denmid*std::exp(gm0/p_over_r*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
  den         = dentem;
  return std::max(den,dfloor);
}

//----------------------------------------------------------------------------------------
//! \f  computes pressure/density in cylindrical coordinates

Real PoverR(const Real rad, const Real phi, const Real z) {
  Real poverr;
  poverr = p0_over_r0*std::pow(rad/r0, pslope);
  return poverr;
}

//----------------------------------------------------------------------------------------
//! \f  computes rotational velocity in cylindrical coordinates

void VelProfileCyl(const Real rad, const Real phi, const Real z,
                   Real &v1, Real &v2, Real &v3) {
  Real p_over_r = PoverR(rad, phi, z);
  Real vel = (dslope+pslope)*p_over_r/(gm0/rad) + (1.0+pslope) - pslope*rad/std::sqrt(rad*rad+z*z);
  //Real vel = (dslope+pslope)*p_over_r/(gm0/rad) + 1.0;
  vel = std::sqrt(gm0/rad)*std::sqrt(vel);
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    v1=0.0;
    v2=vel;
    v3=0.0;
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    v1=0.0;
    v2=0.0;
    v3=vel;
  }
  return;
}

void VelProfileCyl_DustFluids(const Real rad, const Real phi, const Real z,
                   Real &v1, Real &v2, Real &v3) {
  Real vel = std::sqrt(gm0/rad);
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    v1=0.0;
    v2=vel;
    v3=0.0;
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    v1=0.0;
    v2=0.0;
    v3=vel;
  }
  return;
}


void Linear_interpolate(const Real x0, const Real x1, const Real y0, const Real y1,
    Real &x, Real &y){
  y = (x-x1)*y0/(x0-x1) + (x-x0)*y1/(x1-x0);
  return;
}

void Keplerian_interpolate(const Real r_active, const Real r_ghost, const Real vphi_active,
    Real &vphi_ghost){
  vphi_ghost = vphi_active*std::sqrt(r_active/r_ghost);
  return;
}

void Density_interpolate(const Real r_active, const Real r_ghost, const Real rho_active,
    const Real slope, Real &rho_ghost){
  rho_ghost = rho_active * std::pow(r_ghost/r_active, slope);
  return;
}

void Vr_interpolate_inner_nomatter(const Real r_active, const Real r_ghost, const Real sigma_active,
    const Real sigma_ghost, const Real vr_active, Real &vr_ghost){
  vr_ghost = vr_active <= 0.0 ? (sigma_active*r_active*vr_active)/(sigma_ghost*r_ghost) : 0.0;
  return;
}

void Vr_interpolate_outer_nomatter(const Real r_active, const Real r_ghost, const Real sigma_active,
    const Real sigma_ghost, const Real vr_active, Real &vr_ghost){
  if (sigma_active < 1e-20)
    vr_ghost = vr_active >= 0.0 ? ((sigma_active+1e-20)*r_active*vr_active)/(sigma_ghost*r_ghost) : 0.0;
  else
    vr_ghost = vr_active >= 0.0 ? (sigma_active*r_active*vr_active)/(sigma_ghost*r_ghost) : 0.0;
  //vr_ghost = (sigma_active*r_active*vr_active)/(sigma_ghost*r_ghost);
  return;
}

} // namespace

//----------------------------------------------------------------------------------------
//!\f: User-defined boundary Conditions: sets solution in ghost zones to initial values
//

void InnerX1_NoMatterInput(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad, phi, z, rad_ac, phi_ac, z_ac;
  Real v1, v2, v3, df_v1, df_v2, df_v3, df_v1_ac, df_v2_ac, df_v3_ac;
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        GetCylCoord(pco, rad_ac, phi_ac, z_ac, il,   j, k);
        GetCylCoord(pco, rad,    phi,    z,    il-i, j, k);
        VelProfileCyl(rad, phi, z, v1, v2, v3);
        //prim(IDN,k,j,il-i) = DenProfileCyl(rad,phi,z);
        //prim(IM1,k,j,il-i) = v1;
        //prim(IM2,k,j,il-i) = v2;
        //prim(IM3,k,j,il-i) = v3;

        Real &gas_den = prim(IDN, k, j, il-i);
        Real &gas_v1  = prim(IM1, k, j, il-i);
        Real &gas_v2  = prim(IM2, k, j, il-i);
        Real &gas_v3  = prim(IM3, k, j, il-i);

        Real &gas_den_ac = prim(IDN, k, j, il);
        Real &gas_v1_ac  = prim(IM1, k, j, il);
        Real &gas_v2_ac  = prim(IM2, k, j, il);
        Real &gas_v3_ac  = prim(IM3, k, j, il);

        gas_den = DenProfileCyl(rad,phi,z);
        //gas_den = gas_den_ac;
        gas_v2  = v2;
        gas_v3  = gas_v3_ac;
        Vr_interpolate_inner_nomatter(rad_ac, rad, gas_den_ac, gas_den, gas_v1_ac, gas_v1);
        //gas_v1 = gas_v1_ac;


        if (NON_BAROTROPIC_EOS)
          prim(IEN,k,j,il-i) = PoverR(rad, phi, z)*gas_den;
          //prim(IEN,k,j,il-i) = PoverR(rad, phi, z)*prim(IDN,k,j,il-i);

        if (NDUSTFLUIDS > 0) {
          VelProfileCyl_DustFluids(rad,    phi,    z,    df_v1,    df_v2,    df_v3);
          VelProfileCyl_DustFluids(rad_ac, phi_ac, z_ac, df_v1_ac, df_v2_ac, df_v3_ac);
          for (int n=0; n<NDUSTFLUIDS; n++) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real &dust_den = pmb->pdustfluids->df_prim(rho_id, k, j, il-i);
            Real &dust_v1  = pmb->pdustfluids->df_prim(v1_id,  k, j, il-i);
            Real &dust_v2  = pmb->pdustfluids->df_prim(v2_id,  k, j, il-i);
            Real &dust_v3  = pmb->pdustfluids->df_prim(v3_id,  k, j, il-i);

            Real &dust_den_ac = pmb->pdustfluids->df_prim(rho_id, k, j, il);
            Real &dust_v1_ac  = pmb->pdustfluids->df_prim(v1_id,  k, j, il);
            Real &dust_v2_ac  = pmb->pdustfluids->df_prim(v2_id,  k, j, il);
            Real &dust_v3_ac  = pmb->pdustfluids->df_prim(v3_id,  k, j, il);

            //dust_den = dust_den_ac;
            //dust_v2  = df_v2;
            //dust_v3  = dust_v3_ac;

            dust_den = initial_D2G*prim(IDN,k,j,il-i);
            dust_v2  = df_v2;
            dust_v3  = df_v3;

            Vr_interpolate_inner_nomatter(rad_ac, rad, dust_den_ac, dust_den, dust_v1_ac, dust_v1);

            //dust_den = dust_den_ac;
            //dust_v1  = dust_v1_ac;
            //dust_den = dust_den_ac;

            //Density_interpolate(rad_ac,   rad, dust_den_ac, dslope, dust_den);
            //Vr_interpolate_inner_powerlaw(rad_ac,  rad, dust_v1_ac,  dslope, dust_v1);
            //dust_v1 = dust_v1_ac;
          }

        }
      }
    }
  }
}

void OuterX1_NoMatterInput(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad, phi, z, rad_ac, phi_ac, z_ac;
  Real v1, v2, v3, df_v1, df_v2, df_v3;
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        GetCylCoord(pco, rad_ac, phi_ac, z_ac, iu,   j, k);
        GetCylCoord(pco, rad,    phi,    z,    iu+i, j, k);
        VelProfileCyl(rad, phi, z, v1, v2, v3);

        Real &gas_den = prim(IDN, k, j, iu+i);
        Real &gas_v1  = prim(IM1, k, j, iu+i);
        Real &gas_v2  = prim(IM2, k, j, iu+i);
        Real &gas_v3  = prim(IM3, k, j, iu+i);

        Real &gas_den_ac = prim(IDN, k, j, iu);
        Real &gas_v1_ac  = prim(IM1, k, j, iu);
        Real &gas_v2_ac  = prim(IM2, k, j, iu);
        Real &gas_v3_ac  = prim(IM3, k, j, iu);

        gas_den = DenProfileCyl(rad,phi,z);
        //gas_den = gas_den_ac;
        gas_v2 = v2;
        gas_v3 = gas_v3_ac;
        Vr_interpolate_outer_nomatter(rad_ac, rad, gas_den_ac, gas_den, gas_v1_ac, gas_v1);
        //gas_v1 = gas_v1_ac;

        if (NON_BAROTROPIC_EOS)
          prim(IEN,k,j,iu+i) = PoverR(rad, phi, z)*prim(IDN,k,j,iu+i);

        if (NDUSTFLUIDS > 0) {
          GetCylCoord(pco, rad_ac, phi_ac, z_ac, iu,   j, k);
          GetCylCoord(pco, rad,    phi,    z,    iu+i, j, k);
          VelProfileCyl_DustFluids(rad, phi, z, df_v1, df_v2, df_v3);
          for (int n=0; n<NDUSTFLUIDS; n++) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real &dust_den = pmb->pdustfluids->df_prim(rho_id, k, j, iu+i);
            Real &dust_v1  = pmb->pdustfluids->df_prim(v1_id,  k, j, iu+i);
            Real &dust_v2  = pmb->pdustfluids->df_prim(v2_id,  k, j, iu+i);
            Real &dust_v3  = pmb->pdustfluids->df_prim(v3_id,  k, j, iu+i);

            Real &dust_den_ac = pmb->pdustfluids->df_prim(rho_id, k, j, iu);
            Real &dust_v1_ac  = pmb->pdustfluids->df_prim(v1_id,  k, j, iu);
            Real &dust_v2_ac  = pmb->pdustfluids->df_prim(v2_id,  k, j, iu);
            Real &dust_v3_ac  = pmb->pdustfluids->df_prim(v3_id,  k, j, iu);

            //dust_v1  = dust_v1_ac > 0.0 ? dust_v1_ac : 0.0;
            dust_den = 0.0;
            //Keplerian_interpolate(rad_ac, rad, dust_v2_ac,  dust_v2);
            //dust_v1  = dust_v1_ac;
            dust_v2  = df_v2;
            dust_v3  = dust_v3_ac;
            Vr_interpolate_outer_nomatter(rad_ac, rad, dust_den_ac, dust_den, dust_v1_ac, dust_v1);

            //dust_den = dust_den_ac;
            //dust_v1  = dust_v1_ac;
            //dust_den = dust_den_ac;

            //Density_interpolate(rad_ac,   rad, dust_den_ac, dslope, dust_den);
            //Vr_interpolate_inner_powerlaw(rad_ac,  rad, dust_v1_ac,  dslope, dust_v1);
            //dust_v1 = dust_v1_ac;
          }
        }

      }
    }
  }
}



void MeshBlock::UserWorkInLoop() {
  Real initial_w_IDN(0.0);
  Real initial_u_IM1(0.0), initial_u_IM2(0.0), initial_u_IM3(0.0);
  Real rad(0.0), phi(0.0), z(0.0);
  Real v1(0.0), v2(0.0), v3(0.0), df_v1, df_v2, df_v3;
  Real igm1 = 1.0/(gamma_gas - 1.0);
  //MeshBlock *pmb         = pmy_mesh;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        GetCylCoord(pcoord,rad,phi,z,i,j,k); // convert to cylindrical coordinates
        // compute initial conditions in cylindrical coordinates
        //initial_w_IDN = DenProfileCyl(rad,phi,z);
        //VelProfileCyl(rad, phi, z, v1, v2, v3);
        //VelProfileCyl_DustFluids(rad, phi, z, df_v1, df_v2, df_v3);

        //phydro->w(IVX,k,j,i) = 0.0;
        //phydro->w(IVY,k,j,i) = v2;
        //phydro->w(IDN,k,j,i) = initial_w_IDN;

        if (NON_BAROTROPIC_EOS) {
          Real p_over_r        = PoverR(rad,phi,z);
          phydro->w(IPR,k,j,i) = p_over_r*phydro->w(IDN,k,j,i);
          //Real di = 1./initial_w_IDN;
          //phydro->u(IEN,k,j,i) = phydro->w(IPR,k,j,i)/igm1;
          //phydro->u(IEN,k,j,i) += 0.5*(SQR(initial_u_IM1)+SQR(initial_u_IM2)
                                      //+ SQR(initial_u_IM3))/initial_w_IDN;
        }

        //if (NDUSTFLUIDS > 0)
          //for (int n=0; n<NDUSTFLUIDS; n++) {
            //int dust_id = n;
            //int rho_id  = 4*dust_id;
            //int v1_id   = rho_id + 1;
            //int v2_id   = rho_id + 2;
            //int v3_id   = rho_id + 3;
            //pdustfluids->df_prim(rho_id,k,j,i) = initial_D2G*phydro->w(IDN,k,j,i);
            //pdustfluids->df_prim(v1_id,k,j,i)  = df_v1;
            //pdustfluids->df_prim(v2_id,k,j,i)  = df_v2;
        //}

      }
    }
  }
  return;
}
