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
Real PoverR(const Real rad,        const Real phi, const Real z);
Real RadialD2G(const Real rad, const Real initial_dust2gas, const Real slope);
void GasVelProfileCyl(const Real rad, const Real phi, const Real z,
                   Real &v1, Real &v2, Real &v3);
void DustVelProfileCyl(const Real rad, const Real phi, const Real z,
                   Real &v1, Real &v2, Real &v3);
void GasVelProfileCyl_NSH(const Real SN, const Real QN, const Real Psi,
    const Real rad, const Real phi, const Real z, Real &v1, Real &v2, Real &v3);
void DustVelProfileCyl_NSH(const Real Ts, const Real SN, const Real QN, const Real Psi,
    const Real rad, const Real phi, const Real z, Real &v1, Real &v2, Real &v3);
void Keplerian_interpolate(const Real r_active, const Real r_ghost, const Real vphi_active,
    Real &vphi_ghost);
void Density_interpolate(const Real r_active, const Real r_ghost, const Real rho_active,
    const Real slope, Real &rho_ghost);
// problem parameters which are useful to make global to this file
//
Real Keplerian_velocity(const Real rad);
Real Delta_gas_vr(const Real vk,    const Real SN, const Real QN,    const Real Psi);
Real Delta_gas_vphi(const Real vk,  const Real SN, const Real QN,    const Real Psi);
Real Delta_dust_vr(const Real ts,   const Real vk, const Real d_vgr, const Real d_vgphi);
Real Delta_dust_vphi(const Real ts, const Real vk, const Real d_vgr, const Real d_vgphi);

// User Sources
void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
void InnerWavedamping(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
void OuterWavedamping(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
void LocalIsothermalEOS(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);

Real gm0, r0, rho_0, dslope, p0_over_r0, pslope, gamma_gas, dfloor, user_dt, iso_cs2_r0;
Real tau_relax, rs, gmp, rad_planet, phi_planet, t0pot, omega_p, Bump_flag, A0, dwidth, rn, rand_amp, dust_dens_slope;
Real x1min, x1max, tau_damping, damping_rate;
Real radius_inner_damping, radius_outer_damping, inner_ratio_region, outer_ratio_region,
     inner_width_damping, outer_width_damping;

Real SN_const(0.0), QN_const(0.0), Psi_const(0.0);

Real initial_D2G[NDUSTFLUIDS], Stokes_number[NDUSTFLUIDS];
Real eta_gas, beta_gas, ks_gas;
bool Damping_Flag;

// User defined time step
Real MyTimeStep(MeshBlock *pmb);
} // namespace

// User-defined boundary conditions for disk simulations
void InnerX1_NSH(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void OuterX1_NSH(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
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
  gm0 = pin->GetOrAddReal("problem", "GM", 0.0);
  r0  = pin->GetOrAddReal("problem", "r0", 1.0);

  // Get parameters for initial density and velocity
  rho_0           = pin->GetReal("problem",      "rho0");
  dslope          = pin->GetOrAddReal("problem", "dslope", 0.0);
  dust_dens_slope = pin->GetOrAddReal("problem", "dust_dens_slope", 0.0);

  // The parameters of the amplitude of random perturbation on the radial velocity
  rand_amp     = pin->GetOrAddReal("problem", "random_vel_r_amp", 0.0);
  Damping_Flag = pin->GetBoolean("problem",   "Damping_Flag");

  // The parameters of damping zones
  x1min = pin->GetReal("mesh", "x1min");
  x1max = pin->GetReal("mesh", "x1max");

  //ratio of the orbital periods between the edge of the wave-killing zone and the corresponding edge of the mesh
  inner_ratio_region = pin->GetOrAddReal("problem", "inner_dampingregion_ratio", 2.5);
  outer_ratio_region = pin->GetOrAddReal("problem", "outer_dampingregion_ratio", 1.2);

  radius_inner_damping = x1min*pow(inner_ratio_region, 2./3.);
  radius_outer_damping = x1max*pow(outer_ratio_region, -2./3.);

  inner_width_damping = radius_inner_damping - x1min;
  outer_width_damping = x1max - radius_inner_damping;

  // The normalized wave damping timescale, in unit of dynamical timescale.
  damping_rate = pin->GetOrAddReal("problem", "damping_rate", 1.0);

  // The parameters of one planet
  tau_relax  = pin->GetOrAddReal("hydro",   "tau_relax",  0.01);
  rad_planet = pin->GetOrAddReal("problem", "rad_planet", 1.0); // radial position of the planet
  phi_planet = pin->GetOrAddReal("problem", "phi_planet", 0.0); // azimuthal position of the planet
  t0pot      = pin->GetOrAddReal("problem", "t0pot",      0.0); // time to put in the planet
  gmp        = pin->GetOrAddReal("problem", "GMp",        0.0); // GM of the planet
  rs         = pin->GetOrAddReal("problem", "rs",         0.1); // softening length of the gravitational potential of planets
  user_dt    = pin->GetOrAddReal("problem", "user_dt",    0.0);
  omega_p    = sqrt(gm0/pow(rad_planet, 3));          // The Omega of planetary orbit

  // Dust to gas ratio && dust stopping time
  if (NDUSTFLUIDS > 0) {
    for (int n=0; n<NDUSTFLUIDS; n++) {
      initial_D2G[n]   = pin->GetReal("dust", "initial_D2G_"      + std::to_string(n+1));
      Stokes_number[n] = pin->GetReal("dust", "internal_density_" + std::to_string(n+1));
    }
  }

  // Get parameters of initial pressure and cooling parameters
  if (NON_BAROTROPIC_EOS) {
    iso_cs2_r0 = SQR(pin->GetReal("hydro", "iso_sound_speed"));
    pslope     = pin->GetReal("problem",   "pslope");
    gamma_gas  = pin->GetReal("hydro",     "gamma");
  } else {
    iso_cs2_r0 = SQR(pin->GetReal("hydro", "iso_sound_speed"));
  }

  Real float_min = std::numeric_limits<float>::min();
  dfloor         = pin->GetOrAddReal("hydro", "dfloor", (1024*(float_min)));

  eta_gas  = 0.5*iso_cs2_r0*(pslope + dslope);
  beta_gas = std::sqrt(1.0 + 2.0*eta_gas);
  ks_gas   = 0.5 * beta_gas;

  // Dust to gas ratio && dust stopping time
  if (NDUSTFLUIDS > 0) {
    for (int n=0; n<NDUSTFLUIDS; n++) {
      SN_const += (initial_D2G[n])/(1.0 + SQR(Stokes_number[n]));
      QN_const += (initial_D2G[n]*Stokes_number[n])/(1.0 + SQR(Stokes_number[n]));
    }
    Psi_const = 1.0/((SN_const + beta_gas)*(SN_const + 2.0*ks_gas) + SQR(QN_const));
  }

  // Enroll damping zone and local isothermal equation of state
  EnrollUserExplicitSourceFunction(MySource);

  // Enroll user-defined boundary condition
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user"))
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, InnerX1_NSH);

  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user"))
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, OuterX1_NSH);

  // Enroll user-defined time step
  if (user_dt > 0.0)
    EnrollUserTimeStepFunction(MyTimeStep);

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Initializes Keplerian accretion disk.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real rad(0.0),   phi(0.0),   z(0.0);
  Real g_v1(0.0),  g_v2(0.0),  g_v3(0.0);
  Real df_v1(0.0), df_v2(0.0), df_v3(0.0);

  Real igm1 = 1.0/(gamma_gas - 1.0);
  // Initialize density and momenta
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real &gas_den    = phydro->u(IDN, k, j, i);
        Real &gas_m1     = phydro->u(IM1, k, j, i);
        Real &gas_m2     = phydro->u(IM2, k, j, i);
        Real &gas_m3     = phydro->u(IM3, k, j, i);

        // convert to cylindrical coordinates
        GetCylCoord(pcoord, rad, phi, z, i, j, k);

        // compute initial conditions in cylindrical coordinates
        gas_den          = DenProfileCyl(rad, phi, z);
        Real inv_gas_den = 1.0/gas_den;

        Real SN(0.0), QN(0.0), Psi(0.0);
        for (int n=0; n<NDUSTFLUIDS; n++) {
          int dust_id    = n;
          int rho_id     = 4*dust_id;
          Real &dust_den = pdustfluids->df_cons(rho_id, k, j, i);
          dust_den       = initial_D2G[dust_id] * gas_den;

          SN += (dust_den*inv_gas_den)/(1.0 + SQR(Stokes_number[n]));
          QN += (dust_den*inv_gas_den*Stokes_number[n])/(1.0 + SQR(Stokes_number[n]));
        }
        Psi = 1.0/((SN + beta_gas)*(SN + 2.0*ks_gas) + SQR(QN));

        //GasVelProfileCyl_NSH(SN, QN, Psi, rad, phi, z, g_v1, g_v2, g_v3);
        GasVelProfileCyl_NSH(SN_const, QN_const, Psi_const, rad, phi, z, g_v1, g_v2, g_v3);
        gas_m1 = gas_den * g_v1;
        gas_m2 = gas_den * g_v2;
        gas_m3 = gas_den * g_v3;

        if (NON_BAROTROPIC_EOS) {
          Real &gas_erg  = phydro->u(IEN, k, j, i);
          Real p_over_r  = PoverR(rad, phi, z);
          gas_erg        = p_over_r * gas_den * igm1;
          gas_erg       += 0.5 * (SQR(gas_m1) + SQR(gas_m2) + SQR(gas_m3))/gas_den;
        }

        for (int n=0; n<NDUSTFLUIDS; n++) {
          int dust_id = n;
          int rho_id  = 4*dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;

          Real &dust_den = pdustfluids->df_cons(rho_id, k, j, i);
          Real &dust_m1  = pdustfluids->df_cons(v1_id,  k, j, i);
          Real &dust_m2  = pdustfluids->df_cons(v2_id,  k, j, i);
          Real &dust_m3  = pdustfluids->df_cons(v3_id,  k, j, i);

          //DustVelProfileCyl_NSH(Stokes_number[dust_id], SN, QN, Psi, rad, phi, z, df_v1, df_v2, df_v3);
          DustVelProfileCyl_NSH(Stokes_number[dust_id], SN_const, QN_const, Psi_const, rad, phi, z, df_v1, df_v2, df_v3);
          dust_m1 = dust_den * df_v1;
          dust_m2 = dust_den * df_v2;
          dust_m3 = dust_den * df_v3;
        }

      }
    }
  }
  return;
}

namespace {
//----------------------------------------------------------------------------------------
void MySource(MeshBlock *pmb, const Real time, const Real dt,
    const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons)
{
  if (Damping_Flag) {
    InnerWavedamping(pmb, time, dt, prim, bcc, cons);
    OuterWavedamping(pmb, time, dt, prim, bcc, cons);
  }

  if (NON_BAROTROPIC_EOS)
    LocalIsothermalEOS(pmb, time, dt, prim, bcc, cons);

  return;
}

Real MyTimeStep(MeshBlock *pmb)
{
  Real min_user_dt = user_dt;
  return min_user_dt;
}

//----------------------------------------------------------------------------------------
// Wavedamping function
void InnerWavedamping(MeshBlock *pmb, const Real time, const Real dt,
        const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
        AthenaArray<Real> &cons) {

  DustFluids *pdf = pmb->pdustfluids;
  Real rad, phi, z, gas_v1_0, gas_v2_0, gas_v3_0, dust_v1_0, dust_v2_0, dust_v3_0;

  int is = pmb->is; int ie = pmb->ie;
  int js = pmb->js; int je = pmb->je;
  int ks = pmb->ks; int ke = pmb->ke;

  Real igm1           = 1.0/(gamma_gas - 1.0);
  Real inv_inner_damp = 1.0/inner_width_damping;

  for (int k=ks; k<=ke; ++k){
    for (int j=js; j<=je; ++j){
#pragma omp simd
      for (int i=is; i<=ie; ++i){
        GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);

        if (rad >= x1min && rad < radius_inner_damping) {
          Real omega_dyn     = sqrt(gm0/(rad*rad*rad));
          Real gas_den_0     = DenProfileCyl(rad, phi, z);
          Real inv_gas_den_0 = 1.0/gas_den_0;

          //Real SN(0.0), QN(0.0), Psi(0.0);
          //for (int n=0; n<NDUSTFLUIDS; n++) {
            //int dust_id     = n;
            //Real dust_den_0 = initial_D2G[dust_id] * gas_den_0;

            //SN += (dust_den_0*inv_gas_den_0)/(1.0 + SQR(Stokes_number[n]));
            //QN += (dust_den_0*inv_gas_den_0*Stokes_number[n])/(1.0 + SQR(Stokes_number[n]));
          //}
          //Psi = 1.0/((SN + beta_gas)*(SN + 2.0*ks_gas) + SQR(QN));

          //GasVelProfileCyl_NSH(SN, QN, Psi, rad, phi, z, gas_v1_0, gas_v2_0, gas_v3_0);
          GasVelProfileCyl_NSH(SN_const, QN_const, Psi_const, rad, phi, z, gas_v1_0, gas_v2_0, gas_v3_0);

          // See de Val-Borro et al. 2006 & 2007
          Real R_func        = SQR((rad - radius_inner_damping)*inv_inner_damp);
          Real damping_speed = damping_rate*omega_dyn*R_func;
          Real alpha_ori     = 1.0/(1.0 + dt*damping_speed);
          //Real alpha_ori     = 1.0/(1.0 + damping_speed);
          Real alpha_dam     = 1.0 - alpha_ori;

          Real &gas_den        = cons(IDN, k, j, i);
          Real &gas_m1         = cons(IM1, k, j, i);
          Real &gas_m2         = cons(IM2, k, j, i);
          Real &gas_m3         = cons(IM3, k, j, i);
          Real inv_den_gas_ori = 1.0/gas_den;

          gas_den = gas_den*alpha_ori                         + gas_den_0*alpha_dam;
          gas_m1  = gas_den*(gas_m1*inv_den_gas_ori*alpha_ori + gas_v1_0*alpha_dam);
          gas_m2  = gas_den*(gas_m2*inv_den_gas_ori*alpha_ori + gas_v2_0*alpha_dam);
          //gas_m3  = gas_den*(gas_m3*inv_den_gas_ori*alpha_ori + gas_v3_0*alpha_dam);

          //if (NON_BAROTROPIC_EOS) {
            //Real gas_erg_0  = PoverR(rad,phi,z)*gas_den_0*igm1;
            //gas_erg_0      += 0.5*(SQR(gas_den_0*gas_v1_0) + SQR(gas_den_0*gas_v2_0) + SQR(gas_den_0*gas_v3_0))/gas_den_0;

            //Real &gas_erg   = cons(IEN,k,j,i);
            //gas_erg         = gas_erg*alpha_ori + gas_erg_0*alpha_dam;
          //}

          for (int n=0; n<NDUSTFLUIDS; n++) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real dust_den_0 = initial_D2G[dust_id]*gas_den_0;
            //DustVelProfileCyl_NSH(Stokes_number[dust_id], SN, QN, Psi, rad, phi, z, dust_v1_0, dust_v2_0, dust_v3_0);
            DustVelProfileCyl_NSH(Stokes_number[dust_id], SN_const, QN_const, Psi_const, rad, phi, z, dust_v1_0, dust_v2_0, dust_v3_0);

            Real &dust_den        = pdf->df_cons(rho_id, k, j, i);
            Real &dust_m1         = pdf->df_cons(v1_id,  k, j, i);
            Real &dust_m2         = pdf->df_cons(v2_id,  k, j, i);
            Real &dust_m3         = pdf->df_cons(v3_id,  k, j, i);
            Real inv_den_dust_ori = 1./dust_den;

            dust_den = dust_den*alpha_ori                           + dust_den_0*alpha_dam;
            dust_m1  = dust_den*(dust_m1*inv_den_dust_ori*alpha_ori + dust_v1_0*alpha_dam);
            dust_m2  = dust_den*(dust_m2*inv_den_dust_ori*alpha_ori + dust_v2_0*alpha_dam);
            //dust_m3  = dust_den*(dust_m3*inv_den_dust_ori*alpha_ori + dust_v3_0*alpha_dam);
          }
        }

      }
    }
  }
  return;
}


void OuterWavedamping(MeshBlock *pmb, const Real time, const Real dt,
        const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
        AthenaArray<Real> &cons) {

  DustFluids *pdf = pmb->pdustfluids;
  Real rad, phi, z, gas_v1_0, gas_v2_0, gas_v3_0, dust_v1_0, dust_v2_0, dust_v3_0;

  int is = pmb->is; int ie = pmb->ie;
  int js = pmb->js; int je = pmb->je;
  int ks = pmb->ks; int ke = pmb->ke;

  Real igm1           = 1.0/(gamma_gas - 1.0);
  Real inv_outer_damp = 1.0/outer_width_damping;

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);

        if (rad <= x1max && rad > radius_outer_damping) {
          Real omega_dyn     = sqrt(gm0/(rad*rad*rad));
          Real gas_den_0     = DenProfileCyl(rad, phi, z);
          Real inv_gas_den_0 = 1.0/gas_den_0;

          //Real SN(0.0), QN(0.0), Psi(0.0);
          //for (int n=0; n<NDUSTFLUIDS; n++) {
            //int dust_id     = n;
            //Real dust_den_0 = initial_D2G[dust_id] * gas_den_0;

            //SN += (dust_den_0*inv_gas_den_0)/(1.0 + SQR(Stokes_number[n]));
            //QN += (dust_den_0*inv_gas_den_0*Stokes_number[n])/(1.0 + SQR(Stokes_number[n]));
          //}
          //Psi = 1.0/((SN + beta_gas)*(SN + 2.0*ks_gas) + SQR(QN));

          //GasVelProfileCyl_NSH(SN, QN, Psi, rad, phi, z, gas_v1_0, gas_v2_0, gas_v3_0);
          GasVelProfileCyl_NSH(SN_const, QN_const, Psi_const, rad, phi, z, gas_v1_0, gas_v2_0, gas_v3_0);

          // See de Val-Borro et al. 2006 & 2007
          Real R_func        = SQR((rad - radius_outer_damping)*inv_outer_damp);
          Real damping_speed = damping_rate*omega_dyn*R_func;
          Real alpha_ori     = 1.0/(1.0 + dt*damping_speed);
          //Real alpha_ori     = 1.0/(1.0 + damping_speed);
          Real alpha_dam     = 1.0 - alpha_ori;

          Real &gas_den        = cons(IDN, k, j, i);
          Real &gas_m1         = cons(IM1, k, j, i);
          Real &gas_m2         = cons(IM2, k, j, i);
          Real &gas_m3         = cons(IM3, k, j, i);
          Real inv_den_gas_ori = 1.0/gas_den;

          gas_den = gas_den*alpha_ori                         + gas_den_0*alpha_dam;
          gas_m1  = gas_den*(gas_m1*inv_den_gas_ori*alpha_ori + gas_v1_0*alpha_dam);
          gas_m2  = gas_den*(gas_m2*inv_den_gas_ori*alpha_ori + gas_v2_0*alpha_dam);
          //gas_m3  = gas_den*(gas_m3*inv_den_gas_ori*alpha_ori + gas_v3_0*alpha_dam);

          //if (NON_BAROTROPIC_EOS) {
            //Real gas_erg_0  = PoverR(rad,phi,z)*gas_den_0*igm1;
            //gas_erg_0      += 0.5*(SQR(gas_den_0*gas_v1_0) + SQR(gas_den_0*gas_v2_0) + SQR(gas_den_0*gas_v3_0))/gas_den_0;

            //Real &gas_erg   = cons(IEN,k,j,i);
            //gas_erg         = gas_erg*alpha_ori + gas_erg_0*alpha_dam;
          //}

          for (int n=0; n<NDUSTFLUIDS; n++) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real dust_den_0 = initial_D2G[dust_id]*gas_den_0;
            //DustVelProfileCyl_NSH(Stokes_number[dust_id], SN, QN, Psi, rad, phi, z, dust_v1_0, dust_v2_0, dust_v3_0);
            DustVelProfileCyl_NSH(Stokes_number[dust_id], SN_const, QN_const, Psi_const, rad, phi, z, dust_v1_0, dust_v2_0, dust_v3_0);

            Real &dust_den        = pdf->df_cons(rho_id, k, j, i);
            Real &dust_m1         = pdf->df_cons(v1_id,  k, j, i);
            Real &dust_m2         = pdf->df_cons(v2_id,  k, j, i);
            Real &dust_m3         = pdf->df_cons(v3_id,  k, j, i);
            Real inv_den_dust_ori = 1./dust_den;

            dust_den = dust_den*alpha_ori                           + dust_den_0*alpha_dam;
            dust_m1  = dust_den*(dust_m1*inv_den_dust_ori*alpha_ori + dust_v1_0*alpha_dam);
            dust_m2  = dust_den*(dust_m2*inv_den_dust_ori*alpha_ori + dust_v2_0*alpha_dam);
            //dust_m3  = dust_den*(dust_m3*inv_den_dust_ori*alpha_ori + dust_v3_0*alpha_dam);
          }
        }

      }
    }
  }
  return;
}


void LocalIsothermalEOS(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc, AthenaArray<Real> &cons) {
  // Local Isothermal equation of state
  DustFluids *pdf = pmb->pdustfluids;
  Real rad, phi, z, g_v1, g_v2, g_v3;
  int is = pmb->is; int ie = pmb->ie;
  int js = pmb->js; int je = pmb->je;
  int ks = pmb->ks; int ke = pmb->ke;

  Real inv_gamma = 1./gamma_gas;
  Real igm1      = 1.0/(gamma_gas - 1.0);

  for (int k=ks; k<=ke; ++k) { // include ghost zone
    for (int j=js; j<=je; ++j) { // prim, cons
      for (int i=is; i<=ie; ++i) {
        GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);

        Real &gas_den    = cons(IDN, k, j, i);
        Real &gas_m1     = cons(IM1, k, j, i);
        Real &gas_m2     = cons(IM2, k, j, i);
        Real &gas_m3     = cons(IM3, k, j, i);
        Real &gas_erg    = cons(IEN, k, j, i);

        // compute the initial density in cylindrical and spherical coordinates
        //gas_den = DenProfileCyl(rad, phi, z);

        //Real SN(0.0), QN(0.0), Psi(0.0);
        //for (int n=0; n<NDUSTFLUIDS; n++) {
          //int dust_id    = n;
          //int rho_id     = 4*dust_id;
          //Real &dust_den = pdf->df_cons(rho_id, k, j, i);
          //dust_den       = initial_D2G[dust_id] * gas_den;

          //SN += (dust_den*inv_gas_den)/(1.0 + SQR(Stokes_number[n]));
          //QN += (dust_den*inv_gas_den*Stokes_number[n])/(1.0 + SQR(Stokes_number[n]));
        //}
        //Psi = 1.0/((SN + beta_gas)*(SN + 2.0*ks_gas) + SQR(QN));

        //GasVelProfileCyl_NSH(SN,       QN,       Psi,       rad, phi, z, g_v1, g_v2, g_v3);
        ////GasVelProfileCyl_NSH(SN_const, QN_const, Psi_const, rad, phi, z, g_v1, g_v2, g_v3);

        //gas_m1 = gas_den * g_v1;
        //gas_m2 = gas_den * g_v2;
        //gas_m3 = gas_den * g_v3;

        Real inv_gas_den = 1.0/gas_den;
        Real press       = PoverR(rad, phi, z)*gas_den;
        gas_erg          = press*igm1 + 0.5*(SQR(gas_m1) + SQR(gas_m2) + SQR(gas_m3))*inv_gas_den;

        //for (int n = 0; n<NDUSTFLUIDS; n++) {
          //int dust_id = n;
          //int rho_id  = 4*dust_id;
          //int v1_id   = rho_id + 1;
          //int v2_id   = rho_id + 2;
          //int v3_id   = rho_id + 3;

          //Real &dust_den = pdf->df_cons(rho_id, k, j, i);
          //dust_den       = initial_D2G[dust_id] * gas_den;
        //}

      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k) {
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    rad = pco->x1v(i);
    phi = pco->x2v(j);
    z   = pco->x3v(k);
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    rad = std::abs(pco->x1v(i)*std::sin(pco->x2v(j)));
    phi = pco->x3v(i);
    z   = pco->x1v(i)*std::cos(pco->x2v(j));
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \f  computes density in cylindrical coordinates
Real RadialD2G(const Real rad, const Real initial_dust2gas, const Real slope)
{
  Real dust2gas = initial_dust2gas*std::pow(rad/r0, slope);
  return dust2gas;
}


Real DenProfileCyl(const Real rad, const Real phi, const Real z) {
  Real rho_mid  = rho_0*std::pow(rad/r0, dslope); // 2D
  return std::max(rho_mid, dfloor);
}


//! \f  computes pressure/density in cylindrical coordinates
Real PoverR(const Real rad, const Real phi, const Real z) {
  Real poverr;
  poverr = iso_cs2_r0*std::pow(rad/r0, pslope);
  return poverr;
}


//----------------------------------------------------------------------------------------
//! \f  computes rotational velocity in cylindrical coordinates
void GasVelProfileCyl(const Real rad, const Real phi, const Real z,
                   Real &v1, Real &v2, Real &v3) {
  Real iso_cs2 = PoverR(rad, phi, z);
  Real vel     = (dslope+pslope)*iso_cs2/(gm0/rad) + 1.0;
  vel          = std::sqrt(gm0/rad)*std::sqrt(vel);

  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    v1 = 0.0;
    v2 = vel;
    v3 = 0.0;
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    v1 = 0.0;
    v2 = 0.0;
    v3 = vel;
  }
  return;
}

void DustVelProfileCyl(const Real rad, const Real phi, const Real z,
                   Real &v1, Real &v2, Real &v3) {
  Real vel = std::sqrt(gm0/rad);

  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    v1 = 0.0;
    v2 = vel;
    v3 = 0.0;
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    v1 = 0.0;
    v2 = 0.0;
    v3 = vel;
  }
  return;
}

//! \f  computes rotational velocity in cylindrical coordinates
void GasVelProfileCyl_NSH(const Real SN, const Real QN, const Real Psi,
    const Real rad, const Real phi, const Real z, Real &v1, Real &v2, Real &v3) {
  //Real iso_cs2        = PoverR(rad, phi, z);
  //Real vel            = (dslope+pslope)*iso_cs2/(gm0/rad) + 1.0;
  //vel                 = std::sqrt(gm0/rad)*std::sqrt(vel);

  Real vel_Keplerian  = Keplerian_velocity(rad);
  Real vel            = beta_gas*vel_Keplerian;

  Real delta_gas_vr   = Delta_gas_vr(vel_Keplerian,   SN, QN, Psi);
  Real delta_gas_vphi = Delta_gas_vphi(vel_Keplerian, SN, QN, Psi);

  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    v1 = delta_gas_vr;
    v2 = vel + delta_gas_vphi;
    v3 = 0.0;
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    v1 = delta_gas_vr;
    v2 = 0.0;
    v3 = vel + delta_gas_vphi;
  }
  return;
}


void DustVelProfileCyl_NSH(const Real ts, const Real SN, const Real QN, const Real Psi,
    const Real rad, const Real phi, const Real z, Real &v1, Real &v2, Real &v3) {

  Real vel_Keplerian   = Keplerian_velocity(rad);
  Real delta_gas_vr    = Delta_gas_vr(vel_Keplerian,   SN, QN, Psi);
  Real delta_gas_vphi  = Delta_gas_vphi(vel_Keplerian, SN, QN, Psi);

  Real delta_dust_vr   = Delta_dust_vr(ts,   vel_Keplerian, delta_gas_vr, delta_gas_vphi);
  Real delta_dust_vphi = Delta_dust_vphi(ts, vel_Keplerian, delta_gas_vr, delta_gas_vphi);

  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    v1 = delta_dust_vr;
    v2 = vel_Keplerian+delta_dust_vphi;
    v3 = 0.0;
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    v1 = delta_dust_vr;
    v2 = 0.0;
    v3 = vel_Keplerian+delta_dust_vphi;
  }
  return;
}

Real Keplerian_velocity(const Real rad) {
  Real vk = std::sqrt(gm0/rad);
  return vk;
}


Real Delta_gas_vr(const Real vk, const Real SN, const Real QN, const Real Psi) {
  Real d_g_vr = -2.0*beta_gas*QN*Psi*(beta_gas - 1.0)*vk;
  return d_g_vr;
}


Real Delta_gas_vphi(const Real vk, const Real SN, const Real QN, const Real Psi) {
  Real d_g_vphi = -1.0*((SN + 2.0*ks_gas)*SN + SQR(QN))*Psi*(beta_gas - 1.0)*vk;
  return d_g_vphi;
}


Real Delta_dust_vr(const Real ts, const Real vk, const Real d_vgr, const Real d_vgphi) {
  Real d_d_vr = (2.0*ts*(beta_gas - 1.0)*vk)/(1.0+SQR(ts)) + ((d_vgr + 2.0*ts*d_vgphi)/(1.0+SQR(ts)));
  return d_d_vr;
}


Real Delta_dust_vphi(const Real ts, const Real vk, const Real d_vgr, const Real d_vgphi) {
  Real d_d_vphi = ((beta_gas - 1.0)*vk)/(1.0+SQR(ts)) + ((2.0*d_vgphi - ts*d_vgr)/(2.0+2.0*SQR(ts)));
  return d_d_vphi;
}


void Keplerian_interpolate(const Real r_active, const Real r_ghost,
    const Real vphi_active, Real &vphi_ghost) {
  vphi_ghost = vphi_active*std::sqrt(r_active/r_ghost);
  return;
}


void Density_interpolate(const Real r_active, const Real r_ghost, const Real rho_active,
    const Real slope, Real &rho_ghost) {
  rho_ghost = rho_active * std::pow(r_ghost/r_active, slope);
  return;
}


void Vr_interpolate_inner_nomatter(const Real r_active, const Real r_ghost, const Real sigma_active,
    const Real sigma_ghost, const Real vr_active, Real &vr_ghost) {
  //vr_ghost = vr_active <= 0.0 ? (sigma_active*r_active*vr_active)/(sigma_ghost*r_ghost) : 0.0;
  vr_ghost = (sigma_active*r_active*vr_active)/(sigma_ghost*r_ghost);
  return;
}


void Vr_interpolate_outer_nomatter(const Real r_active, const Real r_ghost, const Real sigma_active,
    const Real sigma_ghost, const Real vr_active, Real &vr_ghost) {
  //if (sigma_active < TINY_NUMBER)
    //vr_ghost = vr_active >= 0.0 ? ((sigma_active+TINY_NUMBER)*r_active*vr_active)/(sigma_ghost*r_ghost) : 0.0;
  //else
    //vr_ghost = vr_active >= 0.0 ? (sigma_active*r_active*vr_active)/(sigma_ghost*r_ghost) : 0.0;
  vr_ghost = (sigma_active*r_active*vr_active)/(sigma_ghost*r_ghost);
  return;
}

} // namespace


void InnerX1_NSH(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad_gh, phi_gh, z_gh, rad_ac, phi_ac, z_ac;
  Real v1_gh,  v2_gh,  v3_gh, df_v1_gh, df_v2_gh, df_v3_gh;
  Real v1_ac,  v2_ac,  v3_ac, df_v1_ac, df_v2_ac, df_v3_ac;

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        GetCylCoord(pco, rad_gh, phi_gh, z_gh, il-i, j, k);

        Real &gas_rho_gh    = prim(IDN, k, j, il-i);
        Real &gas_v1_gh     = prim(IM1, k, j, il-i);
        Real &gas_v2_gh     = prim(IM2, k, j, il-i);
        Real &gas_v3_gh     = prim(IM3, k, j, il-i);
        gas_rho_gh          = DenProfileCyl(rad_gh, phi_gh, z_gh);
        Real inv_gas_rho_gh = 1.0/gas_rho_gh;

        Real SN(0.0), QN(0.0), Psi(0.0);
        for (int n=0; n<NDUSTFLUIDS; n++) {
          int dust_id       = n;
          int rho_id        = 4*dust_id;
          Real &dust_rho_gh = pmb->pdustfluids->df_prim(rho_id, k, j, il-i);
          dust_rho_gh       = initial_D2G[dust_id]*gas_rho_gh;

          SN += (dust_rho_gh*inv_gas_rho_gh)/(1.0 + SQR(Stokes_number[n]));
          QN += (dust_rho_gh*inv_gas_rho_gh*Stokes_number[n])/(1.0 + SQR(Stokes_number[n]));
        }
        Psi = 1.0/((SN + beta_gas)*(SN + 2.0*ks_gas) + SQR(QN));

        //GasVelProfileCyl_NSH(SN, QN, Psi, rad_gh, phi_gh, z_gh, v1_gh, v2_gh, v3_gh);
        GasVelProfileCyl_NSH(SN_const, QN_const, Psi_const, rad_gh, phi_gh, z_gh, v1_gh, v2_gh, v3_gh);
        gas_v1_gh = v1_gh;
        gas_v2_gh = v2_gh;
        gas_v3_gh = v3_gh;

        if (NON_BAROTROPIC_EOS) {
          Real &gas_pre_gh = prim(IPR, k, j, il-i);
          gas_pre_gh       = PoverR(rad_gh, phi_gh, z_gh)*gas_rho_gh;
        }

        if (NDUSTFLUIDS > 0) {
          for (int n = 0; n<NDUSTFLUIDS; n++){
						int dust_id = n;
						int rho_id  = 4*dust_id;
						int v1_id   = rho_id + 1;
						int v2_id   = rho_id + 2;
						int v3_id   = rho_id + 3;

						Real &dust_rho_gh = pmb->pdustfluids->df_prim(rho_id, k, j, il-i);
						Real &dust_v1_gh  = pmb->pdustfluids->df_prim(v1_id,  k, j, il-i);
						Real &dust_v2_gh  = pmb->pdustfluids->df_prim(v2_id,  k, j, il-i);
						Real &dust_v3_gh  = pmb->pdustfluids->df_prim(v3_id,  k, j, il-i);

            //DustVelProfileCyl_NSH(Stokes_number[dust_id], SN, QN, Psi, rad_gh, phi_gh, z_gh,
                //df_v1_gh, df_v2_gh, df_v3_gh);
            DustVelProfileCyl_NSH(Stokes_number[dust_id], SN_const, QN_const, Psi_const, rad_gh, phi_gh, z_gh,
                df_v1_gh, df_v2_gh, df_v3_gh);

						dust_v1_gh = df_v1_gh;
						dust_v2_gh = df_v2_gh;
						dust_v3_gh = df_v3_gh;
					}
        }

      }
    }
  }
}


void OuterX1_NSH(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad_gh, phi_gh, z_gh,  rad_ac,   phi_ac,   z_ac;
  Real v1_gh,  v2_gh,  v3_gh, df_v1_gh, df_v2_gh, df_v3_gh;
  Real v1_ac,  v2_ac,  v3_ac, df_v1_ac, df_v2_ac, df_v3_ac;

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        GetCylCoord(pco, rad_gh, phi_gh, z_gh, iu+i, j, k);

        Real &gas_rho_gh    = prim(IDN, k, j, iu+i);
        Real &gas_v1_gh     = prim(IM1, k, j, iu+i);
        Real &gas_v2_gh     = prim(IM2, k, j, iu+i);
        Real &gas_v3_gh     = prim(IM3, k, j, iu+i);
        gas_rho_gh          = DenProfileCyl(rad_gh, phi_gh, z_gh);
        Real inv_gas_rho_gh = 1.0/gas_rho_gh;

        Real SN(0.0), QN(0.0), Psi(0.0);
        for (int n=0; n<NDUSTFLUIDS; n++) {
          int dust_id       = n;
          int rho_id        = 4*dust_id;
          Real &dust_rho_gh = pmb->pdustfluids->df_prim(rho_id, k, j, iu+i);
          dust_rho_gh       = initial_D2G[dust_id]*gas_rho_gh;

          SN += (dust_rho_gh*inv_gas_rho_gh)/(1.0 + SQR(Stokes_number[n]));
          QN += (dust_rho_gh*inv_gas_rho_gh*Stokes_number[n])/(1.0 + SQR(Stokes_number[n]));
        }
        Psi = 1.0/((SN + beta_gas)*(SN + 2.0*ks_gas) + SQR(QN));

        //GasVelProfileCyl_NSH(SN, QN, Psi, rad_gh, phi_gh, z_gh, v1_gh, v2_gh, v3_gh);
        GasVelProfileCyl_NSH(SN_const, QN_const, Psi_const, rad_gh, phi_gh, z_gh, v1_gh, v2_gh, v3_gh);
        gas_v1_gh = v1_gh;
        gas_v2_gh = v2_gh;
        gas_v3_gh = v3_gh;

        if (NON_BAROTROPIC_EOS) {
          Real &gas_pre_gh = prim(IPR, k, j, iu+i);
          gas_pre_gh       = PoverR(rad_gh, phi_gh, z_gh)*gas_rho_gh;
        }

        if (NDUSTFLUIDS > 0) {
          for (int n = 0; n<NDUSTFLUIDS; n++){
						int dust_id = n;
						int rho_id  = 4*dust_id;
						int v1_id   = rho_id + 1;
						int v2_id   = rho_id + 2;
						int v3_id   = rho_id + 3;

						Real &dust_rho_gh = pmb->pdustfluids->df_prim(rho_id, k, j, iu+i);
						Real &dust_v1_gh  = pmb->pdustfluids->df_prim(v1_id,  k, j, iu+i);
						Real &dust_v2_gh  = pmb->pdustfluids->df_prim(v2_id,  k, j, iu+i);
						Real &dust_v3_gh  = pmb->pdustfluids->df_prim(v3_id,  k, j, iu+i);

            //DustVelProfileCyl_NSH(Stokes_number[dust_id], SN, QN, Psi, rad_gh, phi_gh, z_gh,
                //df_v1_gh, df_v2_gh, df_v3_gh);
            DustVelProfileCyl_NSH(Stokes_number[dust_id], SN_const, QN_const, Psi_const, rad_gh, phi_gh, z_gh,
                df_v1_gh, df_v2_gh, df_v3_gh);

						dust_v1_gh = df_v1_gh;
						dust_v2_gh = df_v2_gh;
						dust_v3_gh = df_v3_gh;
					}
        }

      }
    }
  }
}
