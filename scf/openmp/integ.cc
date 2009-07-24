#include <iostream>
#include <cmath>
#include <cstdio>

using namespace std;

#include "scfdata.h"
#include "integ.h"


// Stuff for computing F0, etc
static const double delta = 28.0/2000.0; // spacing of interpolation grid
static const double delo2 = delta*0.5;
static const double rdelta = 1.0/delta;
static double fm[2001*5]; // fm[i,m] = d^m f0 / dx^m
static const double PI = 3.1415926535897932385;
    
// initalize computation of f0 by recursion down from f200
void setfm() {
    for (int ii=0; ii<2001; ii++) {
        double t = delta*ii;
        double et = exp(-t);
        t *= 2.0;
        
        double val=0.0;
        for (int i=200; i>=4; i--) {
            double rr = 1.0/(2.0*i+1.0);
            val=(et + t*val)*rr;
        }
        fm[ii*5+4] = val;
        
        for (int i=3; i>=0; i--) {
            double rr=1.0/(2.0*i+1.0);
            fm[ii*5 + i] = (et + t*fm[ii*5 + i + 1])*rr;
        }
        
        // Incorporate factors scaling fm to d^mf0/dx^m
        for (int i=1; i<5; i++) {
            fm[ii*5 + i] /= i;
        }
    }
}

// Computes f0
static double f0(double t) {
    const double fac0=0.8862269254527580; // 0.5*sqrt(pi) 
    const double t0 = 28.0;
    
    // computes f0 to a relative accuracy of better than 4.e-13 for
    // all t.  uses 4th order taylor expansion on grid out to
    // t=28.0 asymptotic expansion accurate for t greater than 28
    
    if (t >= t0) {
        return fac0 / sqrt(t);
    }
    else {
        int n = (t+delo2)*rdelta;
        double x = delta*n - t;
        n *= 5;
        return fm[n] + x*(fm[n+1] + x*(fm[n+2] + x*(fm[n+3] + x*fm[n+4])));
    }
}

// compute the two-electon integral (ij|kl) over primitive 1s gaussians
double g(const BasisFunction& bfi, const BasisFunction& bfj, 
         const BasisFunction& bfk, const BasisFunction& bfl) 
{
    const double twopi25 = 34.986836655249725694; // 2 * pi^2.5

    const double xij = bfi.x - bfj.x;
    const double yij = bfi.y - bfj.y;
    const double zij = bfi.z - bfj.z;
    const double rij2 = xij*xij + yij*yij + zij*zij;
    const double expntij = bfi.expnt*bfj.expnt/(bfi.expnt+bfj.expnt);
    const double argij = rij2*expntij;
    if (argij > 46.0) return 0.0;

    const double xkl = bfk.x - bfl.x;
    const double ykl = bfk.y - bfl.y;
    const double zkl = bfk.z - bfl.z;
    const double rkl2 = xkl*xkl + ykl*ykl + zkl*zkl;
    const double expntkl = bfk.expnt*bfl.expnt/(bfk.expnt+bfl.expnt);
    const double argkl = rkl2*expntkl;

    const double argijkl = argij + argkl;
    if (argijkl > 46.0) return 0.0;

    const double exijkl = exp(-argijkl);
    const double denom = (bfi.expnt+bfj.expnt)*(bfk.expnt+bfl.expnt)*sqrt(bfi.expnt+bfj.expnt+bfk.expnt+bfl.expnt);
    const double fac = (bfi.expnt+bfj.expnt)*(bfk.expnt+bfl.expnt) / (bfi.expnt+bfj.expnt+bfk.expnt+bfl.expnt);

    const double xp = (bfi.x*bfi.expnt + bfj.x*bfj.expnt)/(bfi.expnt+bfj.expnt);
    const double yp = (bfi.y*bfi.expnt + bfj.y*bfj.expnt)/(bfi.expnt+bfj.expnt);
    const double zp = (bfi.z*bfi.expnt + bfj.z*bfj.expnt)/(bfi.expnt+bfj.expnt);
    const double xq = (bfk.x*bfk.expnt + bfl.x*bfl.expnt)/(bfk.expnt+bfl.expnt);
    const double yq = (bfk.y*bfk.expnt + bfl.y*bfl.expnt)/(bfk.expnt+bfl.expnt);
    const double zq = (bfk.z*bfk.expnt + bfl.z*bfl.expnt)/(bfk.expnt+bfl.expnt);

    const double xpq = xp - xq;
    const double ypq = yp - yq;
    const double zpq = zp - zq;
    const double rpq2 = xpq*xpq + ypq*ypq + zpq*zpq;

    const double f0val = f0(fac*rpq2);

    return (twopi25 / denom) * exijkl * f0val * bfi.coeff * bfj.coeff * bfk.coeff * bfl.coeff;
}


// Compute the one-particle Hamiltonian matrix element over primitive 1s gaussians
double h(const BasisFunction& bfi, const BasisFunction& bfj,
         int natom, const Atom* atoms)
{
    const double xij = bfi.x - bfj.x;
    const double yij = bfi.y - bfj.y;
    const double zij = bfi.z - bfj.z;
    const double rij2 = xij*xij + yij*yij + zij*zij;
    const double reipej = 1.0/(bfi.expnt+bfj.expnt);
    const double expntij = bfi.expnt*bfj.expnt*reipej;
    const double argij = rij2*expntij;
    if (argij > 46.0) return 0.0;

    const double expij = exp(-argij);

    const double twopi = 2.0*PI;
    const double repij = twopi * expij * reipej;

    const double xp = (bfi.x*bfi.expnt + bfj.x*bfj.expnt)*reipej;
    const double yp = (bfi.y*bfi.expnt + bfj.y*bfj.expnt)*reipej;
    const double zp = (bfi.z*bfi.expnt + bfj.z*bfj.expnt)*reipej;
    
    // first do the nuclear attraction integrals
    double sum = 0.0;
    for (int iat=0; iat<natom; iat++) {
        const double xpa = xp - atoms[iat].x;
        const double ypa = yp - atoms[iat].y;
        const double zpa = zp - atoms[iat].z;
        const double rpa2 = xpa*xpa + ypa*ypa + zpa*zpa;
 
        sum += atoms[iat].q * f0((bfi.expnt+bfj.expnt)*rpa2);
    }
    sum = - repij*sum;

    // add on the kinetic energy term
    sum += expntij*(3.0-2.0*expntij*rij2) * pow(PI*reipej,1.5) * expij;

    // multiply by the normalization constants
    return sum * bfi.coeff * bfj.coeff;
}

// Compute the overlap matrix element over primitive 1s gaussians
double s(const BasisFunction& bfi, const BasisFunction& bfj) 
{
    const double xij = bfi.x - bfj.x;
    const double yij = bfi.y - bfj.y;
    const double zij = bfi.z - bfj.z;
    const double rij2 = xij*xij + yij*yij + zij*zij;
    const double reipej = 1.0/(bfi.expnt+bfj.expnt);
    const double expntij = bfi.expnt*bfj.expnt*reipej;
    const double argij = rij2*expntij;
    if (argij > 46.0) return 0.0;

    return pow(PI*reipej,1.5) * exp(-argij) * bfi.coeff * bfj.coeff;
}

