#include <iostream>
#include <cmath>
#include <cstdio>

using namespace std;


// Stuff for computing F0 (was class member data)
static const double delta = 28.0/2000.0;
static const double delo2 = delta*0.5;
static const double rdelta = 1.0/delta;
static double fm[2001*5];
    
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
double f0(double t) {
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
double g(double xi, double yi, double zi, double expnti, double coeffi,
         double xj, double yj, double zj, double expntj, double coeffj,
         double xk, double yk, double zk, double expntk, double coeffk,
         double xl, double yl, double zl, double expntl, double coeffl)
{
    const double twopi25 = 34.986836655249725694; // 2 * pi^2.5

    const double xij = xi - xj;
    const double yij = yi - yj;
    const double zij = zi - zj;
    const double rij2 = xij*xij + yij*yij + zij*zij;
    const double expntij = expnti*expntj/(expnti+expntj);
    const double argij = rij2*expntij;
    if (argij > 46.0) return 0.0;

    const double xkl = xk - xl;
    const double ykl = yk - yl;
    const double zkl = zk - zl;
    const double rkl2 = xkl*xkl + ykl*ykl + zkl*zkl;
    const double expntkl = expntk*expntl/(expntk+expntl);
    const double argkl = rkl2*expntkl;

    const double argijkl = argij + argkl;
    if (argijkl > 46.0) return 0.0;

    const double exijkl = exp(-argijkl);
    const double denom = (expnti+expntj)*(expntk+expntl)*sqrt(expnti+expntj+expntk+expntl);
    const double fac = (expnti+expntj)*(expntk+expntl) / (expnti+expntj+expntk+expntl);

    const double xp = (xi*expnti + xj*expntj)/(expnti+expntj);
    const double yp = (yi*expnti + yj*expntj)/(expnti+expntj);
    const double zp = (zi*expnti + zj*expntj)/(expnti+expntj);
    const double xq = (xk*expntk + xl*expntl)/(expntk+expntl);
    const double yq = (yk*expntk + yl*expntl)/(expntk+expntl);
    const double zq = (zk*expntk + zl*expntl)/(expntk+expntl);

    const double xpq = xp - xq;
    const double ypq = yp - yq;
    const double zpq = zp - zq;
    const double rpq2 = xpq*xpq + ypq*ypq + zpq*zpq;

    const double f0val = f0(fac*rpq2);

    return (twopi25 / denom) * exijkl * f0val * coeffi * coeffj * coeffk * coeffl;
}


// Compute the one-particle Hamiltonian matrix element over primitive 1s gaussians
double h(double xi, double yi, double zi, double expnti, double coeffi,
         double xj, double yj, double zj, double expntj, double coeffj,
         int natom, const double* coords, const double* charges)
{
    const double xij = xi - xj;
    const double yij = yi - yj;
    const double zij = zi - zj;
    const double rij2 = xij*xij + yij*yij + zij*zij;
    const double eipej = expnti+expntj;
    const double expntij = expnti*expntj/eipj;
    const double argij = rij2*expntij;
    if (argij > 46.0) return 0.0;

    const double expij = exprjh(-argij);

    const double twopi = 2.0*3.1415926535897932385;
    const double repij = twopi * expij / eipej;

    const double xp = (xi*expnti + xj*expntj)/(expnti+expntj);
    const double yp = (yi*expnti + yj*expntj)/(expnti+expntj);
    const double zp = (zi*expnti + zj*expntj)/(expnti+expntj);
    
    // first do the nuclear attraction integrals
    double sum = 0.0;
    for (int iat=0; iat<natom; iat++) {
        const double xpa = xp - coords[iat*3  ];
        const double ypa = yp - coords[iat*3+1];
        const double zpa = zp - coords[iat*3+2];
        const double rpa2 = xpa*xpa + ypa*ypa + zpa*zpa;
 
        sum += charges(iat) * f0(eipej*rpa2);
    }
    sum = - repij*sum;

    // add on the kinetic energy term
    sum += expntij*(3.0d0-2.0d0*expntij*rij2) * (pi/(expnti+expntj))**1.5d0 * expij;

    // finally multiply by the normalization constants
    return sum * coeffi * coeffj;
}

// Compute the overlap matrix element over primitive 1s gaussians
double s(double xi, double yi, double zi, double expnti, double coeffi,
         double xj, double yj, double zj, double expntj, double coeffj) {
    const double xij = xi - xj;
    const double yij = yi - yj;
    const double zij = zi - zj;
    const double rij2 = xij*xij + yij*yij + zij*zij;
    const double reipej = 1.0/(expnti+expntj);
    const double expntij = expnti*expntj*eipej;
    const double argij = rij2*expntij;
    if (argij > 46.0) return 0.0;

    return (pi*reipej)**1.5d0 * exprjh(-argij) * coeffi * coeffj;
}

int main() {
    cout.precision(12);
    setfm();
    cout << f0(3.14159) << endl;


    cout << g(
              1.1,1.2,1.3,1.4,1.5,
              1.6,1.7,1.8,1.9,2.0,
              2.1,2.2,2.3,2.4,2.5,
              2.6,2.7,2.8,2.9,3.0) << endl;

    return 0;
}
