#ifndef INTEG_H
#define INTEG_H

// This is the low-level integral API ... you should probably
// be using the API in scfdata.h

void setfm();

double g(const BasisFunction& bfi, const BasisFunction& bfj, 
         const BasisFunction& bfk, const BasisFunction& bfl);


double h(const BasisFunction& bfi, const BasisFunction& bfj,
         int natom, const Atom* atoms);

double s(const BasisFunction& bfi, const BasisFunction& bfj);

#endif
