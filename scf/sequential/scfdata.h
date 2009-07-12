#ifndef SCF_DATA_H
#define SCF_DATA_H

#include <iostream>

static const int MAX_NATOM = 100;
static const int MAX_NBF_PER_ATOM = 30;
static const int MAX_NBF = MAX_NATOM * MAX_NBF_PER_ATOM;

// Using char[8] rather than std::string so that structures are POD and
// can therefore be easily communicated via MPI.

struct Atom {
    double x, y, z, q;          // coordinates and charge
    int lo, hi;                 // inclusive range of basis functions on atom a
    char tag[8];
};

struct BasisFunction {          // A single atomic basis function (s primitive)
    double x, y, z, coeff, expnt;
};

struct AtomicBasis {            // Basis functions associated with atom of type tag
    char types[MAX_NBF_PER_ATOM][8];
    double expnts[MAX_NBF_PER_ATOM];
    char tag[8];
    int n;
};
    
void read_input();              // Read the input

double g(int i, int j, int k, int l);  // Compute two-electron integerl (ij|kl)

double h(int i, int j);         // Compute one-electron Hamiltonian <i|h|j>

double s(int i, int j);         // Compute overlap integral <i|j>

int num_atom();                 // Returns the number of atoms

int num_basis_functions();      // Returns the number of basis functions

int num_electrons();            // Returns the number of electrons

const Atom& get_atom(int i);    // Returns const reference to i'th atom, i=0,..,natom-1

const Atom* get_atoms();        // Returns const pointer to array of atoms

const BasisFunction& get_bfn(int i); // Returns const reference to i'th basis function, i=0,...,nbf-1

#endif
