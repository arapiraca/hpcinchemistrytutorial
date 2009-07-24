#ifndef SCF_DATA_H
#define SCF_DATA_H

#include <iostream>

static const int MAX_NATOM = 100;
static const int MAX_NBF_PER_ATOM = 30;
static const int MAX_NBF = MAX_NATOM * MAX_NBF_PER_ATOM;

// Using char[8] rather than std::string so that structures are POD and
// can therefore be easily communicated via MPI.


/// Attributes of an atom
struct Atom {
    double x, y, z, q;          // coordinates and charge
    int lo, hi;                 // inclusive range of basis functions on atom
    char tag[8];                // tag used to associate with basis functions
};


/// A single atomic basis function (s primitive)
struct BasisFunction {
    double x, y, z, coeff, expnt;
};


/// Basis functions associated with atom of type tag
struct AtomicBasis {
    char types[MAX_NBF_PER_ATOM][8];
    double expnts[MAX_NBF_PER_ATOM];
    char tag[8];
    int n;
};
    

/// Read the user input file
void read_input(); 

/// Compute two-electron integral (ij|kl)
double g(int i, int j, int k, int l);


/// Compute one-electron Hamiltonian <i|h|j>
double h(int i, int j);


/// Compute overlap integral <i|j>
double s(int i, int j);


/// Returns the number of atoms
int num_atom();


/// Returns the number of basis functions
int num_basis_functions();


/// Returns the number of electrons
int num_electrons(); 

/// Returns const reference to i'th atom, i=0,..,natom-1
const Atom& get_atom(int i);


/// Returns const pointer to array of atoms
const Atom* get_atoms(); 


/// Returns const reference to i'th basis function, i=0,...,nbf-1
const BasisFunction& get_bfn(int i);

/// Returns the nuclear repulsion energy
double nuclear_repulsion_energy();


/// Returns the wall time since the first call to the routine
double wall_time();

/// Returns no. of iterations to damp
int get_ndamp();

/// Returns damping factor
double get_damp();

#endif
