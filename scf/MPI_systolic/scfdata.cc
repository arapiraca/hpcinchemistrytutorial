#include <iostream>
#include <string>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sys/time.h>

#include <mpi.h>

using namespace std;

#include "scfdata.h"
#include "integ.h"

static int natom = 0;                // No. of atoms
static int nbf = 0;                  // No. of basis functions
static int nbf_tag = 0;              // No. of unique basis centers/atoms
static int nelec = 0;                // No. of electrons
static int ndamp = 5;                // No. of iterations to apply damping
static double damp = 0.3;            // Damping factor

static Atom atoms[MAX_NATOM];        // Atoms in the calculation
static AtomicBasis bfset[MAX_NATOM]; // Sets of atom-centered basis functions
static BasisFunction bfns[MAX_NBF];  // Actual basis functions and coordinates

static void read_geometry() {
    /*
      atomic units

      geometry 
        tag q x y z
        ...
      end

      tag is used to associate atoms with basis sets
    */
    
    cout << "Geometry (a.u.)\n--------\n\n";
    natom = 0;
    while (1) {
        string tag;
        cin >> tag;
        if (tag == "end") break;

        if (tag.size()>7) throw "Atom tags must be 7 chars or less";
        strcpy(atoms[natom].tag,tag.c_str());

        cin >> atoms[natom].q >> atoms[natom].x >> atoms[natom].y >> atoms[natom].z;
        printf("%3d  %12s  %6.2f     %12.6f  %12.6f  %12.6f\n",
               natom, tag.c_str(), atoms[natom].q,
               atoms[natom].x, atoms[natom].y, atoms[natom].z);
        natom++;
    }
    cout << "\n";
}
        
static void read_basis_set() {
    /*
      basis
         tag
            s   expnt
            spd expnt
         ...
      end
    */

    cout << "Basis set\n---------\n\n";
    nbf_tag = 0;
    string tag;
    cin >> tag;
    while (1) {
        if (tag == "end") break;

        if (tag.size()>7) throw "Atom tags must be 7 chars or less";
        strcpy(bfset[nbf_tag].tag, tag.c_str());

        printf("    %s\n", tag.c_str());

        int n = 0;
        string bftag;
        while (cin >> bftag) {
            if (bftag == "s" || bftag == "sp" || bftag == "spd") {
                if (n == MAX_NBF_PER_ATOM) throw "too many basis functions on atom";
                double expnt;
                cin >> expnt;

                printf("        %2d  %4s  %12.6f\n", n, bftag.c_str(), expnt);

                if (bftag.size()>7) throw "BFN tags must be 7 chars or less";
                strcpy(bfset[nbf_tag].types[n], bftag.c_str());
                bfset[nbf_tag].expnts[n] = expnt;
                n++;
                bfset[nbf_tag].n = n;
            }
            else {
                nbf_tag++;
                tag = bftag;
                printf("\n");
                break;
            }
        }
    }
}

// Add a single primitive s function to the basis set
static void add_bfn(double x, double y, double z, double expnt) {
    if (nbf > MAX_NBF)
        throw "Too many basis functions";
    const double PI = 3.1415926535897932385;
    bfns[nbf].x = x;
    bfns[nbf].y = y;
    bfns[nbf].z = z;
    bfns[nbf].expnt = expnt;
    bfns[nbf].coeff = pow(expnt*2.0/PI,0.75);
    nbf++;
}

// After reading the basis set and the geometry must join the data structures
static void build_full_basis() {
    printf("\n Atom basis function ranges\n");
    nbf = 0;
    for (int a=0; a<natom; a++) {
        // Find the matching basis set for this atom
        bool found = false;
        for (int b=0; b<nbf_tag; b++) {
            if (!strcmp(bfset[b].tag,atoms[a].tag)) {

                found = true;

                atoms[a].lo = nbf;
                for (int i=0; i<bfset[b].n; i++) {
                    if (!strcmp(bfset[b].types[i],"s")) {
                        add_bfn(atoms[a].x, atoms[a].y, atoms[a].z, bfset[b].expnts[i]);
                    }
                    else if (!strcmp(bfset[b].types[i],"sp")) {
                        double s = 0.25/sqrt(bfset[b].expnts[i]);
                        add_bfn(atoms[a].x  , atoms[a].y  , atoms[a].z  , bfset[b].expnts[i]);
                        add_bfn(atoms[a].x+s, atoms[a].y  , atoms[a].z  , bfset[b].expnts[i]);
                        add_bfn(atoms[a].x-s, atoms[a].y  , atoms[a].z  , bfset[b].expnts[i]);
                        add_bfn(atoms[a].x  , atoms[a].y+s, atoms[a].z  , bfset[b].expnts[i]);
                        add_bfn(atoms[a].x  , atoms[a].y-s, atoms[a].z  , bfset[b].expnts[i]);
                        add_bfn(atoms[a].x  , atoms[a].y  , atoms[a].z+s, bfset[b].expnts[i]);
                        add_bfn(atoms[a].x  , atoms[a].y  , atoms[a].z-s, bfset[b].expnts[i]);
                    }
                    else if (!strcmp(bfset[b].types[i],"spd")) {
                        double s = 0.25/sqrt(bfset[b].expnts[i]);
                        add_bfn(atoms[a].x  , atoms[a].y  , atoms[a].z  , bfset[b].expnts[i]);
                        add_bfn(atoms[a].x+s, atoms[a].y+s, atoms[a].z+s, bfset[b].expnts[i]);
                        add_bfn(atoms[a].x-s, atoms[a].y+s, atoms[a].z+s, bfset[b].expnts[i]);
                        add_bfn(atoms[a].x+s, atoms[a].y-s, atoms[a].z+s, bfset[b].expnts[i]);
                        add_bfn(atoms[a].x-s, atoms[a].y-s, atoms[a].z+s, bfset[b].expnts[i]);
                        add_bfn(atoms[a].x+s, atoms[a].y+s, atoms[a].z-s, bfset[b].expnts[i]);
                        add_bfn(atoms[a].x-s, atoms[a].y+s, atoms[a].z-s, bfset[b].expnts[i]);
                        add_bfn(atoms[a].x+s, atoms[a].y-s, atoms[a].z-s, bfset[b].expnts[i]);
                        add_bfn(atoms[a].x-s, atoms[a].y-s, atoms[a].z-s, bfset[b].expnts[i]);
                    }
                    else {
                        throw "bad basis function type building full basis?";
                    }
                }
                atoms[a].hi = nbf-1;
                printf("  %3d   %4d : %4d\n", a, atoms[a].lo, atoms[a].hi);
                break;
            }
        }
        if (!found) 
            printf("No basis functions on center %d?\n", a);
    }
    printf("\nTotal number of basis function %d\n", nbf);

    // Print the basis
    //for (int i=0; i<nbf; i++) {
    //    printf("%4d   %12.6f %12.6f     %12.6f %12.6f %12.6f\n", i, bfns[i].expnt, bfns[i].coeff, bfns[i].x, bfns[i].y, bfns[i].z);
    //}
}

static void read_scf() {
    nelec = 0;

    string tag;
    while (cin >> tag) {
        if (tag == "end") {
            break;
        }
        else if (tag == "nelec") {
            cin >> nelec;
        }
        else if (tag == "damp") {
            cin >> damp;
        }
        else if (tag == "ndamp") {
            cin >> ndamp;
        }
        else {
            throw "unkown directive reading scf input";
        }
    }
}

void broadcast_data() {
    MPI::COMM_WORLD.Bcast(&natom,  sizeof(natom),  MPI::BYTE, 0);
    MPI::COMM_WORLD.Bcast(&nbf,    sizeof(nbf),    MPI::BYTE, 0);
    MPI::COMM_WORLD.Bcast(&nbf_tag,sizeof(nbf_tag),MPI::BYTE, 0);
    MPI::COMM_WORLD.Bcast(&nelec,  sizeof(nelec),  MPI::BYTE, 0);
    MPI::COMM_WORLD.Bcast(&ndamp,  sizeof(ndamp),  MPI::BYTE, 0);
    MPI::COMM_WORLD.Bcast(&damp,   sizeof(damp),   MPI::BYTE, 0);

    cout << sizeof(atoms) << endl;
    MPI::COMM_WORLD.Bcast( atoms,  sizeof(atoms),  MPI::BYTE, 0);
    MPI::COMM_WORLD.Bcast( bfset,  sizeof(bfset),  MPI::BYTE, 0);
    MPI::COMM_WORLD.Bcast( bfns,   sizeof(bfns),   MPI::BYTE, 0);
}

void read_input() {

    if (MPI::COMM_WORLD.Get_rank() == 0) {
        string tag;
        while (cin >> tag) {
            if (tag == "geometry") {
                read_geometry();
            }
            else if (tag == "basis") {
                read_basis_set();
            }
            else if (tag == "scf") {
                read_scf();
            }
            else {
                throw "unknown directive";
            }
        }
        
        build_full_basis();
        
        // If nelec was not specified, default is a neutral molecule
        if (nelec == 0) {
            for (int i=0; i<natom; i++) {
                nelec += atoms[i].q;
            }
        }

        printf("\nNumber of electrons %d\n\n", nelec);
    }

    broadcast_data();
}

double g(int i, int j, int k, int l) {
    return g(get_bfn(i), get_bfn(j), get_bfn(k), get_bfn(l));
}

double h(int i, int j) {
    return h(get_bfn(i), get_bfn(j), natom, atoms);
}

double s(int i, int j) {
    return s(get_bfn(i), get_bfn(j));
}

int num_atoms() {
    return natom;
}

int num_basis_functions() {
    return nbf;
}

int num_electrons() {
    return nelec;
}

const Atom& get_atom(int i) {
    if (i<0 || i>=natom) throw "Bad i in get_atom";
    return atoms[i];
}

const Atom* get_atoms() {
    return atoms;
}

const BasisFunction& get_bfn(int i) {
    if (i<0 || i>=nbf) throw "Bad i in get_bfn";
    return bfns[i];
}

double nuclear_repulsion_energy() {
    double sum = 0.0;
    for (int i=0; i<natom; i++) {
        for (int j=0; j<i; j++) {
            double xx = atoms[i].x - atoms[j].x;
            double yy = atoms[i].y - atoms[j].y;
            double zz = atoms[i].z - atoms[j].z;
            sum += atoms[i].q*atoms[j].q / sqrt(xx*xx + yy*yy + zz*zz);
        }
    }
    return sum;
}

double wall_time() {
    static bool first_call = true;
    static double start_time;
    
    struct timeval tv;
    gettimeofday(&tv,0);
    double now = tv.tv_sec + 1e-6*tv.tv_usec;
    
    if (first_call) {
        first_call = false;
        start_time = now;
    }
    return now - start_time;
}

int get_ndamp() {
    return ndamp;
}

double get_damp() {
    return damp;
}

