#include <iostream>
#include <string>
#include <cmath>
#include <cstdio>

using namespace std;

static const int MAX_NATOM = 100;
static const int MAX_NBF_PER_ATOM = 30;
static const int MAX_NBF = MAX_NATOM * MAX_NBF_PER_ATOM;

static int natom=0;             // No. of atoms
static int nbf=0;               // No. of basis functions
static int nbf_tag=0;           // No. of unique basis centers/atoms

struct Atom {
    double x, y, z, q;          // coordinates and charge
    int lo, hi;                 // inclusive range of basis functions on atom a
    string tag;
};

struct AtomicBasis {            // Basis functions associated with atom of type tag
    string types[MAX_NBF_PER_ATOM];
    double expnts[MAX_NBF_PER_ATOM];
    string tag;
    int n;
};
    

struct BasisFunction {    // a single atomic basis function (s primitive)
    double x, y, z, coeff, expnt;
};


static Atom atoms[MAX_NATOM];
static AtomicBasis bfset[MAX_NATOM];
static BasisFunction bfns[MAX_NBF];

void read_geometry() {
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

        atoms[natom].tag = tag;
        cin >> atoms[natom].q >> atoms[natom].x >> atoms[natom].y >> atoms[natom].z;
        printf("%3d  %12s  %6.2f     %12.6f  %12.6f  %12.6f\n",
               natom, tag.c_str(), atoms[natom].q,
               atoms[natom].x, atoms[natom].y, atoms[natom].z);
        natom++;
    }
    cout << "\n";
}
        
void read_basis_set() {
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
        bfset[nbf_tag].tag = tag;

        printf("    %s\n", tag.c_str());

        int n = 0;
        string bftag;
        while (cin >> bftag) {
            if (bftag == "s" || bftag == "sp" || bftag == "spd") {
                if (n == MAX_NBF_PER_ATOM) throw "too many basis functions on atom";
                double expnt;
                cin >> expnt;

                printf("        %2d  %4s  %12.6f\n", n, bftag.c_str(), expnt);

                bfset[nbf_tag].types[n] = bftag;
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
void add_bfn(double x, double y, double z, double expnt) {
    if (nbf > MAX_NBF)
        throw "Too many basis functions";
    const double PI = 3.1415926535897932385;
    bfns[nbf].x = x;
    bfns[nbf].y = y;
    bfns[nbf].z = z;
    bfns[nbf].expnt = expnt;
    bfns[nbf].coeff = pow(expnt*2.0*PI,0.75);
    nbf++;
}

// After reading the basis set and the geometry must join the data structures
void build_full_basis() {
    
    printf("\n Atom basis function ranges\n");
    nbf = 0;
    for (int a=0; a<natom; a++) {
        // Find the matching basis set for this atom
        bool found = false;
        for (int b=0; b<nbf_tag; b++) {
            if (bfset[b].tag == atoms[a].tag) {

                printf("bfset n %d\n", bfset[b].n);

                found = true;

                atoms[a].lo = nbf;
                for (int i=0; i<bfset[b].n; i++) {
                    if (bfset[b].types[i] == "s") {
                        add_bfn(atoms[a].x, atoms[a].y, atoms[a].z, bfset[b].expnts[i]);
                    }
                    else if (bfset[b].types[i] == "sp") {
                        double s = 0.25/sqrt(bfset[b].expnts[i]);
                        add_bfn(atoms[a].x  , atoms[a].y  , atoms[a].z  , bfset[b].expnts[i]);
                        add_bfn(atoms[a].x+s, atoms[a].y  , atoms[a].z  , bfset[b].expnts[i]);
                        add_bfn(atoms[a].x-s, atoms[a].y  , atoms[a].z  , bfset[b].expnts[i]);
                        add_bfn(atoms[a].x  , atoms[a].y+s, atoms[a].z  , bfset[b].expnts[i]);
                        add_bfn(atoms[a].x  , atoms[a].y-s, atoms[a].z  , bfset[b].expnts[i]);
                        add_bfn(atoms[a].x  , atoms[a].y  , atoms[a].z+s, bfset[b].expnts[i]);
                        add_bfn(atoms[a].x  , atoms[a].y  , atoms[a].z-s, bfset[b].expnts[i]);
                    }
                    else if (bfset[b].types[i] == "spd") {
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
                        throw "bad atom type building full basis?";
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
}

void read_input() {
    string tag;
    while (cin >> tag) {
        if (tag == "geometry") {
            read_geometry();
        }
        else if (tag == "basis") {
            read_basis_set();
        }
        else {
            throw "unknown directive";
        }
    }
    build_full_basis();
}

int main() {
    read_input();
    return 0;
}
        

