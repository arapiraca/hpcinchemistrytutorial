#include <iostream>
#include <cstdio>

using namespace std;

#include "scfdata.h"
#include "integ.h"
#include "matrixutil.h"

// Returns matrix[nbf,nbf] allocated with new and holding overlap matrix
double* make_overlap() {
    int nbf = num_basis_functions();
    double* m = new double[nbf*nbf];

    for (int i=0; i<nbf; i++) {
        for (int j=0; j<=i; j++) {
            int ij = i*nbf + j;
            int ji = j*nbf + i;
            m[ij] = m[ji] = s(i,j);
            printf("s %d %d %12.6f %12.6f\n", i, j, s(i,j), s(j,i));
        }
    }
    return m;
}

// Returns matrix[nbf,nbf] allocated with new and holding one-electron Hamiltonian matrix
double* make_hone() {
    int nbf = num_basis_functions();
    double* m = new double[nbf*nbf];

    for (int i=0; i<nbf; i++) {
        for (int j=0; j<=i; j++) {
            int ij = i*nbf + j;
            int ji = j*nbf + i;
            m[ij] = m[ji] = h(i,j);
        }
    }
    return m;
}

int main() {
    try {
        read_input();
        setfm();
        
        double* s = make_overlap();
        double* h = make_hone();
        
        int nbf = num_basis_functions();
        printf("The overlap matrix\n\n");
        print(nbf, nbf, s);
        printf("\n\nThe one-electron Hamiltonian matrix\n\n");
        print(nbf, nbf, h);
        
        double* C = new double[nbf*nbf];
        double* eval = new double[nbf];
        
        real_sym_gen_diag(nbf, h, s, C, eval);
        
        print(nbf, nbf, C);
        
        print(nbf, eval);
    }
    catch (const char* s) {
        printf("!! Caught string exception '%s'\n", s);
        return 1;
    }

    return 0;
}
