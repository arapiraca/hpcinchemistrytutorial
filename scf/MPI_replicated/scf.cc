#include <iostream>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <mpi.h>

using namespace std;

#include "scfdata.h"
#include "integ.h"
#include "matrixutil.h"

/// Makes the overlap matrix
void make_overlap(int nbf, double* overlap) {

    fill(nbf*nbf, 0.0, overlap);

    const int rank = MPI::COMM_WORLD.Get_rank();
    const int nproc= MPI::COMM_WORLD.Get_size();

    for (int i=rank; i<nbf; i+=nproc) {
        for (int j=0; j<=i; j++) {
            int ij = i*nbf + j;
            int ji = j*nbf + i;
            overlap[ij] = overlap[ji] = s(i,j);
        }
    }

    MPI::COMM_WORLD.Allreduce(MPI_IN_PLACE, overlap, nbf*nbf, MPI::DOUBLE, MPI::SUM);
}

/// Makes the one-electron Hamiltonian matrix
void make_hone(int nbf, double* hone) {

    fill(nbf*nbf, 0.0, hone);

    const int rank = MPI::COMM_WORLD.Get_rank();
    const int nproc= MPI::COMM_WORLD.Get_size();

    for (int i=rank; i<nbf; i+=nproc) {
        for (int j=0; j<=i; j++) {
            int ij = i*nbf + j;
            int ji = j*nbf + i;
            hone[ij] = hone[ji] = h(i,j);
        }
    }

    MPI::COMM_WORLD.Allreduce(MPI_IN_PLACE, hone, nbf*nbf, MPI::DOUBLE, MPI::SUM);
}

/// Makes the density matrix ... D(mu,nu) = sum(k=1,nocc) C(k,mu) C(k,nu)
void make_dens(int nbf, int nocc, const double* C, double* D) {
    mTxm(nbf, nbf, nocc, C, C, D);
    symmetrize(nbf, D);
}

/// Builds the closed-shell Fock matrix in the simplest manner possible ... but slow!

/// F[mu,nu] = h[mu,nu] + sum[omega,lambda] D[omega,lambda]*(2*g(mu,nu,omega,lambda) - g(mu,omega,nu,lambda))
///
/// Computes the integrals about 16x too many times and does not use any sparsity.
void make_fock_very_simple(int nbf, const double* D, const double* hone, double* F) {

    fill(nbf*nbf, 0.0, F);

    const int rank = MPI::COMM_WORLD.Get_rank();
    const int nproc= MPI::COMM_WORLD.Get_size();

    int count = 0;
    for (int mu=0; mu<nbf; mu++) {
        for (int nu=0; nu<nbf; nu++) {
            if ( (count%nproc) == rank) {
                double sum = 0.0;
                for (int omega=0; omega<nbf; omega++) {
                    for (int lambda=0; lambda<nbf; lambda++) {
                        sum += D[omega*nbf + lambda] * (2.0*g(mu,nu,omega,lambda) - g(mu,omega,nu,lambda));
                    }
                }
                F[mu*nbf + nu] = hone[mu*nbf + nu] + sum;
            }
            count++;
        }
    }
    MPI::COMM_WORLD.Allreduce(MPI_IN_PLACE, F, nbf*nbf, MPI::DOUBLE, MPI::SUM);

    symmetrize(nbf, F);
}

/// Builds the closed-shell Fock matrix in a little more efficient manner

/// F[mu,nu] = h[mu,nu] + sum[omega,lambda] D[omega,lambda]*(2*g(mu,nu,omega,lambda) - g(mu,omega,nu,lambda))
///
/// Computes the integrals the minimum number of times but does not use any sparsity.
/// Algorithm for adding Fock matrix elements is slow, but it works.
void make_fock_simple(int nbf, double* D, const double* hone, double* F) {
#define FF(i,j) F[i*nbf + j]
#define DD(i,j) D[i*nbf + j]

    fill(nbf*nbf, 0.0, F);

    const int rank = MPI::COMM_WORLD.Get_rank();
    const int nproc= MPI::COMM_WORLD.Get_size();

    int count = 0;
    for (int mu=0; mu<nbf; mu++) {
        for (int nu=0; nu<=mu; nu++) {
            if ( (count%nproc) == rank ) {
                double facmunu = 4.0;
                if (mu == nu) facmunu *= 0.5;
                for (int omega=0; omega<=mu; omega++) {
                    int lambdatop = omega;
                    if (omega  ==  mu) lambdatop = nu;
                    for (int lambda=0; lambda<=lambdatop; lambda++) {
                        double fac = facmunu;
                        if (omega == lambda) fac *= 0.5;
                        if (mu == omega && nu == lambda) fac *= 0.5;
                        double gval = g(mu,nu,omega,lambda)*fac;
                        FF(mu,nu)        += DD(omega,lambda)*gval*2.0;
                        FF(omega,lambda) += DD(mu,nu)*gval*2.0;
                        FF(mu,omega)     -= DD(nu,lambda)*gval*0.5;
                        FF(mu,lambda)    -= DD(nu,omega)*gval*0.5;
                        FF(nu,omega)     -= DD(mu,lambda)*gval*0.5;
                        FF(nu,lambda)    -= DD(mu,omega)*gval*0.5;
                    }
                }
            }
            count++;
        }
    }
    
    MPI::COMM_WORLD.Allreduce(MPI_IN_PLACE, F, nbf*nbf, MPI::DOUBLE, MPI::SUM);

    for (int i=0; i<nbf*nbf; i++) F[i] += hone[i];

    symmetrize(nbf, F);
#undef FF
#undef DD
}


double make_energy(int nbf, const double* D, const double* hone, const double* F) {
    return nuclear_repulsion_energy() + dot_product(nbf*nbf,D,hone) + dot_product(nbf*nbf,D,F);
}

int main(int argc, char** argv) {
    MPI::Init(argc, argv);
    const int rank = MPI::COMM_WORLD.Get_rank();

    try {
        read_input();
        setfm();
        wall_time();            // Initialize timer

        const int nbf = num_basis_functions();
        const int nelec = num_electrons();
        if ((nelec&0x1)) 
            throw "Number of electrons is odd";
        const int nocc = nelec/2;

        if (rank == 0) printf("\nNumber of doubly-occupied orbitals %d\n", nocc);

        double* overlap = new double[nbf*nbf]; // Overlap matrix
        double* hone = new double[nbf*nbf];    // One-electron Hamiltonian
        double* D = new double[nbf*nbf];       // Density matrix
        double* Dprev = new double[nbf*nbf];   // Density matrix previous
        double* C = new double[nbf*nbf];       // Molecular orbital coefficients
        double* F = new double[nbf*nbf];       // Fock matrix
        double* eval = new double[nbf];        // Eigenvalues
        double eprev = 0.0;                          // Energy previous

        double damp = get_damp();              // Fraction of previous density to mix in
        int ndamp = get_ndamp();               // No. of iterations to damp
        
        make_overlap(nbf, overlap);
        make_hone(nbf, hone);
        // printf("The overlap matrix\n\n");
        // print(nbf, nbf, overlap);
        // printf("The one-electron Hamiltonian\n\n");
        // print(nbf, nbf, hone);
        
        // Compute eigenvalues of the overlap matrix to monitor linear dependence
        real_sym_diag(nbf, overlap, C, eval);
        if (rank == 0) 
            printf("\nSmallest eigenvalue of the overlap matrix %.1e\n",eval[0]);
        
        // Initial guess for density matrix is zero
        fill(nbf*nbf, 0.0, D);

        if (rank == 0) {
            printf("\n");
            printf(" iter     energy      deltaE    deltaD   damp   time\n");
            printf("------ ------------ --------- --------- ------ ------\n");
        }
        for (int iter=0; iter<50; iter++) {
            // Make the new Fock matrix and the energy

            //make_fock_very_simple(nbf, D, hone, F);
            //print(nbf,nbf,F);
            make_fock_simple(nbf, D, hone, F);
            //print(nbf,nbf,F);

            double energy = make_energy(nbf, D, hone, F);
            double deltae = energy - eprev;
            eprev = energy;

            real_sym_gen_diag(nbf, F, overlap, C, eval); // Diagonalize
            
            memcpy(Dprev, D, nbf*nbf*sizeof(double)); // Keep copy of D

            make_dens(nbf, nocc, C, D); // Make the new density

            // Compute change in the density to track convergence
            double deltad = 0.0;
            for (int i=0; i<nbf*nbf; i++) {
                double dd = D[i] - Dprev[i];
                deltad += dd*dd;
            }
            deltad = sqrt(deltad);

            // Damp for the first ndamp iterations
            if (iter > ndamp) damp = 0.0;
            if (iter > 0 && damp != 0.0) {
                for (int i=0; i<nbf*nbf; i++) {
                    D[i] = (1.0-damp)*D[i] + damp*Dprev[i];
                }
            }

            // Let the user know what's going on
            if (rank == 0)
                printf("%6d %12.6f %9.1e %9.1e %6.1f %6.1f\n",
                       iter, energy, deltae, deltad, damp, wall_time());


            // Broadcast converged status to ensure everyone agrees
            bool converged = abs(deltae) < 1e-6 && deltad < 1e-3;
            MPI::COMM_WORLD.Bcast(&converged, sizeof(converged), MPI::BYTE, 0);

            if (converged) {
                if (rank == 0) printf("\n Converged!\n");
                break;
            }
        }

        if (rank == 0) printf("\n Total wall time = %.1f\n", wall_time());

        // Print final results
        //printf("\nFinal eigen-values\n");
        //print(nbf, 1, eval);
        //printf("\nFinal eigen-vectors (in rows)\n");
        //print(nbf, nbf, eval);
    }
    catch (const char* s) {
        printf("!! Caught string exception '%s'\n", s);
        return 1;
    }

    MPI::Finalize();
    return 0;
}
