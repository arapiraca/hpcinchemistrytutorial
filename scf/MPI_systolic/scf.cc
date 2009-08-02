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
void make_overlap(int nbf, int lo, int hi, double* overlap) {
    printf("lo = %d    hi = %d\n", lo, hi);
    for (int i=lo; i<hi; i++) {
        for (int j=0; j<nbf; j++) {
            int ij = (i-lo)*nbf + j;
            overlap[ij] = s(i,j);
        }
    }
    printf("s6060 %f\n", s(60,60));
}


/// Makes the one-electron Hamiltonian matrix
void make_hone(int nbf, int lo, int hi, double* hone) {
    for (int i=lo; i<hi; i++) {
        for (int j=0; j<nbf; j++) {
            int ij = (i-lo)*nbf + j;
            hone[ij] = h(i,j);
        }
    }
}


/// Makes the density matrix ... D(mu,nu) = sum(k=1,nocc) C(k,mu) C(k,nu)
void make_dens(int nbf, int lo, int hi, int nocc, const double* C, double* D) {
    return;
    mTxm(nbf, nbf, nocc, C, C, D);
    symmetrize(nbf, D);
}


/// Systolic algorithm computes the integrals 4x too many times
void make_fock_systolic(int nbf, int lo, int hi, const double* D, const double* hone, double* F) {
    return;
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

double make_energy(int nbf, int lo, int hi, 
                   const double* D, const double* hone, const double* F) {
    int nrow = hi - lo;
    double E = dot_product(nbf*nrow,D,hone) + dot_product(nbf*nrow,D,F);
    MPI::COMM_WORLD.Allreduce(MPI_IN_PLACE, &E, 1, MPI::DOUBLE, MPI::SUM);
    return E + nuclear_repulsion_energy();
}


// A is distribted along its column dimension ... collect it onto process 0
void collect(int nbf, int lo, int hi, const double* A, double* A_local) {
    const int rank = MPI::COMM_WORLD.Get_rank();
    const int nproc= MPI::COMM_WORLD.Get_size();
    const int nrow = hi-lo;

    if (rank == 0) {
        memcpy(A_local, A, nrow*nbf*sizeof(double));
        for (int p=1; p<nproc; p++) {
            int plo = min(nbf,p*nrow);
            int phi = min(nbf,plo+nrow);
            int pnrow = phi - plo;
            printf(" COLLECT rank=%d lo=%d hi=%d\n", p, plo, phi);
            if (pnrow > 0) {
                MPI::COMM_WORLD.Recv(A_local+plo*nbf, 
                                     pnrow*nbf*sizeof(double), 
                                     MPI::BYTE, 
                                     p,
                                     1);
            }
        }
    }
    else if (nrow > 0) {
        printf("asdjklfasjklfdasjlf\n");
        print(nrow, nbf, A);
        printf("asdjklfasjklfdasjlf\n");
        MPI::COMM_WORLD.Send(A,
                             nrow*nbf*sizeof(double), 
                             MPI::BYTE, 
                             0,
                             1);
    }
    cout << "RETURNING " << rank << endl;
}

// Opposite of collect
void distribute(int nbf, int lo, int hi, double* A, const double* A_local) {
    const int rank = MPI::COMM_WORLD.Get_rank();
    const int nproc= MPI::COMM_WORLD.Get_size();
    const int nrow = hi-lo;

    if (rank == 0) {
        memcpy(A, A_local, nrow*nbf*sizeof(double));
        for (int p=1; p<nproc; p++) {
            int plo = min(nbf,p*nrow);
            int phi = min(nbf,plo+nrow);
            int pnrow = phi - plo;
            if (pnrow > 0) {
                MPI::COMM_WORLD.Send(A_local+plo*nbf, 
                                     pnrow*nbf*sizeof(double), 
                                     MPI::BYTE, 
                                     p,
                                     1);
            }
        }
    }
    else if (nrow > 0) {
        MPI::COMM_WORLD.Recv(A,
                             nrow*nbf*sizeof(double), 
                             MPI::BYTE, 
                             0,
                             1);
    }
}


void parallel_diag(int nbf, int lo, int hi, 
                   const double* A, double* C, double* eval) {
    // A truly parallel diag is too much like hard work for this
    // example ... collect on one processor, diagonalize, and
    // redistribute
    const int rank = MPI::COMM_WORLD.Get_rank();

    double* A_local = new double[nbf*nbf];
    double* C_local = new double[nbf*nbf];

    MPI::COMM_WORLD.Barrier();

    collect(nbf, lo, hi, A, A_local);
    
    MPI::COMM_WORLD.Barrier();
    if (rank == 0) {
        printf("-------------------\n");
        print(nbf, nbf, A_local);
        printf("*******************\n");
        real_sym_diag(nbf, A_local, C_local, eval);
    }
    MPI::COMM_WORLD.Barrier();

    MPI::COMM_WORLD.Bcast(eval, nbf*sizeof(double), MPI::BYTE, 0);

    distribute(nbf, lo, hi, C, C_local);

    delete C_local;
    delete A_local;

    return;
}

void parallel_generalized_diag(int nbf, int lo, int hi, 
                               const double* A, const double* B, double* C, double* eval) {
    return;
}
            

// For debugging it is convenient to have separate log files
void redirectio() {
    char filename[256];
    sprintf(filename, "log.%5.5d", MPI::COMM_WORLD.Get_rank());
    if (!freopen(filename, "w", stdout)) throw "reopening stdout failed";
    if (!freopen(filename, "w", stderr)) throw "reopening stderr failed";
}


int main(int argc, char** argv) {
    MPI::Init(argc, argv);
    redirectio();
    const int rank = MPI::COMM_WORLD.Get_rank();
    const int nproc= MPI::COMM_WORLD.Get_size();

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

        // Determine which rows of the matrices will be owned
        // by this process ... [lo,hi)
        const int nrow_per_proc = (((nbf+1)/2-1)/nproc + 1)*2;
        const int lo = min(nbf,nrow_per_proc*rank);
        const int hi = min(nbf,lo+nrow_per_proc);
        const int nrow = hi - lo;

        // Allocate space just for my data
        double* overlap = new double[nrow*nbf]; // Overlap matrix
        double* hone = new double[nrow*nbf];    // One-electron Hamiltonian
        double* D = new double[nrow*nbf];       // Density matrix
        double* Dprev = new double[nrow*nbf];   // Density matrix previous
        double* F = new double[nrow*nbf];       // Fock matrix
        double* C = new double[nrow*nbf];       // Molecular orbital coefficients
        double* eval = new double[nbf];         // Eigenvalues
        double eprev = 0.0;                     // Energy previous

        double damp = get_damp();               // Fraction of previous density to mix in
        int ndamp = get_ndamp();                // No. of iterations to damp
        
        make_overlap(nbf, lo, hi, overlap);
        make_hone(nbf, lo, hi, hone);
        
        // Compute eigenvalues of the overlap matrix to monitor linear dependence
        parallel_diag(nbf, lo, hi, overlap, C, eval);
        if (rank == 0) print(nbf, 1, eval);
        if (rank == 0) 
            printf("\nSmallest eigenvalue of the overlap matrix %.1e\n",eval[0]);
        
        // Initial guess for density matrix is zero
        fill(nrow*nbf, 0.0, D);

        if (rank == 0) {
            printf("\n");
            printf(" iter     energy      deltaE    deltaD   damp   time\n");
            printf("------ ------------ --------- --------- ------ ------\n");
        }
        for (int iter=0; iter<50; iter++) {
            // Make the new Fock matrix and the energy
            make_fock_systolic(nbf, lo, hi, D, hone, F);

            double energy = make_energy(nbf, lo, hi, D, hone, F);
            double deltae = energy - eprev;
            eprev = energy;

            parallel_generalized_diag(nbf, lo, hi, F, overlap, C, eval); // Diagonalize
            
            memcpy(Dprev, D, nrow*nbf*sizeof(double)); // Keep copy of D

            make_dens(nbf, lo, hi, nocc, C, D); // Make the new density

            // Compute change in the density to track convergence
            double deltad = 0.0;
            for (int i=0; i<nrow*nbf; i++) {
                double dd = D[i] - Dprev[i];
                deltad += dd*dd;
            }
            MPI::COMM_WORLD.Allreduce(MPI_IN_PLACE, &deltad, 1, MPI::DOUBLE, MPI::SUM);
            deltad = sqrt(deltad);

            // Damp for the first ndamp iterations
            if (iter > ndamp) damp = 0.0;
            if (iter > 0 && damp != 0.0) {
                for (int i=0; i<nrow*nbf; i++) {
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
        cout.flush();
        printf("!! Caught string exception '%s'\n", s);
        fflush(stdout);
        MPI::COMM_WORLD.Abort(1);
    }
    catch (const MPI::Exception& e) {
        cout.flush();
        printf("!! Caught an MPI exception\n");
        fflush(stdout);
        cout << e;
        cout.flush();
        MPI::COMM_WORLD.Abort(1);
    }
    catch (...) {
        cout.flush();
        printf("!! Caught unknown exception\n");
        fflush(stdout);
        MPI::COMM_WORLD.Abort(1);
    }        

    MPI::Finalize();
    return 0;
}
