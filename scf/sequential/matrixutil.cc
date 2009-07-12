#include <cstdio>
#include <cstdlib>
#include <cstring>
using namespace std;

// Prints a matrix[n,m] in a vaguely human-readable form
void print(int n, int m, const double* s) {
    printf("    ");
    for (int j=0; j<m; j++) 
        printf("    %4d    ", j);
    printf("\n");

    printf("    ");
    for (int j=0; j<m; j++) 
        printf(" -----------");
    printf("\n");

    for (int i=0; i<n; i++) {
        printf("%3d ", i);
        for (int j=0; j<m; j++)
            printf("%12.4f", *s++);
        printf("\n");
    }
}

// Prints a vector[n] in a vaguely human-readable form
void print(int n, const double* s) {
    print(1, n, s);
}

// Makes a new copy of a matrix[n,m] .. result will be allocated with new double[n*m]
double* copy(int n, int m, const double* a) {
    double* b = new double[n*m];
    memcpy((void *) b, (void *) a, n*m*sizeof(double));
    return b;
}




typedef long integer;           // C type of Fortran integers

extern "C"  void dsygv_(const integer* ITYPE, 
                        const char* JOB,
                        const char* UPLO,
                        integer* N, 
                        double* A, 
                        const integer* LDA, 
                        double* B, 
                        const integer* LDB, 
                        double* E, 
                        double* WORK, 
                        const integer* LWORK, 
                        integer* INFO, 
                        int joblen, 
                        int ulolen);

extern "C"  void dsyev_(const char* JOB,
                        const char* UPLO,
                        integer* N, 
                        double* A, 
                        const integer* LDA, 
                        double* E, 
                        double* WORK, 
                        const integer* LWORK, 
                        integer* INFO, 
                        int joblen, 
                        int ulolen);


/// Real symmetric generalized diagonalization using dsygv() ... solves F C = S C E

/// n = matrix dimension
///
/// F[n*n] = input matrix ... unchanged on output
///
/// S[n*n] = input matrix ... unchanged on output
///
/// C[n*n] = output matrix ... i'th row of C contains i'th eigenvector
///
/// E[n] = output vector ... E[i] = i'th eigenvalue
///
/// Solution satisfies   sum[k] F[i,k] C[j,k] = sum[k] S[i,k] C[j,k] E[j]
///
void real_sym_gen_diag(int n, const double* F, const double* S, double* C, double* E) {
    memcpy(C, F, n*n*sizeof(double));
    double* B = copy(n, n, S);
    double* WORK = new double[n*32];
    
    integer N = n;
    integer ITYPE = 1;
    integer LDA = n;
    integer LDB = n;
    integer LWORK = n*32;
    integer INFO=0;

    dsygv_(&ITYPE, "V", "U", &N, C, &LDA, B, &LDB, E, WORK, &LWORK, &INFO, 1, 1);

    if (INFO != 0) {
        printf("!!! info from dsygv = %ld\n", INFO);
        fflush(stdout);
        throw "LAPACK dsygv returned non-zero info";
    }

    delete [] WORK;
    delete [] B;
}


/// Real symmetric diagonalization using dsyev() ... solves A C = C E

/// n = matrix dimension
///
/// A[n*n] = input matrix ... unchanged on output
///
/// C[n*n] = output matrix ... i'th row of C contains i'th eigenvector
///
/// E[n] = output vector ... E[i] = i'th eigenvalue
///
/// Solution satisfies   sum[k] A[i,k] C[j,k] = C[i,j] E[j]
///
void real_sym_diag(int n, const double* A, double* C, double* E) {
    memcpy(C, A, n*n*sizeof(double));
    double* WORK = new double[n*32];
    
    integer N = n;
    integer LDA = n;
    integer LWORK = n*32;
    integer INFO=0;

    dsyev_("V", "U", &N, C, &LDA, E, WORK, &LWORK, &INFO, 1, 1);

    if (INFO != 0) {
        printf("!!! info from dsyev = %ld\n", INFO);
        fflush(stdout);
        throw "LAPACK dsyev returned non-zero info";
    }

    delete [] WORK;
}
