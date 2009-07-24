#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace std;

#include "matrixutil.h"

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

void print(int n, const double* s) {
    print(1, n, s);
}

double* copy(int n, int m, const double* a) {
    double* b = new double[n*m];
    memcpy((void *) b, (void *) a, n*m*sizeof(double));
    return b;
}

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


void mxmT(int dimi, int dimj, int dimk, const double* A, const double* B, double* C) {
    for (int i=0; i<dimi; i++) {
        for (int j=0; j<dimj; j++) {
            double sum = 0.0;
            for (int k=0; k<dimk; k++) {
                sum += A[i*dimk + k] * B[j*dimk + k];
            }
            C[i*dimj + j] = sum;
        }
    }
}


void mTxm(int dimi, int dimj, int dimk, const double* A, const double* B, double* C) {
    for (int i=0; i<dimi; i++) {
        for (int j=0; j<dimj; j++)
            C[i*dimj + j] = 0.0;

        for (int k=0; k<dimk; k++) {
            double aki = A[k*dimi + i];
            for (int j=0; j<dimj; j++) {
                C[i*dimj + j] += aki * B[k*dimj + j];
            }
        }
    }
}


void fill(int n, double s, double* v) {
    for (int i=0; i<n; i++)
        v[i] = s;
}

double dot_product(int n, const double* a, const double* b) {
    double sum = 0.0;
    for (int i=0; i<n; i++) sum += a[i]*b[i];
    return sum;
}

void symmetrize(int n, double* a) {
    for (int i=0; i<n; i++) {
        for (int j=0; j<i; j++) {
            int ij = i*n + j;
            int ji = j*n + i;
            a[ij] = a[ji] = 0.5*(a[ij] + a[ji]);
        }
    }
}

