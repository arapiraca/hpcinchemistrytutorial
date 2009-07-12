#ifndef MATRIX_UTIL_H
#define MATRIX_UTIL_H

// Prints a matrix[n,m] in a vaguely human-readable form
void print(int n, int m, const double* s);

// Prints a vector[n] in a vaguely human-readable form
void print(int n, const double* s);

// Makes a new copy of a matrix[n,m] .. result will be allocated with new double[n*m]
double* copy(int n, int m, const double* a);



/// Real symmetric generalized diagonalization using dsygev() ... solves F C = S C E

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
/// Solution satisfies   sum[k] F[i,k] S[k,j] = sum[k] S[i,k] C[k,j] E[j]
///
void real_sym_gen_diag(int n, const double* F, const double* S, double* C, double* E);

#endif
