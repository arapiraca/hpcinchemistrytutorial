/***************************************************************************

                  COPYRIGHT

The following is a notice of limited availability of the code, and disclaimer
which must be included in the prologue of the code and in all source listings
of the code.

Copyright Notice
 + 2009 University of Chicago

Permission is hereby granted to use, reproduce, prepare derivative works, and
to redistribute to others.  This software was authored by:

Jeff R. Hammond
Leadership Computing Facility
Argonne National Laboratory
Argonne IL 60439 USA
phone: (630) 252-5381
e-mail: jhammond@mcs.anl.gov

                  GOVERNMENT LICENSE

Portions of this material resulted from work developed under a U.S.
Government Contract and are subject to the following license: the Government
is granted for itself and others acting on its behalf a paid-up, nonexclusive,
irrevocable worldwide license in this computer software to reproduce, prepare
derivative works, and perform publicly and display publicly.

                  DISCLAIMER

This computer code material was prepared, in part, as an account of work
sponsored by an agency of the United States Government.  Neither the United
States, nor the University of Chicago, nor any of their employees, makes any
warranty express or implied, or assumes any legal liability or responsibility
for the accuracy, completeness, or usefulness of any information, apparatus,
product, or process disclosed, or represents that its use would not infringe
privately owned rights.

 ***************************************************************************/

#ifndef BLAS_UTILS_H
#define BLAS_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef OPENMP
  #include "omp.h"
#else
  #include <time.h>
#endif

double gettime(void);

void zero_host_floats(int num, float* ptr);
void zero_host_doubles(int num, double* ptr);
float* alloc_host_floats(int num);
double* alloc_host_doubles(int num);
void copy_host_floats(int num, float* in_ptr, float* out_ptr);
void copy_host_doubles(int num, double* in_ptr, double* out_ptr);
void free_host_floats(float* ptr);
void free_host_doubles(double* ptr);
void randomize_floats(size_t num, float* ptr);
void randomize_doubles(size_t num, double* ptr);

#endif

