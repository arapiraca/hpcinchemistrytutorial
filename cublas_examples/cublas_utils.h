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

#ifndef CUBLAS_UTILS_H
#define CUBLAS_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef CUDA

#include "cuda_runtime.h"
#include "cublas.h"

// void start_cublas(void);
// void stop_cublas(void);
float* alloc_device_floats(int num);
double* alloc_device_doubles(int num);
void free_device_floats(float* ptr);
void free_device_doubles(double* ptr);
void push_floats(int num, float* h_ptr, float* d_ptr);
void push_doubles(int num, double* h_ptr, double* d_ptr);
void pull_floats(int num, float* d_ptr, float* h_ptr);
void pull_doubles(int num, double* d_ptr, double* h_ptr);

#endif

#endif