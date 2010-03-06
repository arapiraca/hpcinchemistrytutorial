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

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <mpi.h>

unsigned long long getticks(void);

void flush_cache(size_t n, float alpha, float* in, float* out)
{
    int i;
    for (i=0;i<n;i++) out[i] = alpha*in[i];
}

void test_01arg(int* arg01)
{
   (*arg01)++;
}

void test_02arg(int* arg01, int* arg02)
{
   (*arg01)++;
   (*arg02)++;
}

void test_03arg(int* arg01, int* arg02, int* arg03)
{
   (*arg01)++;
   (*arg02)++;
   (*arg03)++;
}

void test_04arg(int* arg01, int* arg02, int* arg03, int* arg04)
{
   (*arg01)++;
   (*arg02)++;
   (*arg03)++;
   (*arg04)++;
}

void test_05arg(int* arg01, int* arg02, int* arg03, int* arg04,
                int* arg05)
{
   (*arg01)++;
   (*arg02)++;
   (*arg03)++;
   (*arg04)++;
   (*arg05)++;
}

void test_06arg(int* arg01, int* arg02, int* arg03, int* arg04,
                int* arg05, int* arg06)
{
   (*arg01)++;
   (*arg02)++;
   (*arg03)++;
   (*arg04)++;
   (*arg05)++;
   (*arg06)++;
}

void test_07arg(int* arg01, int* arg02, int* arg03, int* arg04,
                int* arg05, int* arg06, int* arg07)
{
   (*arg01)++;
   (*arg02)++;
   (*arg03)++;
   (*arg04)++;
   (*arg05)++;
   (*arg06)++;
   (*arg07)++;
}

void test_08arg(int* arg01, int* arg02, int* arg03, int* arg04,
                int* arg05, int* arg06, int* arg07, int* arg08)
{
   (*arg01)++;
   (*arg02)++;
   (*arg03)++;
   (*arg04)++;
   (*arg05)++;
   (*arg06)++;
   (*arg07)++;
   (*arg08)++;
}

void test_09arg(int* arg01, int* arg02, int* arg03, int* arg04, 
                int* arg05, int* arg06, int* arg07, int* arg08, 
                int* arg09)
{
   (*arg01)++;
   (*arg02)++;
   (*arg03)++;
   (*arg04)++;
   (*arg05)++;
   (*arg06)++;
   (*arg07)++;
   (*arg08)++;
   (*arg09)++;
}

void test_10arg(int* arg01, int* arg02, int* arg03, int* arg04, 
                int* arg05, int* arg06, int* arg07, int* arg08, 
                int* arg09, int* arg10)
{
   (*arg01)++;
   (*arg02)++;
   (*arg03)++;
   (*arg04)++;
   (*arg05)++;
   (*arg06)++;
   (*arg07)++;
   (*arg08)++;
   (*arg09)++;
   (*arg10)++;
}

void test_11arg(int* arg01, int* arg02, int* arg03, int* arg04, 
                int* arg05, int* arg06, int* arg07, int* arg08, 
                int* arg09, int* arg10, int* arg11)
{
   (*arg01)++;
   (*arg02)++;
   (*arg03)++;
   (*arg04)++;
   (*arg05)++;
   (*arg06)++;
   (*arg07)++;
   (*arg08)++;
   (*arg09)++;
   (*arg10)++;
   (*arg11)++;
}

void test_12arg(int* arg01, int* arg02, int* arg03, int* arg04, 
                int* arg05, int* arg06, int* arg07, int* arg08, 
                int* arg09, int* arg10, int* arg11, int* arg12)
{
   (*arg01)++;
   (*arg02)++;
   (*arg03)++;
   (*arg04)++;
   (*arg05)++;
   (*arg06)++;
   (*arg07)++;
   (*arg08)++;
   (*arg09)++;
   (*arg10)++;
   (*arg11)++;
   (*arg12)++;
}

void test_13arg(int* arg01, int* arg02, int* arg03, int* arg04, 
                int* arg05, int* arg06, int* arg07, int* arg08, 
                int* arg09, int* arg10, int* arg11, int* arg12,
                int* arg13)
{
   (*arg01)++;
   (*arg02)++;
   (*arg03)++;
   (*arg04)++;
   (*arg05)++;
   (*arg06)++;
   (*arg07)++;
   (*arg08)++;
   (*arg09)++;
   (*arg10)++;
   (*arg11)++;
   (*arg12)++;
   (*arg13)++;
}

void test_14arg(int* arg01, int* arg02, int* arg03, int* arg04, 
                int* arg05, int* arg06, int* arg07, int* arg08, 
                int* arg09, int* arg10, int* arg11, int* arg12,
                int* arg13, int* arg14)
{
   (*arg01)++;
   (*arg02)++;
   (*arg03)++;
   (*arg04)++;
   (*arg05)++;
   (*arg06)++;
   (*arg07)++;
   (*arg08)++;
   (*arg09)++;
   (*arg10)++;
   (*arg11)++;
   (*arg12)++;
   (*arg13)++;
   (*arg14)++;
}

void test_15arg(int* arg01, int* arg02, int* arg03, int* arg04, 
                int* arg05, int* arg06, int* arg07, int* arg08, 
                int* arg09, int* arg10, int* arg11, int* arg12,
                int* arg13, int* arg14, int* arg15)
{
   (*arg01)++;
   (*arg02)++;
   (*arg03)++;
   (*arg04)++;
   (*arg05)++;
   (*arg06)++;
   (*arg07)++;
   (*arg08)++;
   (*arg09)++;
   (*arg10)++;
   (*arg11)++;
   (*arg12)++;
   (*arg13)++;
   (*arg14)++;
   (*arg15)++;
}

void test_16arg(int* arg01, int* arg02, int* arg03, int* arg04, 
                int* arg05, int* arg06, int* arg07, int* arg08, 
                int* arg09, int* arg10, int* arg11, int* arg12,
                int* arg13, int* arg14, int* arg15, int* arg16)
{
   (*arg01)++;
   (*arg02)++;
   (*arg03)++;
   (*arg04)++;
   (*arg05)++;
   (*arg06)++;
   (*arg07)++;
   (*arg08)++;
   (*arg09)++;
   (*arg10)++;
   (*arg11)++;
   (*arg12)++;
   (*arg13)++;
   (*arg14)++;
   (*arg15)++;
   (*arg16)++;
}

void test_17arg(int* arg01, int* arg02, int* arg03, int* arg04, 
                int* arg05, int* arg06, int* arg07, int* arg08, 
                int* arg09, int* arg10, int* arg11, int* arg12,
                int* arg13, int* arg14, int* arg15, int* arg16,
                int* arg17)
{
   (*arg01)++;
   (*arg02)++;
   (*arg03)++;
   (*arg04)++;
   (*arg05)++;
   (*arg06)++;
   (*arg07)++;
   (*arg08)++;
   (*arg09)++;
   (*arg10)++;
   (*arg11)++;
   (*arg12)++;
   (*arg13)++;
   (*arg14)++;
   (*arg15)++;
   (*arg16)++;
   (*arg17)++;
}

void test_18arg(int* arg01, int* arg02, int* arg03, int* arg04, 
                int* arg05, int* arg06, int* arg07, int* arg08, 
                int* arg09, int* arg10, int* arg11, int* arg12,
                int* arg13, int* arg14, int* arg15, int* arg16,
                int* arg17, int* arg18)
{
   (*arg01)++;
   (*arg02)++;
   (*arg03)++;
   (*arg04)++;
   (*arg05)++;
   (*arg06)++;
   (*arg07)++;
   (*arg08)++;
   (*arg09)++;
   (*arg10)++;
   (*arg11)++;
   (*arg12)++;
   (*arg13)++;
   (*arg14)++;
   (*arg15)++;
   (*arg16)++;
   (*arg17)++;
   (*arg18)++;
}

void test_19arg(int* arg01, int* arg02, int* arg03, int* arg04, 
                int* arg05, int* arg06, int* arg07, int* arg08, 
                int* arg09, int* arg10, int* arg11, int* arg12,
                int* arg13, int* arg14, int* arg15, int* arg16,
                int* arg17, int* arg18, int* arg19)
{
   (*arg01)++;
   (*arg02)++;
   (*arg03)++;
   (*arg04)++;
   (*arg05)++;
   (*arg06)++;
   (*arg07)++;
   (*arg08)++;
   (*arg09)++;
   (*arg10)++;
   (*arg11)++;
   (*arg12)++;
   (*arg13)++;
   (*arg14)++;
   (*arg15)++;
   (*arg16)++;
   (*arg17)++;
   (*arg18)++;
   (*arg19)++;
}

void test_20arg(int* arg01, int* arg02, int* arg03, int* arg04, 
                int* arg05, int* arg06, int* arg07, int* arg08, 
                int* arg09, int* arg10, int* arg11, int* arg12,
                int* arg13, int* arg14, int* arg15, int* arg16,
                int* arg17, int* arg18, int* arg19, int* arg20)
{
   (*arg01)++;
   (*arg02)++;
   (*arg03)++;
   (*arg04)++;
   (*arg05)++;
   (*arg06)++;
   (*arg07)++;
   (*arg08)++;
   (*arg09)++;
   (*arg10)++;
   (*arg11)++;
   (*arg12)++;
   (*arg13)++;
   (*arg14)++;
   (*arg15)++;
   (*arg16)++;
   (*arg17)++;
   (*arg18)++;
   (*arg19)++;
   (*arg20)++;
}

int main(int argc, char **argv)
{
    int desired = MPI_THREAD_SINGLE;
    int provided;
    MPI_Init_thread(&argc, &argv, desired, &provided);

    int me;
    int nproc;
    MPI_Comm_rank(MPI_COMM_WORLD,&me);
    MPI_Comm_size(MPI_COMM_WORLD,&nproc);

    /********************************************/

    unsigned long long int t0, t1;

    int arg01=0;
    int arg02=0;
    int arg03=0;
    int arg04=0;
    int arg05=0;
    int arg06=0;
    int arg07=0;
    int arg08=0;
    int arg09=0;
    int arg10=0;
    int arg11=0;
    int arg12=0;
    int arg13=0;
    int arg14=0;
    int arg15=0;
    int arg16=0;
    int arg17=0;
    int arg18=0;
    int arg19=0;
    int arg20=0;

    size_t i;
    size_t n = 16384;
    float alpha = 13.7;
    float* in  = (float*) malloc(n*sizeof(float));
    float* out = (float*) malloc(n*sizeof(float));
    
    for (i=0;i<n;i++) in[i]  = 0.9;
    for (i=0;i<n;i++) out[i] = 1.1;

    flush_cache(n,alpha,in,out);

    t0=getticks();
    test_01arg(&arg01);
    t1=getticks();
    printf("test_01arg = %llu\n",t1-t0);

    flush_cache(n,alpha,in,out);

    t0=getticks();
    test_02arg(&arg01, &arg02);
    t1=getticks();
    printf("test_02arg = %llu\n",t1-t0);

    flush_cache(n,alpha,in,out);

    t0=getticks();
    test_03arg(&arg01, &arg02, &arg03);
    t1=getticks();
    printf("test_03arg = %llu\n",t1-t0);

    flush_cache(n,alpha,in,out);

    t0=getticks();
    test_04arg(&arg01, &arg02, &arg03, &arg04);
    t1=getticks();
    printf("test_04arg = %llu\n",t1-t0);

    flush_cache(n,alpha,in,out);

    t0=getticks();
    test_05arg(&arg01, &arg02, &arg03, &arg04,
               &arg05);
    t1=getticks();
    printf("test_05arg = %llu\n",t1-t0);

    flush_cache(n,alpha,in,out);

    t0=getticks();
    test_06arg(&arg01, &arg02, &arg03, &arg04,
               &arg05, &arg06);
    t1=getticks();
    printf("test_06arg = %llu\n",t1-t0);

    flush_cache(n,alpha,in,out);

    t0=getticks();
    test_07arg(&arg01, &arg02, &arg03, &arg04,
               &arg05, &arg06, &arg07);
    t1=getticks();
    printf("test_07arg = %llu\n",t1-t0);

    flush_cache(n,alpha,in,out);

    t0=getticks();
    test_08arg(&arg01, &arg02, &arg03, &arg04,
               &arg05, &arg06, &arg07, &arg08);
    t1=getticks();
    printf("test_08arg = %llu\n",t1-t0);

    flush_cache(n,alpha,in,out);

    t0=getticks();
    test_09arg(&arg01, &arg02, &arg03, &arg04,
               &arg05, &arg06, &arg07, &arg08,
               &arg09);
    t1=getticks();
    printf("test_09arg = %llu\n",t1-t0);

    flush_cache(n,alpha,in,out);

    t0=getticks();
    test_10arg(&arg01, &arg02, &arg03, &arg04,
               &arg05, &arg06, &arg07, &arg08,
               &arg09, &arg10);
    t1=getticks();
    printf("test_10arg = %llu\n",t1-t0);

    flush_cache(n,alpha,in,out);

    t0=getticks();
    t0=getticks();
    test_11arg(&arg01, &arg02, &arg03, &arg04,
               &arg05, &arg06, &arg07, &arg08,
               &arg09, &arg10, &arg11);
    t1=getticks();
    printf("test_11arg = %llu\n",t1-t0);

    flush_cache(n,alpha,in,out);

    t0=getticks();
    t0=getticks();
    test_12arg(&arg01, &arg02, &arg03, &arg04,
               &arg05, &arg06, &arg07, &arg08,
               &arg09, &arg10, &arg11, &arg12);
    t1=getticks();
    printf("test_12arg = %llu\n",t1-t0);

    flush_cache(n,alpha,in,out);

    t0=getticks();
    t0=getticks();
    test_13arg(&arg01, &arg02, &arg03, &arg04,
               &arg05, &arg06, &arg07, &arg08,
               &arg09, &arg10, &arg11, &arg12,
               &arg13);
    t1=getticks();
    printf("test_13arg = %llu\n",t1-t0);

    flush_cache(n,alpha,in,out);

    t0=getticks();
    test_14arg(&arg01, &arg02, &arg03, &arg04,
               &arg05, &arg06, &arg07, &arg08,
               &arg09, &arg10, &arg11, &arg12,
               &arg13, &arg14);
    t1=getticks();
    printf("test_14arg = %llu\n",t1-t0);

    flush_cache(n,alpha,in,out);

    t0=getticks();
    test_15arg(&arg01, &arg02, &arg03, &arg04,
               &arg05, &arg06, &arg07, &arg08,
               &arg09, &arg10, &arg11, &arg12,
               &arg13, &arg14, &arg15);
    t1=getticks();
    printf("test_15arg = %llu\n",t1-t0);

    flush_cache(n,alpha,in,out);

    t0=getticks();
    test_16arg(&arg01, &arg02, &arg03, &arg04,
               &arg05, &arg06, &arg07, &arg08,
               &arg09, &arg10, &arg11, &arg12,
               &arg13, &arg14, &arg15, &arg16);
    t1=getticks();
    printf("test_16arg = %llu\n",t1-t0);

    flush_cache(n,alpha,in,out);

    t0=getticks();
    test_17arg(&arg01, &arg02, &arg03, &arg04,
               &arg05, &arg06, &arg07, &arg08,
               &arg09, &arg10, &arg11, &arg12,
               &arg13, &arg14, &arg15, &arg16,
               &arg17);
    t1=getticks();
    printf("test_17arg = %llu\n",t1-t0);

    flush_cache(n,alpha,in,out);

    t0=getticks();
    test_18arg(&arg01, &arg02, &arg03, &arg04,
               &arg05, &arg06, &arg07, &arg08,
               &arg09, &arg10, &arg11, &arg12,
               &arg13, &arg14, &arg15, &arg16,
               &arg17, &arg18);
    t1=getticks();
    printf("test_18arg = %llu\n",t1-t0);

    flush_cache(n,alpha,in,out);

    t0=getticks();
    test_19arg(&arg01, &arg02, &arg03, &arg04,
               &arg05, &arg06, &arg07, &arg08,
               &arg09, &arg10, &arg11, &arg12,
               &arg13, &arg14, &arg15, &arg16,
               &arg17, &arg18, &arg19);
    t1=getticks();
    printf("test_19arg = %llu\n",t1-t0);

    flush_cache(n,alpha,in,out);

    t0=getticks();
    test_20arg(&arg01, &arg02, &arg03, &arg04,
               &arg05, &arg06, &arg07, &arg08,
               &arg09, &arg10, &arg11, &arg12,
               &arg13, &arg14, &arg15, &arg16,
               &arg17, &arg18, &arg19, &arg20);
    t1=getticks();
    printf("test_20arg = %llu\n",t1-t0);

    flush_cache(n,alpha,in,out);

    /********************************************/

    MPI_Finalize();

    return(0);
}



