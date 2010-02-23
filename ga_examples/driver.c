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


#include "driver.h"

int hello(); // hello world
int simple(); // very simple test
int transpose(int dim1, int dim2); // matrix transpose
int matmul(int dim1, int dim2); // matrix multiplication
int matmul2(int dim1, int dim2); // matrix multiplication for symmetric matrices
int matvec(int dim1, int dim2); // fake sparse matrix-vector product
int gemm_test(int dim1);
int overlap(int len); // test of comm/comp overlap
int ga_dgemm_test(int dim1);
int bigtest(int dim1);
int diagonalize(int dim1);
int gemm_test2(int dim1, int dim2, int dim3);
int ga_sgemm_test(int dim1);

int main(int argc, char **argv)
{

    int desired = MPI_THREAD_MULTIPLE;
    int provided;
    MPI_Init_thread(&argc, &argv, desired, &provided);

    int me,nproc;

    MPI_Comm_size(MPI_COMM_WORLD,&nproc);
    MPI_Comm_rank(MPI_COMM_WORLD,&me);

#ifdef GA_INIT_ARGS
    GA_Initialize_args(&argc, &argv);
#else
    GA_Initialize();
#endif
    MA_init(MT_DBL, 128*1024*1024, 32*1024*1024);

#ifdef HPM_PROFILING
    HPM_Init();
#endif

    int test = ( argc>1 ? atoi(argv[1]) :  911 );
    int dim1 = ( argc>2 ? atoi(argv[2]) : 1000 );
    int dim2 = ( argc>3 ? atoi(argv[3]) :   -1 );
    int dim3 = ( argc>4 ? atoi(argv[4]) :   -1 );

    if (test==911){
        printf(" <test> = name(args)\n");
        printf(" 0 = hello()\n");
        printf(" 1 = simple()\n");
        printf(" 2 = transpose(dim1,dim2)\n");
        printf(" 3 = matmul(dim1,dim2)\n");
        printf(" 4 = matmul2(dim1,dim2)\n");
        printf(" 5 = matvec(dim1,dim2)\n");
        printf(" 6 = gemm_test(dim1)\n");
        printf(" 7 = overlap(dim1)\n");
        printf(" 8 = ga_dgemm_test(dim1)\n");
        printf(" 9 = bigtest(dim1)\n");
        printf("10 = diagonalize(dim1)\n");
        printf("11 = gemm_test2(dim1,dim2,dim3)\n");
        printf("12 = ga_sgemm_test(dim1)\n");
        return(1);
    }

    if (me==0) printf("Running test %d\n",test); fflush(stdout);

#ifdef DEBUG
    if ((me==0) && (test != 0) && (test != 6) && (test != 11))
        printf("The result of GA_Nnodes is %d\n",nproc); fflush(stdout);
#endif

    int status=1;
    if (test==0)       status = hello();
    else if (test==1)  status = simple();
    else if (test==2)  status = transpose(dim1,dim2);
    else if (test==3)  status = matmul(dim1,dim2);
    else if (test==4)  status = matmul2(dim1,dim2);
    else if (test==5)  status = matvec(dim1,dim2);
    else if (test==6)  status = gemm_test(dim1);
    else if (test==7)  status = overlap(dim1);
    else if (test==8)  status = ga_dgemm_test(dim1);
    else if (test==9)  status = bigtest(dim1);
    else if (test==10) status = diagonalize(dim1);
    else if (test==11) status = gemm_test2(dim1,dim2,dim3);
    else if (test==12) status = ga_sgemm_test(dim1);

    if ((status!= 0) && (me==0)) printf("test failed\n");

#ifdef HPM_PROFILING
    HPM_Print();
#endif

//     if ((me==0) && (test != 0) && (test != 6) && (test != 11)) GA_Print_stats();

    GA_Terminate();
    MPI_Finalize();

    return(0);
}



