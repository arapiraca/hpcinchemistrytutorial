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
int transpose(int rank, int blksz); // matrix transpose
int matmul(int rank, int blksz); // matrix multiplication
int matmul2(int rank, int blksz); // matrix multiplication for symmetric matrices
int matvec(int rank, int blksz); // fake sparse matrix-vector product
int gemm_test(int rank);
int overlap(int len); // test of comm/comp overlap
int bigtest(int rank);

#ifndef DCMF
unsigned long long DCMF_Timebase(void)
{
    return (unsigned long long) clock();
}
#endif

int main(int argc, char **argv)
{
	int me,nproc;
    int test=0;
    int status;

    int rank,blksz;

    rank = 6;
    blksz = 2;

    int desired = MPI_THREAD_MULTIPLE;
    int provided;
    MPI_Init_thread(&argc, &argv, desired, &provided);

    //nproc=GA_Nnodes();
    //me=GA_Nodeid();

    MPI_Comm_size(MPI_COMM_WORLD,&nproc);
    MPI_Comm_rank(MPI_COMM_WORLD,&me);

    if ( provided != MPI_THREAD_MULTIPLE ){
      if ( me == 0 ) fprintf(stderr,"provided != MPI_THREAD_MULTIPLE\n");
    }

#ifdef GA_INIT_ARGS
    GA_Initialize_args(&argc, &argv);
#else
    GA_Initialize();
#endif
    MA_init(MT_DBL, 32*1024*1024, 2*1024*1024);

#ifdef HPM_PROFILING
    HPM_Init();
#endif

    if (argc > 1){
        test = atoi(argv[1]);
    } else {
        printf("0 = hello\n");
        printf("1 = simple\n");
        printf("2 = transpose\n");
        printf("3 = matmul\n");
        printf("4 = matmul2\n");
        printf("5 = matvec\n");
        printf("6 = gemm_test\n");
        printf("7 = overlap\n");
        printf("8 = ga_dgemm_test\n");
        printf("9 = bigtest\n");
        return(1);        
    }

    if (me == 0){
        printf("Running test %d\n",test);
        fflush(stdout);
    }
    if (test > 1){
        if (argc  > 2){
            rank = atoi(argv[2]);
        } else {
            rank = 1000;
        }
        if (argc  > 3){
            blksz = atoi(argv[3]);
        } else {
            blksz = -1;
        }
    }

#ifdef DEBUG
    if(me == 0){
       printf("The result of GA_Nnodes is %d\n",nproc);
       fflush(stdout);
    }
#endif

    if (test == 0){
        status = hello();
        if(status != 0){
            if (me == 0){
                printf("%s: hello() failed at line %d\n",__FILE__,__LINE__);
                fflush(stdout);
            }
        }
    } else if (test == 1){
        if(me == 0){
            printf("Running simple with %d processes\n",nproc);
            fflush(stdout);
        }
        status = simple();
        if(status != 0){
        	if (me == 0){
                printf("%s: simple() failed at line %d\n",__FILE__,__LINE__);
                fflush(stdout);
            }
        }
    } else if (test == 2){
        if(me == 0){
            printf("Running transpose with %d processes\n",nproc);
            fflush(stdout);
        }
        status = transpose(rank,blksz);
        if(status != 0){
        	if (me == 0){
                printf("%s: transpose() failed at line %d\n",__FILE__,__LINE__);
                fflush(stdout);
            }
        }
    } else if (test == 3){
        if(me == 0){
            printf("Running matmul with %d processes\n",nproc);
            fflush(stdout);
        }
        status = matmul(rank,blksz);
        if(status != 0){
        	if (me == 0){
                printf("%s: matmul() failed at line %d\n",__FILE__,__LINE__);
                fflush(stdout);
            }
        }
    } else if (test == 4){
        if(me == 0){
            printf("Running matmul2 with %d processes\n",nproc);
            fflush(stdout);
        }
        status = matmul2(rank,blksz);
        if(status != 0){
            if (me == 0){
                printf("%s: matmul2() failed at line %d\n",__FILE__,__LINE__);
                fflush(stdout);
            }
        }
    } else if (test == 5){
        if(me == 0){
            printf("Running matvec with %d processes\n",nproc);
            fflush(stdout);
        }
        status = matvec(rank,blksz);
        if(status != 0){
            if (me == 0){
                printf("%s: matvec() failed at line %d\n",__FILE__,__LINE__);
                fflush(stdout);
            }
        }
    } else if (test == 6){
        if(me == 0){
            printf("Running gemm_test on process %d\n",me);
            fflush(stdout);
            status = gemm_test(rank);
            if(status != 0){
                printf("%s: gemm_test() failed at line %d\n",__FILE__,__LINE__);
                fflush(stdout);
            }
        }
    } else if (test == 7){
        if(nproc%2 != 0){
            if (me == 0){
                printf("You need to use an even number of processes\n");
                fflush(stdout);
                ARMCI_Cleanup();
                MPI_Abort(MPI_COMM_WORLD,test);
            }
        }
        if(me == 0){
            printf("Running overlap with %d processes\n",nproc);
            fflush(stdout);
        }
        int len;
        if (argc  > 2){
            len = atoi(argv[2]);
        } else {
            len = 10;
        }
        status = overlap(len);
        if(status != 0){
            if (me == 0){
                printf("%s: overlap() failed at line %d\n",__FILE__,__LINE__);
                fflush(stdout);
            }
        }
    } else if (test == 8){
        if(me == 0){
            printf("Running ga_dgemm_test with %d processes\n",nproc);
            fflush(stdout);
        }
        status = ga_dgemm_test(rank);
        if(status != 0){
            if (me == 0){
                printf("%s: ga_dgemm_test() failed at line %d\n",__FILE__,__LINE__);
                fflush(stdout);
            }
        }
    } else if (test == 9){
        if(me == 0){
            printf("Running bigtest with %d processes\n",nproc);
            fflush(stdout);
        }
        status = bigtest(rank);
        if(status != 0){
            if (me == 0){
                printf("%s: bigtest() failed at line %d\n",__FILE__,__LINE__);
                fflush(stdout);
            }
        }
    }

#ifdef HPM_PROFILING
    HPM_Print();
#endif

    if ((me == 0) && (test != 0)) GA_Print_stats();

    GA_Terminate();
    MPI_Finalize();

    return(0);
}



