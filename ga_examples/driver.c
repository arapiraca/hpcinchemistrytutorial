/***************************************************************************
 *   Copyright (C) 2009 by Jeff Hammond                                    *
 *   jeff.science@gmail.com                                                *
 *                                                                         *
 * Redistribution and use in source and binary forms, with or without      *
 * modification, are permitted provided that the following conditions      *
 * are met:                                                                *
 * 1. Redistributions of source code must retain the above copyright       *
 *    notice, this list of conditions and the following disclaimer.        *
 * 2. Redistributions in binary form must reproduce the above copyright    *
 *    notice, this list of conditions and the following disclaimer in the  *
 *    documentation and/or other materials provided with the distribution. *
 * 3. The name of the author may not be used to endorse or promote         *
 *    products derived from this software without specific prior written   *
 *    permission.                                                          *
 *                                                                         *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR    *
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED          *
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE  *
 * DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,      *
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES      *
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR      *
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)      *
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,     *
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING   *
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE      *
 * POSSIBILITY OF SUCH DAMAGE.                                             *
 *                                                                         *
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

int main(int argc, char **argv)
{
	int me,nproc;
    int test;
    int status;

    int rank,blksz;

    rank = 6;
    blksz = 2;

    int desired = MPI_THREAD_MULTIPLE;
    int provided;
    MPI_Init_thread(&argc, &argv, desired, &provided);

    if ( provided != MPI_THREAD_MULTIPLE ){
      printf("provided != MPI_THREAD_MULTIPLE\n");
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

    nproc=GA_Nnodes();
    me=GA_Nodeid();

    if (argc > 1){
        test = atoi(argv[1]);
    } else {
        test = 1;
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



