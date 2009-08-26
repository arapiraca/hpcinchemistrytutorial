/***************************************************************************
 *   Copyright (C) 2009 by Jeff Hammond                                    *
 *   jeff.science@gmail.com                                                *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#include "driver.h"

int simple_get(int me, int nproc, int len);
int simple_put(int me, int nproc, int len);
int overlap_b(int me, int nproc, int len);
int overlap_nb(int me, int nproc, int len);

int main(int argc, char **argv)
{
	int me,nproc;
    int test;
    int status;

    MPI_Init(&argc, &argv);
    ARMCI_Init();

#ifdef HPC_PROFILING
    HPM_Init();
#endif

    MPI_Comm_rank(MPI_COMM_WORLD,&me);
    MPI_Comm_size(MPI_COMM_WORLD,&nproc);

    if (argc > 1){
        test = atoi(argv[1]);
    } else {
        test = 1;
    }
    if (me == 0){
        printf("Running test %d\n",test);
        fflush(stdout);
    }

#ifdef DEBUG
    if(me == 0){
       printf("The result of MPI_Comm_size is %d\n",nproc);
       fflush(stdout);
    }
#endif

    if (test == 1){

        if(nproc%2 != 0){
            if (me == 0){
                printf("You need to use an even number of processes\n");
                fflush(stdout);
                ARMCI_Cleanup();
                MPI_Abort(MPI_COMM_WORLD,test);
            }
        }

        int len;
        if (argc > 2){
            len = atoi(argv[2]);
        } else {
            len = 1000;
        }

        if(me == 0){
            printf("Running simple_get with nproc = %d and len = %d\n",nproc,len);
            fflush(stdout);
        }

        status = simple_get(me,nproc,len);
        if(status != 0){
            if (me == 0){
                printf("%s: simple_get() failed at line %d\n",__FILE__,__LINE__);
                fflush(stdout);
                ARMCI_Cleanup();
                MPI_Abort(MPI_COMM_WORLD,status);
            }
        }
    } else if (test == 2){

        if(nproc%2 != 0){
            if (me == 0){
                printf("You need to use an even number of processes\n");
                fflush(stdout);
                ARMCI_Cleanup();
                MPI_Abort(MPI_COMM_WORLD,test);
            }
        }

        int len;
        if (argc > 2){
            len = atoi(argv[2]);
        } else {
            len = 1000;
        }

        if(me == 0){
            printf("Running simple_put with nproc = %d and len = %d\n",nproc,len);
            fflush(stdout);
        }

        status = simple_put(me,nproc,len);
        if(status != 0){
            if (me == 0){
                printf("%s: simple_put() failed at line %d\n",__FILE__,__LINE__);
                fflush(stdout);
                ARMCI_Cleanup();
                MPI_Abort(MPI_COMM_WORLD,status);
            }
        }
    } else if (test == 3){

        if(nproc%2 != 0){
            if (me == 0){
                printf("You need to use more than one process\n");
                fflush(stdout);
                ARMCI_Cleanup();
                MPI_Abort(MPI_COMM_WORLD,test);
            }
        }

        int len;
        if (argc > 2){
            len = atoi(argv[2]);
        } else {
            len = 1000;
        }

        if(me == 0){
            printf("BLOCKING Running overlap_b with nproc = %d and len = %d\n",nproc,len);
            fflush(stdout);
        }

        status = overlap_b(me,nproc,len);
        if(status != 0){
            if (me == 0){
                printf("%s: overlap_b() failed at line %d\n",__FILE__,__LINE__);
                fflush(stdout);
                ARMCI_Cleanup();
                MPI_Abort(MPI_COMM_WORLD,status);
            }
        }
    } else if (test == 4){

        if(nproc < 2){
            if (me == 0){
                printf("You need to use more than one process\n");
                fflush(stdout);
                ARMCI_Cleanup();
                MPI_Abort(MPI_COMM_WORLD,test);
            }
        }

        int len;
        if (argc > 2){
            len = atoi(argv[2]);
        } else {
            len = 1000;
        }

        if(me == 0){
            printf("NONBLOCK Running overlap_nb with nproc = %d and len = %d\n",nproc,len);
            fflush(stdout);
        }

        status = overlap_nb(me,nproc,len);
        if(status != 0){
            if (me == 0){
                printf("%s: overlap_nb() failed at line %d\n",__FILE__,__LINE__);
                fflush(stdout);
                ARMCI_Cleanup();
                MPI_Abort(MPI_COMM_WORLD,status);
            }
        }
    } else if (test == 5){

        if(nproc%2 != 0){
            if (me == 0){
                printf("You need to use an even number of processes\n");
                fflush(stdout);
                ARMCI_Cleanup();
                MPI_Abort(MPI_COMM_WORLD,test);
            }
        }

        int len;
        if (argc > 2){
            len = atoi(argv[2]);
        } else {
            len = 1000;
        }

        if(me == 0){
            printf("BLOCKING Running overlap_b_ring with nproc = %d and len = %d\n",nproc,len);
            fflush(stdout);
        }

        status = overlap_b_ring(me,nproc,len);
        if(status != 0){
            if (me == 0){
                printf("%s: overlap_b_ring() failed at line %d\n",__FILE__,__LINE__);
                fflush(stdout);
                ARMCI_Cleanup();
                MPI_Abort(MPI_COMM_WORLD,status);
            }
        }
    } else if (test == 6){

        if(nproc%2 != 0){
            if (me == 0){
                printf("You need to use an even number of processes\n");
                fflush(stdout);
                ARMCI_Cleanup();
                MPI_Abort(MPI_COMM_WORLD,test);
            }
        }

        int len;
        if (argc > 2){
            len = atoi(argv[2]);
        } else {
            len = 1000;
        }

        if(me == 0){
            printf("NONBLOCK Running overlap_nb_ring with nproc = %d and len = %d\n",nproc,len);
            fflush(stdout);
        }

        status = overlap_nb_ring(me,nproc,len);
        if(status != 0){
            if (me == 0){
                printf("%s: overlap_nb_ring() failed at line %d\n",__FILE__,__LINE__);
                fflush(stdout);
                ARMCI_Cleanup();
                MPI_Abort(MPI_COMM_WORLD,status);
            }
        }
    } else {
        if(me == 0){
            printf("Invalid test number (%d) requested\n",test);
            fflush(stdout);
            ARMCI_Cleanup();
            MPI_Abort(MPI_COMM_WORLD,911);
        }
    }

#ifdef HPC_PROFILING
    HPM_Print();
#endif

    ARMCI_Finalize();
    MPI_Finalize();

    return(0);
}



