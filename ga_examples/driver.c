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

int test1(); // very simple test
int test2(int rank, int blksz); // matrix transpose
int test3(int rank, int blksz); // matrix multiplication
int test4(int rank, int blksz); // matrix multiplication for symmetric matrices

int main(int argc, char **argv)
{
	int me,nproc;
    int test;
    int status;

    int rank,blksz;

    MPI_Init(&argc, &argv);
    GA_Initialize();
    MA_init(MT_DBL, 128*1024*1024, 8*1024*1024);

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
        if (argc == 4){
            rank = atoi(argv[2]);
            blksz = atoi(argv[3]);
        } else {
            if(me == 0){
                printf("You need to specify rank and blksz for tests 2-4\n");
                fflush(stdout);
            }
            return(1);
        }
    }



#ifdef DEBUG
    if(me == 0){
       printf("The result of GA_Nnodes is %d\n",nproc);
       fflush(stdout);
    }
#endif

    if (test == 1){
        if(me == 0){
            printf("Running test1 with %d processes\n",nproc);
            fflush(stdout);
        }
        status = test1();
        if(status != 0){
        	if (me == 0){
                printf("%s: test1() failed at line %d\n",__FILE__,__LINE__);
                fflush(stdout);
            }
        };
    } else if (test == 2){
        if(me == 0){
            printf("Running test2 with %d processes\n",nproc);
            fflush(stdout);
        }
        status = test2(rank,blksz);
        if(status != 0){
        	if (me == 0){
                printf("%s: test2() failed at line %d\n",__FILE__,__LINE__);
                fflush(stdout);
            }
        };
    } else if (test == 3){
        if(me == 0){
            printf("Running test3 with %d processes\n",nproc);
            fflush(stdout);
        }
        status = test3(rank,blksz);
        if(status != 0){
        	if (me == 0){
                printf("%s: test3() failed at line %d\n",__FILE__,__LINE__);
                fflush(stdout);
            }
        };
    } else if (test == 4){
        if(me == 0){
            printf("Running test4 with %d processes\n",nproc);
            fflush(stdout);
        }
        status = test4(rank,blksz);
        if(status != 0){
            if (me == 0){
                printf("%s: test4() failed at line %d\n",__FILE__,__LINE__);
                fflush(stdout);
            }
        };
    }

    if (me == 0){
        printf("*************************************\n");
        printf("* driver has finished successfully! *\n");
        printf("*************************************\n");
        fflush(stdout);
    }

    if (me == 0) GA_Print_stats();

    GA_Terminate();
    MPI_Finalize();

    return(0);
}



