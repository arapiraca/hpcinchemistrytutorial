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

int test1();

int main(int argc, char **argv)
{
    int test;
    int status;

    MPI_Init(&argc, &argv);
    GA_Initialize();
    MA_init(MT_DBL, 128*1024*1024, 8*1024*1024);

    nproc=GA_Nnodes();
    me=GA_Nodeid();

#ifdef DEBUG
    if(me==0){ 
       printf("The result of GA_Nnodes is %d\n",nproc);
       fflush(stdout);
    }
#endif

    if (argc > 1){
        test = atoi(argv[1]);
    } else {
        test =1;
    }
    if (me==0){
        printf("Running test %d\n",test);
        fflush(stdout);
    }

    if (test == 1){
        status = test1();
        if(0 != status){};
    } else if (test == 2){
        status = test2();
        if(0 != status){};
    }

    if (me==0){
        printf("*************************************\n");
        printf("* driver has finished successfully! *\n");
        printf("*************************************\n");
        fflush(stdout);
    }

    if (me==0) GA_Print_stats();

    GA_Terminate();
    MPI_Finalize();

    return(0);
}



