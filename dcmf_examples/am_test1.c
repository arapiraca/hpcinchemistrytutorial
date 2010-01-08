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

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "dcmf.h"

#define DCMF_CRITICAL( call ) \
    { \
    DCMF_CriticalSection_enter(0); \
    call \
    DCMF_CriticalSection_exit(0); \
    }

#define COMPILE_STAMP \
    if (DCMF_Messager_rank() == 0){ \
        fprintf(stdout, "__FILE__ was compiled on __DATE__ at __TIME__ \n" ); \
    }

/***************************************************************/

#ifdef HPM_PROFILING
    void HPM_Init(void);
    void HPM_Start(char *);
    void HPM_Stop(char *);
    void HPM_Print(void);
    void HPM_Print_flops(void);
#endif

/***************************************************************/

int main(int argc, char **argv)
{
#ifdef HPC_PROFILING
    HPM_Init();
    HPM_Start("all");
#endif

    DCMF_CRITICAL( DCMF_Messager_initialize(); )

    COMPILE_STAMP

    size_t rank = DCMF_Messager_rank();
    if (rank==0) fprintf(stdout, "DCMF_Messager_rank() = %d\n" , rank );

    size_t size = DCMF_Messager_size();
    if (rank==0) fprintf(stdout, "DCMF_Messager_size() = %d\n" , size );

/***************************************************************/

    DCMF_Protocol_t* registration;
    DCMF_Send_Configuration_t* configuration;
    result = DCMF_Send_register( registration,
                                 configuration );

    DCMF_Consistency consistency = DCMF_RELAXED_CONSISTENCY;
    DCMF_Result      result;
    DCMF_Request_t*  request;
    DCMF_Callback_t  cb_done;
    DCQuad*          msginfo;

    size_t target = 1;
    size_t bytes  = 0;
    size_t count  = 0;
    char*  src    = NULL;

    if (rank==0)
    {
        result = DCMF_Send( registration, request, cb_done, consistency,
                            target, bytes, src, msginfo, count );   
    }

/***************************************************************/

    OUTPUT( "__LINE__ of __FILE__\n");
    DCMF_CRITICAL( DCMF_Messager_finalize(); )

#ifdef HPC_PROFILING
    HPM_Stop("all");
    HPM_Print();
#endif

    return(0);
}



