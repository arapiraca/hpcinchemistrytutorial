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

#define OUTPUT( string ) \
    if (DCMF_Messager_rank() == 0){ \
        fprintf(stdout, string ); \
    }

#define OUTPUT2( string1 , string2 ) \
    if (DCMF_Messager_rank() == 0){ \
        fprintf(stdout, string1 , string2 ); \
    }

/***************************************************************/

#ifdef HPM_PROFILING
    void HPM_Init(void);
    void HPM_Start(char *);
    void HPM_Stop(char *);
    void HPM_Print(void);
#endif

/***************************************************************/

unsigned long long getticks(void)
{
     unsigned int rx, ry, rz;
     unsigned long long r64;

     do
     {
         asm volatile ( "mftbu %0" : "=r"(rx) );
         asm volatile ( "mftb %0" : "=r"(ry) );
         asm volatile ( "mftbu %0" : "=r"(rz) );
     }
     while ( rx != rz );

     r64 = rx;
     r64 = ( r64 << 32 ) | ry;
     
     return r64;
}

/***************************************************************/




/***************************************************************/

int main(int argc, char **argv)
{

    DCMF_Result result;

#ifdef HPC_PROFILING
    HPM_Init();
    HPM_Start("all");
#endif

    DCMF_CRITICAL( DCMF_Messager_initialize(); )

    COMPILE_STAMP

    unsigned int rank = DCMF_Messager_rank();
    unsigned int size = DCMF_Messager_size();

    OUTPUT2( "DCMF_Messager_rank() = %d\n" , rank );
    OUTPUT2( "DCMF_Messager_size() = %d\n" , size );

    unsigned long int len = { ( argc > 1 ) ? ( atol(argv[1]) ) : ( 100 ) };
    OUTPUT2( "length of message = %ld\n" , len );

/***************************************************************/

    int* local_buffer;
    int* remote_buffer;

    if ( 0 != posix_memalign( (void **) &local_buffer, 32, len*sizeof(int) ) )
    {
        OUTPUT( "posix_memalign failed for local_buffer\n" );
    }

    if ( 0 != posix_memalign( (void **) &remote_buffer, 32, len*sizeof(int) ) )
    {
        OUTPUT( "posix_memalign failed for remote_buffer\n" );
    }

    DCMF_Memregion_t local_memregion;
    DCMF_Memregion_t remote_memregion;

    size_t bytes;

    result = DCMF_Memregion_create( &local_memregion, &bytes, len*sizeof(int), &local_buffer, NULL );

    switch(result)
    {
        case DCMF_SUCCESS:
            OUTPUT( "DCMF_Memregion_create successful for local_memregion\n" );
            break;

        case DCMF_EAGAIN:
            OUTPUT( "DCMF_Memregion_create failed (unavailable resource) for local_memregion\n" );
            break;

        case DCMF_INVAL:
            OUTPUT( "DCMF_Memregion_create failed (invalid parameter value) for local_memregion\n" );
            break;

        case DCMF_ERROR:
            OUTPUT( "DCMF_Memregion_create failed (memory not pinned) for local_memregion\n" );
            break;

        default:
            OUTPUT( "This should never occur\n" );
            break;
    }

/***************************************************************/







/***************************************************************/

    result = DCMF_Memregion_destroy( &local_memregion );

/***************************************************************/

    OUTPUT( "__LINE__ of __FILE__\n");
    DCMF_CRITICAL( DCMF_Messager_finalize(); )

#ifdef HPC_PROFILING
    HPM_Stop("all");
    HPM_Print();
#endif

    return(0);
}



