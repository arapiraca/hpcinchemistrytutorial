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
    void HPM_Print_flops(void);
#endif

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



