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

****************************************************************************/

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



