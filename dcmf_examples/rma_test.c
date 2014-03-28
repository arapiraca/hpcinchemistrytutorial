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
#include "mpi.h"

/***************************************************************/

#define COMPILE_STAMP \
    if (me == 0){ \
        fprintf(stdout, "__FILE__ was compiled on __DATE__ at __TIME__ \n" ); \
    }

#define LINE_STAMP \
    if (me == 0){ \
        fprintf(stdout, "__LINE__ of __FILE__  \n" ); \
    }

#define OUTPUT( string ) \
    if (me == 0){ \
        fprintf(stdout, string ); \
    }

#define OUTPUT2( string1 , string2 ) \
    if (me == 0){ \
        fprintf(stdout, string1 , string2 ); \
    }

/***************************************************************/

#ifdef HPM_PROFILING
    void HPM_Init( void );
    void HPM_Start( char* );
    void HPM_Stop( char* );
    void HPM_Print( void );
    void HPM_Print_Flops( void );
    void HPM_Print_Flops_Agg( void );
    void HPM_Flops( char* , double* , double* );
#endif

/***************************************************************/

// possible values of rma_optype
#define RMA_GET 1
#define RMA_PUT 2
#define RMA_ACC_SUM  4
#define RMA_ACC_PROD 5
#define RMA_ACC_LXOR 6

// possible values of rma_attributes
// bit field AOBR (atomic, ordered, blocking, remote completion)
#define RMA_ATTR_NONE 0
#define RMA_ATTR_AOBR 15

/***************************************************************/

     void function(void* clientdata, DCMF_Error_t* err)
     {
         fprintf( stdout , "Put is done\n" );
         *((int*)clientdata) = 1;
     }

/***************************************************************/

    typedef struct {
        void* buffer;
        size_t size;
        DCMF_Memregion_t memregion;
    } RMA_Memregion;

    typedef struct {
        volatile int finished;
    } RMA_Request;

    void fake_MPI_RMA_Memregion_create( RMA_Memregion* mem ,
                                        void* buffer );

    void fake_MPI_RMA_Memregion_destroy( RMA_Memregion* mem );

    void fake_MPI_RMA_Memregion_exchange( RMA_Memregion* local ,
                                          RMA_Memregion* remote ,
                                          int rank );

    int datatype_check( MPI_Datatype origin_datatype , 
                        MPI_Datatype target_datatype ,
                        MPI_Comm comm , 
                        int rank );

    void fake_MPI_RMA_xfer( int rma_optype , 
                            void* origin_addr , 
                            int origin_count ,
                            MPI_Datatype origin_datatype , 
                            RMA_Memregion* target_mem , 
                            int target_disp , 
                            int target_count ,
                            MPI_Datatype target_datatype , 
                            int target_rank , 
                            MPI_Comm comm , 
                            int rma_attributes ,
                            RMA_Request* request );

    void fake_MPI_RMA_putb( void* origin_addr , 
                            int count,
                            RMA_Memregion* target_mem , 
                            int target_disp , 
                            int target_rank , 
                            MPI_Comm comm , 
                            int rma_attributes ,
                            RMA_Request* request );

    void fake_MPI_RMA_getb( void* origin_addr , 
                            int count,
                            RMA_Memregion* target_mem , 
                            int target_disp , 
                            int target_rank , 
                            MPI_Comm comm , 
                            int rma_attributes ,
                            RMA_Request* request );

/***************************************************************/

int main(int argc, char **argv)
{
    int me, nproc;
    unsigned long long t0,t1;

    MPI_Init(&argc, &argv);

    COMPILE_STAMP

#ifdef HPC_PROFILING
    HPM_Init();
    HPM_Start("all");
#endif

/***************************************************************/

    MPI_Comm_size(MPI_COMM_WORLD,&nproc);
    MPI_Comm_rank(MPI_COMM_WORLD,&me);

    OUTPUT2( "me    = %d\n" , me );
    OUTPUT2( "nproc = %d\n" , nproc );

    int version = { ( argc > 1 ) ? ( atoi(argv[1]) ) : ( 1 ) };
    if ( version == 1 )
    {
        OUTPUT("testing Vinod's rich RMA interface\n");
    }
    else if ( version == 2 )
    {
        OUTPUT("testing Jeff's minimalist RMA interface\n");
    }
    else
    {
        OUTPUT("you must choose 1 or 2!\n");
    }
 
    LINE_STAMP

/***************************************************************/

    int payload;
    payload = 1984;

    MPI_Datatype origin_datatype ;
    RMA_Memregion target_memregion;
    int target_disp ;
    MPI_Datatype target_datatype ;
    int target_rank ;
    MPI_Comm comm ;
    RMA_Request xfer_request;

    if ( version == 1 )
    {
        t0 = DCMF_Timebase();
        fake_MPI_RMA_xfer( RMA_PUT , &payload , 1 , MPI_INT , &target_memregion , 0 ,
                           1 , MPI_INT , 1 , MPI_COMM_WORLD , RMA_ATTR_NONE , 
                           &xfer_request );
        t1 = DCMF_Timebase();
    }   
    else if ( version == 2 )
    {
        t0 = DCMF_Timebase();
        t1 = DCMF_Timebase();
    }

    LINE_STAMP

/***************************************************************/

#ifdef HPC_PROFILING
    HPM_Stop("all");
    HPM_Print();
    HPM_Print_flops();
#endif

    MPI_Finalize();

    return(0);
}

/***************************************************************/

    void fake_MPI_RMA_Memregion_create( RMA_Memregion* mem ,
                                        void* buffer )
    {
        size_t bytes_out;
        RMA_Memregion local;
        local = *mem;
        local.buffer = buffer;
        DCMF_Memregion_create( &local.memregion , &bytes_out , local.size , local.buffer , 0 );
        if ( bytes_out != local.size )
        {
            fprintf( stdout , "DCMF_Memregion_create failed" );
            MPI_Abort( MPI_COMM_WORLD , 911 );
        }
    }

    void fake_MPI_RMA_Memregion_destroy( RMA_Memregion* mem )
    {
        RMA_Memregion local;
        local = *mem;        
        DCMF_Memregion_destroy(&local.memregion);
    }

    void fake_MPI_RMA_Memregion_exchange( RMA_Memregion* local ,
                                          RMA_Memregion* remote ,
                                          int rank )
    {
        MPI_Status status;
        MPI_Sendrecv(&local,  sizeof(local), MPI_CHAR, rank, 0,
                     &remote, sizeof(remote), MPI_CHAR, rank, 0,
                     MPI_COMM_WORLD, &status);
    }

    int datatype_check( MPI_Datatype origin_datatype ,
                        MPI_Datatype target_datatype ,
                        MPI_Comm comm , 
                        int remote_rank)
    {
        fprintf( stdout , "datatype_check not implemented" );
        MPI_Abort( comm , 911 );
        return(0);
    }

/***************************************************************/

    void fake_MPI_RMA_xfer( int rma_optype , 
                            void* origin_addr , 
                            int origin_count ,
                            MPI_Datatype origin_datatype , 
                            RMA_Memregion* target_mem , 
                            int target_disp , 
                            int target_count ,
                            MPI_Datatype target_datatype , 
                            int target_rank , 
                            MPI_Comm comm , 
                            int rma_attributes ,
                            RMA_Request* request )
    {


        int result;
        if ( origin_datatype != target_datatype && target_datatype != MPI_DATATYPE_NULL )
        {
            result = datatype_check( origin_datatype , target_datatype , comm , target_rank );
            if ( result != 0 )
            {
                fprintf( stdout , "origin_datatype does not match target_datatype" );
                MPI_Abort( comm , 911 );
            }
        }

        int dcmf_attribute;
        if ( rma_attributes == RMA_ATTR_NONE )
        {
            dcmf_attribute = DCMF_RELAXED_CONSISTENCY;
        }
        else if ( rma_attributes == RMA_ATTR_AOBR )
        {
            dcmf_attribute = DCMF_SEQUENTIAL_CONSISTENCY;
        }
        else
        {
            fprintf( stdout , "invalid rma_attributes\n" );
            MPI_Abort( comm , 911 );
        }

        DCMF_Protocol_t dcmf_protocol;
        DCMF_Put_Configuration_t dcmf_put_config;
        DCMF_Request_t dcmf_request;
        DCMF_Callback_t dcmf_cb_done;

        volatile int done = 0;
        dcmf_cb_done.function   = function;
        dcmf_cb_done.clientdata = (void*)&done;
        // I am too stupid with structs to map this cb to RMA_Request

        switch(rma_optype)
        {
            case RMA_PUT:
                dcmf_put_config.protocol = DCMF_DEFAULT_PUT_PROTOCOL;
                DCMF_Put_register(&dcmf_protocol, &dcmf_put_config);
                RMA_Memregion origin_mem;
                fake_MPI_RMA_Memregion_create( &origin_mem , origin_addr );
               // DCMF_Put( &dcmf_protocol , &dcmf_request , done , dcmf_attribute , target_rank , origin_mem.size ,
               //           &origin_mem.memregion, &( (*target_mem).memregion ) , 0 , sizeof(int) , (DCMF_Callback_t){NULL,NULL} );
                break;

            case RMA_GET:
                fprintf( stdout , "RMA_GET not implemented" );
                MPI_Abort( comm , 911 );
                break;

            case RMA_ACC_SUM:
                fprintf( stdout , "RMA_ACC_SUM not implemented" );
                MPI_Abort( comm , 911 );
                break;

            case RMA_ACC_PROD:
                fprintf( stdout , "RMA_ACC_PROD not implemented" );
                MPI_Abort( comm , 911 );
                break;

            case RMA_ACC_LXOR:
                fprintf( stdout , "RMA_ACC_LXOR not implemented" );
                MPI_Abort( comm , 911 );
                break;

            default:
                fprintf( stdout , "Invalid rma_optype" );
                MPI_Abort( comm , 911 );
                break;
        }
        

    }

/***************************************************************/

    void fake_MPI_RMA_putb( void* origin_addr , 
                            int count,
                            RMA_Memregion* target_mem , 
                            int target_disp , 
                            int target_rank , 
                            MPI_Comm comm , 
                            int rma_attributes ,
                            RMA_Request* request )
    {

    }

