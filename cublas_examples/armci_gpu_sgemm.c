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

#include "ga_utils.h"
#include "blas_gemm_test.h"
#include "cublas_gemm_test.h"

int main(int argc, char **argv)
{
    int me, nproc;
    int armci_not_ga = 1;
    start_parallel(&argc,&argv,&me,&nproc,armci_not_ga);

    int status;

    if (me==0 && argc!=4) printf("./armci_gpu_sgemm.x <tilesize> <numtile1> <numtile2> <numtile3>\n");
    if (me==0) for (int a=0;a<argc;a++) printf("argv[%1d] = %s\n",a,argv[a]);

    int tilesize = ( argc>1 ? atoi(argv[1]) : 64 );
    int numtile1 = ( argc>2 ? atoi(argv[2]) :  4 );
    int numtile2 = ( argc>3 ? atoi(argv[3]) :  4 );
    int numtile3 = ( argc>4 ? atoi(argv[4]) :  4 );
    assert((tilesize>0) && (numtile1>0) && (numtile2>0) && (numtile3>0));

    /* scale by the tile dimensions */
    int dim1 = numtile1 * tilesize;
    int dim2 = numtile2 * tilesize;
    int dim3 = numtile3 * tilesize;

    /* bookkeeping matrix dimensions */
    int sizeA = dim1 * dim3;
    int sizeB = dim3 * dim2;
    int sizeC = dim1 * dim2;
    int sizeT = tilesize * tilesize;
    if (me==0) printf("dim1 = %d\n",dim1);
    if (me==0) printf("dim2 = %d\n",dim2);
    if (me==0) printf("dim3 = %d\n",dim3);
    if (me==0) printf("tilesize = %d\n",tilesize);

    /* register remote pointers */
    float** winA = (float **) malloc( nproc * sizeof(void *) );
    float** winB = (float **) malloc( nproc * sizeof(void *) );
    float** winC = (float **) malloc( nproc * sizeof(void *) );
    ARMCI_Malloc( (void **) winA, sizeA * sizeof(float) );
    ARMCI_Malloc( (void **) winB, sizeB * sizeof(float) );
    ARMCI_Malloc( (void **) winC, sizeC * sizeof(float) );
    parallel_sync();

    /* global lock array (GL) */
    int GLsize = ( me==0 ? dim1*dim2 : 0 );
    int** winGL = (int **) malloc( nproc * sizeof(void *) );
    ARMCI_Malloc( (void **) winGL, GLsize * sizeof(int) );
    parallel_sync();
    for(int i=0;i<GLsize;i++) winGL[me][i] = (int) 0;

    float* bufA = alloc_host_floats(sizeA);
    float* bufB = alloc_host_floats(sizeB);
    float* bufC = alloc_host_floats(sizeC);
    float* bufD = alloc_host_floats(sizeC);

    if (me==0) randomize_floats(sizeA, bufA);
    if (me==0) randomize_floats(sizeB, bufB);
    if (me==0) randomize_floats(sizeC, bufC);
    if (me==0) copy_host_floats(sizeC, bufC, bufD);

    double start, finish;
    double rmw_time    = 0.0;
    double bcast_time  = 0.0;
    double reduce_time = 0.0;
    double sgemm_time  = 0.0;
    double push_time   = 0.0;
    double pull_time   = 0.0;

    start = gettime();
    MPI_Bcast(bufA, sizeA, MPI_FLOAT, /* root */ 0, MPI_COMM_WORLD);
    MPI_Bcast(bufB, sizeB, MPI_FLOAT, /* root */ 0, MPI_COMM_WORLD);
    MPI_Bcast(bufC, sizeC, MPI_FLOAT, /* root */ 0, MPI_COMM_WORLD);
    finish = gettime();
    bcast_time += (finish-start);
    MPI_Bcast(bufD, sizeC, MPI_FLOAT, /* root */ 0, MPI_COMM_WORLD);

    float* devAt = alloc_device_floats(sizeT);
    float* devBt = alloc_device_floats(sizeT);
    float* devCt = alloc_device_floats(sizeT);

    int mytasks = 0;
    int t1,t2,t3;
    float alpha = 1.0;
    const float beta  = 1.0;
    for (t1=0;t1<numtile1;t1++){
        for (t2=0;t2<numtile2;t2++){
            int oval, mval;
            start = gettime();
            oval = ARMCI_Rmw(ARMCI_FETCH_AND_ADD, &mval, &winGL[0][t1+t2*numtile1], /* incr */ 1, /* rank */ 0);
            finish = gettime();
            rmw_time += (finish-start);
            //printf("%d: t1 = %2d t2 = %2d oval = %1d mval = %1d\n",me,t1,t2,oval,mval);
            if (mval==0){
                mytasks++;
                printf("process %3d has grabbed task (%3d,%3d)\n",me,t1,t2);
                start = gettime();
                push_floats(sizeT, /* h_ptr */ &bufC[t1+t2*numtile1], /* d_ptr */ devCt);
                finish = gettime();
                push_time += (finish-start);
                for (t3=0;t3<numtile3;t3++){
                    start = gettime();
                    push_floats(sizeT, /* h_ptr */ &bufA[t1+t3*numtile1], /* d_ptr */ devAt);
                    push_floats(sizeT, /* h_ptr */ &bufB[t3+t2*numtile3], /* d_ptr */ devBt);
                    finish = gettime();
                    push_time += (finish-start);
                    cublasSgemm('n','n',tilesize,tilesize,tilesize,alpha,devAt,tilesize,devBt,tilesize,beta,devCt,tilesize);
                    finish = gettime();
                    sgemm_time += (finish-start);
#ifdef VERIFY
                    for (int i=0;i<tilesize;i++){
                        for (int j=0;j<tilesize;j++){
                            for (int k=0;k<tilesize;k++){
                                bufD[t1+t2*numtile1 + i+j*tilesize] += alpha*
                                    bufA[t1+t3*numtile1 + i+k*tilesize] *
                                    bufB[t3+t2*numtile3 + k+j*tilesize];
                            } // k
                        } // j
                    } // i
#endif
                } // t3
                start = gettime();
                pull_floats(sizeT, /* h_ptr */ &bufC[t1+t2*numtile1], /* d_ptr */ devCt);
                finish = gettime();
                pull_time += (finish-start);
#ifdef VERIFY
                for (int i=0;i<tilesize;i++){
                    for (int j=0;j<tilesize;j++){
    //                     printf("%4d %4d %15.7f %15.7f\n",i,j,
    //                            bufC[t1+t2*numtile1 + i+j*tilesize],
    //                            bufD[t1+t2*numtile1 + i+j*tilesize]);
                        assert(abs(bufC[t1+t2*numtile1 + i+j*tilesize]-
                                bufD[t1+t2*numtile1 + i+j*tilesize])<1e-5);
                    } // j
                } // i
#endif
            } // mval==0
        } // t2
    } // t1.
    start = gettime();
    MPI_Allreduce(MPI_IN_PLACE, bufC, sizeC, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    finish = gettime();
    reduce_time += (finish-start);
    printf("process %3d accomplished %d tasks\n",me,mytasks);
    printf("process %3d rmw_time    = %10.3f\n",me,rmw_time   );
    printf("process %3d bcast_time  = %10.3f\n",me,bcast_time );
    printf("process %3d reduce_time = %10.3f\n",me,reduce_time);
    printf("process %3d sgemm_time  = %10.3f\n",me,sgemm_time );
    printf("process %3d push_time   = %10.3f\n",me,push_time  );
    printf("process %3d pull_time   = %10.3f\n",me,pull_time  );

    free_device_floats(devCt);
    free_device_floats(devBt);
    free_device_floats(devAt);

    free_host_floats(bufC);
    free_host_floats(bufB);
    free_host_floats(bufA);

//     status = ARMCI_Free(winC[me]); assert(status==0);
//     status = ARMCI_Free(winB[me]); assert(status==0);
//     status = ARMCI_Free(winA[me]); assert(status==0);

    parallel_sync();
    stop_parallel();

    return(0);
}



