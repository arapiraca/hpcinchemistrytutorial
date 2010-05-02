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
    start_parallel(&argc,&argv,&me,&nproc,armci_not_ga,1);

    int status;

    if (me==0 && argc!=5){
        printf("./armci_gpu_sgemm.x <dim1> <dim2> <dim3> <tilesize>\n");
        return(1);
    }
    int a;
    if (me==0) for (a=0;a<argc;a++) printf("argv[%1d] = %s\n",a,argv[a]);

    int dim1     = ( argc>1 ? atoi(argv[1]) : 1024 );
    int dim2     = ( argc>2 ? atoi(argv[2]) : 1024 );
    int dim3     = ( argc>3 ? atoi(argv[3]) : 1024 );
    int tilesize = ( argc>4 ? atoi(argv[4]) :   64 );
    assert((tilesize>0) && (dim1>0) && (dim2>0) && (dim3>0));

    int rowc = dim1;
    int rowa = dim1;
    int cola = dim3;
    int rowb = dim3;
    int colb = dim2;
    int colc = dim2;

    float alpha = 1.0;
    float beta  = 1.0;

    /* determine tiling */
    int numtile1 = dim1/tilesize;
    int numtile2 = dim2/tilesize;
    int numtile3 = dim3/tilesize;

    int remain1 = dim1 % tilesize;
    int remain2 = dim2 % tilesize;
    int remain3 = dim3 % tilesize;
    assert((remain1==0) && (remain2==0) && (remain3==0));

    /* bookkeeping matrix dimensions */
    int sizeA = dim1 * dim3;
    int sizeB = dim3 * dim2;
    int sizeC = dim1 * dim2;
    int sizeT = tilesize * tilesize;
    if (me==0) printf("dim1 = %d\n",dim1);
    if (me==0) printf("dim2 = %d\n",dim2);
    if (me==0) printf("dim3 = %d\n",dim3);
    if (me==0) printf("tilesize = %d\n",tilesize);

    /* global lock array (GL) */
    int GLsize = ( me==0 ? dim1*dim2 : 0 );
    int** winGL = (int **) malloc( nproc * sizeof(void *) );
    ARMCI_Malloc( (void **) winGL, GLsize * sizeof(int) );
    parallel_sync();
    int i;
    for(i=0;i<GLsize;i++) winGL[me][i] = (int) 0;

    /* used in all loop code and timings of everything */
    int i1,i2,i3;
    int t1,t2,t3;
    float error;
    double start, finish;
    double loops_time  = 0.0;
    double sgemm_time  = 0.0;
    double rmw_time    = 0.0;
    double bcast_time  = 0.0;
    double reduce_time = 0.0;
    double cublas_time = 0.0;
    double push_time   = 0.0;
    double pull_time   = 0.0;

    /* allocate arrays for real computation */
    float* tilA = alloc_host_floats(sizeA);
    float* tilB = alloc_host_floats(sizeB);
    float* tilC = alloc_host_floats(sizeC);
    float* tilD = alloc_host_floats(sizeC);

    float* refA;
    float* refB;
    float* refC;
    float* refD;

    if (me==0){
        /* allocate arrays for reference computation */
        refA = alloc_host_floats(sizeA);
        refB = alloc_host_floats(sizeB);
        refC = alloc_host_floats(sizeC);
        refD = alloc_host_floats(sizeC);

        randomize_floats(sizeA, refA);
        randomize_floats(sizeB, refB);
        randomize_floats(sizeC, refC);
        copy_host_floats(sizeC, refC, refD);

        /* create tiled inputs */
        for (t3=0;t3<numtile3;t3++)
            for (t1=0;t1<numtile1;t1++)
                for (i3=0;i3<tilesize;i3++)
                    for (i1=0;i1<tilesize;i1++)
                        tilA[ (i1+i3*tilesize) + (t1+t3*numtile1)*tilesize*tilesize ] = refA[ (i1 + t1*tilesize) + (i3 + t3*tilesize)*dim1 ];
        for (t2=0;t2<numtile2;t2++)
            for (t3=0;t3<numtile3;t3++)
                for (i2=0;i2<tilesize;i2++)
                    for (i3=0;i3<tilesize;i3++)
                        tilB[ (i3+i2*tilesize) + (t3+t2*numtile3)*tilesize*tilesize ] = refB[ (i3 + t3*tilesize) + (i2 + t2*tilesize)*dim3 ];
        for (t2=0;t2<numtile2;t2++)
            for (t1=0;t1<numtile1;t1++)
                for (i2=0;i2<tilesize;i2++)
                    for (i1=0;i1<tilesize;i1++)
                        tilC[ (i1+i2*tilesize) + (t1+t2*numtile1)*tilesize*tilesize ] = refC[ (i1 + t1*tilesize) + (i2 + t2*tilesize)*dim1 ];

        /* create reference answer */
        if ((dim1<1024) && (dim2<1024) && (dim3<1024)){
            start = gettime();
            for (i1=0;i1<dim1;i1++ ){
                for (i2=0;i2<dim2;i2++ ){
                    refC[i1+i2*rowc] *= beta;
                    for (i3=0;i3<dim3;i3++ ){
                        refC[i1+i2*rowc]+=alpha*refA[i1+i3*rowa]*refB[i3+i2*rowb];
                    }
                }
            }
            finish = gettime();
            loops_time = (finish-start);
            start = gettime();
            sgemm_("n","n",&rowa,&colb,&cola,&alpha,refA,&rowa,refB,&rowb,&beta,refD,&rowc);
            finish = gettime();
            sgemm_time = (finish-start);
            printf("! time for serial loops=%10.3f seconds\n",loops_time);
            printf("! time for serial sgemm=%10.3f seconds\n",sgemm_time);
            printf("! sgemm is %6.2f times faster than loops\n",((float)loops_time)/sgemm_time);
            error=0.0;
            for (i1=0;i1<rowc;i1++ ){
                for (i2=0;i2<colc;i2++ ){
                    //printf("%4d %4d %20.14f %20.14f\n",i,j,p_c[i+j*rowc],p_d[i+j*rowc]);
                    error+=abs(refC[i1+i2*rowc]-refD[i1+i2*rowc]);
                    assert(abs(refC[i1+i2*rowc]-refD[i1+i2*rowc])<1e-6);
                }
            }
            printf("! sgemm error=%20.7f\n",error);
        }
    } // me==0
    parallel_sync();

    start = gettime();
    MPI_Bcast(tilA, sizeA, MPI_FLOAT, /* root */ 0, MPI_COMM_WORLD);
    MPI_Bcast(tilB, sizeB, MPI_FLOAT, /* root */ 0, MPI_COMM_WORLD);
    MPI_Bcast(tilC, sizeC, MPI_FLOAT, /* root */ 0, MPI_COMM_WORLD);
    finish = gettime();
    bcast_time += (finish-start);

    float* devAt = alloc_device_floats(sizeT);
    float* devBt = alloc_device_floats(sizeT);
    float* devCt = alloc_device_floats(sizeT);

    int mytasks = 0;
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
                push_floats(sizeT, /* h_ptr */ &tilC[t1+t2*numtile1], /* d_ptr */ devCt);
                finish = gettime();
                push_time += (finish-start);
                for (t3=0;t3<numtile3;t3++){
                    start = gettime();
                    push_floats(sizeT, /* h_ptr */ &tilA[t1+t3*numtile1], /* d_ptr */ devAt);
                    push_floats(sizeT, /* h_ptr */ &tilB[t3+t2*numtile3], /* d_ptr */ devBt);
                    finish = gettime();
                    push_time += (finish-start);
                    cublasSgemm('n','n',tilesize,tilesize,tilesize,alpha,devAt,tilesize,devBt,tilesize,beta,devCt,tilesize);
                    finish = gettime();
                    cublas_time += (finish-start);
                } // t3
                start = gettime();
                pull_floats(sizeT, /* h_ptr */ &tilC[t1+t2*numtile1], /* d_ptr */ devCt);
                finish = gettime();
                pull_time += (finish-start);
            } // mval==0
        } // t2
    } // t1.
    start = gettime();
    MPI_Reduce(MPI_IN_PLACE, tilC, sizeC, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    finish = gettime();
    reduce_time += (finish-start);

    if (me==0){
        for (t2=0;t2<numtile2;t2++)
            for (t1=0;t1<numtile1;t1++)
                for (i2=0;i2<tilesize;i2++)
                    for (i1=0;i1<tilesize;i1++)
                        refD[ (i1 + t1*tilesize) + (i2 + t2*tilesize)*dim1 ] = tilC[ (i1+i2*tilesize) + (t1+t2*numtile1)*tilesize*tilesize ];
        for (i1=0;i1<dim1;i1++ ){
            for (i2=0;i2<dim2;i2++ ){
                printf("%4d %4d %20.14f %20.14f\n",i1,i2,refC[i1+i2*rowc],refD[i1+i2*rowc]);
                error+=abs(refC[i1+i2*rowc]-refD[i1+i2*rowc]);
//                 assert(abs(refC[i1+i2*rowc]-refD[i1+i2*rowc])<1e-6);
            }
        }
    }





    printf("process %3d accomplished %d tasks\n",me,mytasks);
    printf("process %3d rmw_time    = %10.3f seconds\n",me,rmw_time   );
    printf("process %3d bcast_time  = %10.3f seconds\n",me,bcast_time );
    printf("process %3d reduce_time = %10.3f seconds\n",me,reduce_time);
    printf("process %3d sgemm_time  = %10.3f seconds\n",me,sgemm_time );
    printf("process %3d push_time   = %10.3f seconds\n",me,push_time  );
    printf("process %3d pull_time   = %10.3f seconds\n",me,pull_time  );

    free_device_floats(devCt);
    free_device_floats(devBt);
    free_device_floats(devAt);

    free_host_floats(tilC);
    free_host_floats(tilB);
    free_host_floats(tilA);

    if (me==0){
        free_host_floats(refD);
        free_host_floats(refC);
        free_host_floats(refB);
        free_host_floats(refA);
    }

    parallel_sync();
    stop_parallel(1);

    return(0);
}



