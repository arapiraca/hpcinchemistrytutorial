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
    int armci_not_ga = 0;
    start_parallel(&argc,&argv,&me,&nproc,armci_not_ga);

    int status;

    if (me==0 && argc!=3) printf("./ga_gpu_sgemm.x <rank> <blksz>\n");
    if (me==0) for (int a=0;a<argc;a++) printf("argv[%1d] = %s\n",a,argv[a]);

    int rank  = ( argc>1 ? atoi(argv[1]) : 64 );
    int blksz = ( argc>2 ? atoi(argv[2]) :  4 );

    int ntask,t;
    int ii,jj,kk;
    int i,j,k;
    int g_a,g_b,g_c,g_d1,g_d2,g_error; // GA handles
    int ndim = 2;
    int dims[2];
    int chunk[2];
    int nblock;
    int lo_a[2],lo_b[2],lo_d[2];
    int hi_a[2],hi_b[2],hi_d[2];
    int rng_a[2],rng_b[2];//,rng_c[2];
    int ld_a[1],ld_b[1],ld_d[1];
    int pg_world;   // world processor group
    float alpha,beta;
    double error;
    double start,finish;
    float zero = 0.0;
    float one  = 1.0;
    float temp;
    float* p_in; // pointers for local access to GAs
    float* p_a;  // pointers for local access to GAs
    float* p_b;  // pointers for local access to GAs
    float* p_d;  // pointers for local access to GAs
    float* dp_a;  // device pointer
    float* dp_b;  // device pointer
    float* dp_d;  // device pointer

    if (me==0) printf("matmul2: rank %d matrix with block size %d\n",rank,blksz);

    dims[0] = rank;
    dims[1] = rank;
    chunk[0] = blksz;
    chunk[1] = blksz;
    nblock = rank/blksz;
    g_a = GA_Create_handle();
    GA_Set_array_name(g_a,"matrix A");
    GA_Set_data(g_a,ndim,dims,MT_REAL);
    GA_Set_chunk(g_a,chunk);
    pg_world = GA_Pgroup_get_world();
    GA_Set_pgroup(g_a,pg_world);

    status = GA_Allocate(g_a); assert(status!=0);
    g_b  = GA_Duplicate(g_a,"matrix B"); assert(g_b!=0);
    g_c  = GA_Duplicate(g_a,"matrix C"); assert(g_c!=0);
    g_d1  = GA_Duplicate(g_a,"matrix D1"); assert(g_d1!=0);
    g_d2  = GA_Duplicate(g_a,"matrix D2"); assert(g_d2!=0);
    g_error  = GA_Duplicate(g_a,"error"); assert(g_error!=0);
    GA_Sync();

//     float scale = 1.0/RAND_MAX;
//     NGA_Distribution(g_a,me,lo_a,hi_a);
//     NGA_Access(g_a,lo_a,hi_a,&p_in,&ld_a[0]);
//     rng_a[0] = hi_a[0] - lo_a[0] + 1;
//     rng_a[1] = hi_a[1] - lo_a[1] + 1;
//     for(i=0; i<rng_a[0]; i++){
//         for(j=0; j<rng_a[1]; j++){
//             p_in[ ld_a[0] * i + j ] = (float) ( rand() * scale );
// //          p_in[ ld_a[0] * i + j ] = (float) ( 1 );
//         }
//     }
//     NGA_Release_update(g_b,lo_a,hi_a); /* this function does nothing as of GA 4.2 */
// //     GA_Symmetrize(g_b); // Only for doubles
    GA_Randomize(g_a,&one);
#ifdef COMPARE
    GA_Transpose(g_a,g_d1);
    alpha = 0.5;
    beta  = 0.5;
    GA_Zero(g_error);
    GA_Add(&alpha,g_a,&beta,g_d1,g_error);
    GA_Copy(g_error,g_a);
#endif

//     NGA_Distribution(g_b,me,lo_b,hi_b);
//     NGA_Access(g_b,lo_b,hi_b,&p_in,&ld_b[0]);
//     rng_b[0] = hi_b[0] - lo_b[0] + 1;
//     rng_b[1] = hi_b[1] - lo_b[1] + 1;
//     for(i=0; i<rng_b[0]; i++){
//         for(j=0; j<rng_b[1]; j++){
//                         p_in[ ld_a[0] * i + j ] = (float) ( rand() * scale );
//         }
//     }
//     NGA_Release_update(g_b,lo_b,hi_b); /* this function does nothing as of GA 4.2 */
// //     GA_Symmetrize(g_b); // Only for doubles
    GA_Randomize(g_b,&one);
#ifdef COMPARE
    GA_Transpose(g_b,g_d1);
    alpha = 0.5;
    beta  = 0.5;
    GA_Zero(g_error);
    GA_Add(&alpha,g_b,&beta,g_d1,g_error);
    GA_Copy(g_error,g_b);
#endif

    if (rank<40){
        GA_Print(g_a);
        GA_Print(g_b);
    }

    alpha = 1.0;
    beta  = 0.0;

    unsigned long long nflops = 0;
    if      (alpha==0.0){  nflops += 0; }
    else if (alpha==1.0){  nflops += rank*rank*rank; }
    else                {  nflops += 2*rank*rank*rank; }

    if      (beta==0.0){  nflops += 0; }
    else if (beta==1.0){  nflops += rank*rank; }
    else               {  nflops += 2*rank*rank; }

    double t_ga;
    double gflops;

    GA_Zero(g_d1);
    ntask = nblock * nblock * nblock;
#ifdef DEBUG
    if (me == 0) {
        printf("ntask = %d\n",ntask);
        printf("nproc = %d\n",nproc);
    }
#endif

//     p_a = (float *)ARMCI_Malloc_local((armci_size_t) blksz * blksz * sizeof(float)); assert(p_a!=NULL);
//     p_b = (float *)ARMCI_Malloc_local((armci_size_t) blksz * blksz * sizeof(float)); assert(p_b!=NULL);
//     p_d = (float *)ARMCI_Malloc_local((armci_size_t) blksz * blksz * sizeof(float)); assert(p_d!=NULL);
//     p_a = alloc_host_floats(blksz*blksz);
//     p_b = alloc_host_floats(blksz*blksz);
//     p_d = alloc_host_floats(blksz*blksz);
    cudaError_t cuStatus;
    cuStatus = cudaMallocHost((void**)&p_a, blksz*blksz*sizeof(float) ); assert(cuStatus==CUDA_SUCCESS);
    cuStatus = cudaMallocHost((void**)&p_b, blksz*blksz*sizeof(float) ); assert(cuStatus==CUDA_SUCCESS);
    cuStatus = cudaMallocHost((void**)&p_d, blksz*blksz*sizeof(float) ); assert(cuStatus==CUDA_SUCCESS);
    cuStatus = cudaMalloc((void**)&dp_a, blksz*blksz*sizeof(float) ); assert(cuStatus==CUDA_SUCCESS);
    cuStatus = cudaMalloc((void**)&dp_b, blksz*blksz*sizeof(float) ); assert(cuStatus==CUDA_SUCCESS);
    cuStatus = cudaMalloc((void**)&dp_d, blksz*blksz*sizeof(float) ); assert(cuStatus==CUDA_SUCCESS);
//     size_t alignment = getpagesize();
//     status = posix_memalign((void**)&p_a, alignment, blksz*blksz*sizeof(float));
//     status = posix_memalign((void**)&p_b, alignment, blksz*blksz*sizeof(float));
//     status = posix_memalign((void**)&p_d, alignment, blksz*blksz*sizeof(float));

    double t_start, t_end, t_total;
    double t_get = 0.0;
    double t_acc = 0.0;
    double t_zero = 0.0;
    double t_push = 0.0;
    double t_pull = 0.0;
    double t_sgemm = 0.0;

    t = 0;
    GA_Sync();
    t_start = gettime();
    for (ii = 0 ; ii < nblock; ii++){
        for (jj = 0 ; jj < nblock; jj++){
            if (me==(t%nproc)){
                //for (k=0;k<(blksz*blksz);k++) p_d[k]=0.0;
                start = gettime();
                cuStatus = cudaMemset(dp_d,0,blksz*blksz*sizeof(float)); assert(cuStatus==CUDA_SUCCESS);
                finish = gettime();
                t_zero += finish-start;
                for (kk = 0 ; kk < nblock; kk++){
#ifdef DEBUG
                    printf("proc %d doing work tuple (%d,%d,%d)\n",me,ii,jj,kk); fflush(stdout);
#endif
                    lo_a[0] = blksz * ii;
                    hi_a[0] = blksz * (ii + 1) - 1;
                    lo_a[1] = blksz * kk;
                    hi_a[1] = blksz * (kk + 1) - 1;
                    ld_a[0] = blksz;

                    lo_b[0] = blksz * kk;
                    hi_b[0] = blksz * (kk + 1) - 1;
                    lo_b[1] = blksz * jj;
                    hi_b[1] = blksz * (jj + 1) - 1;
                    ld_b[0] = blksz;

                    lo_d[0] = blksz * ii;
                    hi_d[0] = blksz * (ii + 1) - 1;
                    lo_d[1] = blksz * jj;
                    hi_d[1] = blksz * (jj + 1) - 1;
                    ld_d[0] = blksz;

                    start = gettime();
                    NGA_Get(g_a,lo_a,hi_a,p_a,ld_a);
                    finish = gettime();
                    t_get += finish-start;

                    cuStatus = cudaThreadSynchronize(); assert(cuStatus==CUDA_SUCCESS);
                    start = gettime();
                    cuStatus = cudaMemcpy(dp_a, p_a, blksz*blksz*sizeof(float), cudaMemcpyHostToDevice); assert(cuStatus==CUDA_SUCCESS);
                    cuStatus = cudaThreadSynchronize(); assert(cuStatus==CUDA_SUCCESS);
                    finish = gettime();
                    t_push += finish-start;

                    start = gettime();
                    NGA_Get(g_b,lo_b,hi_b,p_b,ld_b);
                    finish = gettime();
                    t_get += finish-start;

                    cuStatus = cudaThreadSynchronize(); assert(cuStatus==CUDA_SUCCESS);
                    start = gettime();
                    cuStatus = cudaMemcpy(dp_b, p_b, blksz*blksz*sizeof(float), cudaMemcpyHostToDevice); assert(cuStatus==CUDA_SUCCESS);
                    cuStatus = cudaThreadSynchronize(); assert(cuStatus==CUDA_SUCCESS);
                    finish = gettime();
                    t_push += finish-start;

                    //start = gettime();
                    //sgemm_("n","n",&blksz,&blksz,&blksz,&alpha,p_b,&blksz,p_a,&blksz,&one,p_d,&blksz);
                    //finish = gettime();
                    //t_sgemm = finish-start;
                    cuStatus = cudaThreadSynchronize(); assert(cuStatus==CUDA_SUCCESS);
                    start = gettime();
                    cublasSgemm('n','n',blksz,blksz,blksz,alpha,dp_b,blksz,dp_a,blksz,one,dp_d,blksz);
                    cuStatus = cudaThreadSynchronize(); assert(cuStatus==CUDA_SUCCESS);
                    finish = gettime();
                    t_sgemm += finish-start;
                } // kk
                cuStatus = cudaThreadSynchronize(); assert(cuStatus==CUDA_SUCCESS);
                start = gettime();
                cuStatus = cudaMemcpy(p_d, dp_d, blksz*blksz*sizeof(float), cudaMemcpyDeviceToHost); assert(cuStatus==CUDA_SUCCESS);
                cuStatus = cudaThreadSynchronize(); assert(cuStatus==CUDA_SUCCESS);
                finish = gettime();
                t_pull += finish-start;

                start = gettime();
                NGA_Acc(g_d1,lo_d,hi_d,p_d,ld_d,&one);
                finish = gettime();
                t_acc += finish-start;
            } // myturn
            t += 1;
        } // jj
    } // ii
    GA_Sync();
    t_end = gettime();
    t_total = t_end - t_start;
    gflops = nflops/t_total;
    gflops /= 1024;
    gflops /= 1024;
    gflops /= 1024;
    if (me==0){
        printf("performance         = %12.6f gflops\n",gflops);
        printf("time for everything = %12.6f seconds\n",t_total);
        printf("time for GA_Get     = %12.6f seconds\n",t_get);
        printf("time for GA_Acc     = %12.6f seconds\n",t_acc);
        printf("time for Push       = %12.6f seconds\n",t_push);
        printf("time for Pull       = %12.6f seconds\n",t_pull);
        printf("time for Sgemm      = %12.6f seconds\n",t_sgemm);
        printf("time for Memset     = %12.6f seconds\n",t_zero);
    }

//     status = ARMCI_Free_local(p_d); assert(status==0);
//     status = ARMCI_Free_local(p_b); assert(status==0);
//     status = ARMCI_Free_local(p_a); assert(status==0);
//     free_host_floats(p_d);
//     free_host_floats(p_b);
//     free_host_floats(p_a);
    cuStatus = cudaFree(dp_d); assert(cuStatus==CUDA_SUCCESS);
    cuStatus = cudaFree(dp_b); assert(cuStatus==CUDA_SUCCESS);
    cuStatus = cudaFree(dp_a); assert(cuStatus==CUDA_SUCCESS);
    cuStatus = cudaFreeHost(p_d); assert(cuStatus==CUDA_SUCCESS);
    cuStatus = cudaFreeHost(p_b); assert(cuStatus==CUDA_SUCCESS);
    cuStatus = cudaFreeHost(p_a); assert(cuStatus==CUDA_SUCCESS);
//     free(p_d);
//     free(p_b);
//     free(p_a);

#ifdef COMPARE
    //GA_Sgemm('N','N',dims[0],dims[0],dims[0],alpha,g_a,g_b,beta,g_c);
    GA_Sync();
    start = gettime();
    GA_Sgemm('N','N',dims[0],dims[0],dims[0],alpha,g_a,g_b,beta,g_c);
    GA_Sync();
    finish = gettime();
    t_ga =  finish-start;
    gflops = nflops/t_ga;
    gflops /= 1024;
    gflops /= 1024;
    gflops /= 1024;
    if (me == 0) printf("GA_Sgemm took %f seconds %f gflops \n",t_ga,gflops);

    GA_Transpose(g_d1,g_d2);
    alpha = 1.0;
    beta = -1.0;
    GA_Add(&alpha,g_c,&beta,g_d2,g_error);
    if (rank<40){
        GA_Print(g_c);
        GA_Print(g_d2);
    }

    GA_Norm1(g_error,&error);
    if (me == 0) printf("error = %f\n",error);
#endif

    GA_Destroy(g_error);
    GA_Destroy(g_d2);
    GA_Destroy(g_d1);
    GA_Destroy(g_c);
    GA_Destroy(g_b);
    GA_Destroy(g_a);

    parallel_sync();
    stop_parallel();
    return(0);
}
