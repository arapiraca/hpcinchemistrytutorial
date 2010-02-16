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

#include "ga_utils.h"
#include "blas_utils.h"
#include "cublas_utils.h"

void sgemm_(char* , char* ,int* , int* , int* , float* , float* , int* , float* , int* , float* , float* , int* );
void dgemm_(char* , char* ,int* , int* , int* , double* , double* , int* , double* , int* , double* , double* , int* );

void gagpu_sgemm(int rows, int cols, int blksz, float alpha, int g_a, int g_b, float beta, int g_d)
{
    int ndim = 2;
    int dims[ndim];

    const float one = 1.0;

    if (rows!=cols) abort();

    dims[0] = rows;
    dims[1] = cols;

    int me    = parallel_me();
    int nproc = parallel_nproc();

    int nblock = rows/blksz;
    int ntask  = nblock * nblock * nblock;

    float* p_a = (float *)ARMCI_Malloc_local((armci_size_t) blksz * blksz * sizeof(float));
    float* p_b = (float *)ARMCI_Malloc_local((armci_size_t) blksz * blksz * sizeof(float));
    float* p_d = (float *)ARMCI_Malloc_local((armci_size_t) blksz * blksz * sizeof(float));

    int ii,jj,kk;
    int i,j,k;

    int t = 0;

    int lo_a[ndim],lo_b[ndim],lo_d[ndim];
    int hi_a[ndim],hi_b[ndim],hi_d[ndim];
//     int rng_a[ndim],rng_b[ndim];
    int ld_a[ndim-1],ld_b[ndim-1],ld_d[ndim-1];

    parallel_sync();

    for (ii = 0 ; ii < nblock; ii++){
        for (jj = 0 ; jj < nblock; jj++){
            for (kk = 0 ; kk < nblock; kk++){
                if ( me == ( t % nproc )){

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

                    NGA_Get(g_a,lo_a,hi_a,p_a,ld_a);
                    NGA_Get(g_b,lo_b,hi_b,p_b,ld_b);
                    sgemm_("n","n",&blksz,&blksz,&blksz,&alpha,p_b,&blksz,p_a,&blksz,&beta,p_d,&blksz);
                    NGA_Acc(g_d,lo_d,hi_d,p_d,ld_d,(void*)&one);

                } // myturn
                t += 1;
            } // kk
        } // jj
    } // ii

    parallel_sync();

    int status;

    status = ARMCI_Free_local(p_d);
    if(status != 0 ) printf("%s: ARMCI_Free_local failed at line %d\n",__FILE__,__LINE__);
    status = ARMCI_Free_local(p_b);
    if(status != 0 ) printf("%s: ARMCI_Free_local failed at line %d\n",__FILE__,__LINE__);
    status = ARMCI_Free_local(p_a);
    if(status != 0 ) printf("%s: ARMCI_Free_local failed at line %d\n",__FILE__,__LINE__);
}


void gagpu_dgemm(int rows, int cols, int blksz, double alpha, int g_a, int g_b, double beta, int g_d)
{
    int ndim = 2;
    int dims[ndim];

    const double one = 1.0;

    if (rows!=cols) abort();

    dims[0] = rows;
    dims[1] = cols;

    int me    = parallel_me();
    int nproc = parallel_nproc();

    int nblock = rows/blksz;
    int ntask  = nblock * nblock * nblock;

    double* p_a = (double *)ARMCI_Malloc_local((armci_size_t) blksz * blksz * sizeof(double));
    double* p_b = (double *)ARMCI_Malloc_local((armci_size_t) blksz * blksz * sizeof(double));
    double* p_d = (double *)ARMCI_Malloc_local((armci_size_t) blksz * blksz * sizeof(double));

    int ii,jj,kk;
    int i,j,k;

    int t = 0;

    int lo_a[ndim],lo_b[ndim],lo_d[ndim];
    int hi_a[ndim],hi_b[ndim],hi_d[ndim];
//     int rng_a[ndim],rng_b[ndim];
    int ld_a[ndim-1],ld_b[ndim-1],ld_d[ndim-1];

    parallel_sync();

    for (ii = 0 ; ii < nblock; ii++){
        for (jj = 0 ; jj < nblock; jj++){
            for (kk = 0 ; kk < nblock; kk++){
                if ( me == ( t % nproc )){

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

                    NGA_Get(g_a,lo_a,hi_a,p_a,ld_a);
                    NGA_Get(g_b,lo_b,hi_b,p_b,ld_b);
                    dgemm_("n","n",&blksz,&blksz,&blksz,&alpha,p_b,&blksz,p_a,&blksz,&beta,p_d,&blksz);
                    NGA_Acc(g_d,lo_d,hi_d,p_d,ld_d,(void*)&one);

                } // myturn
                t += 1;
            } // kk
        } // jj
    } // ii

    parallel_sync();

    int status;

    status = ARMCI_Free_local(p_d);
    if(status != 0 ) printf("%s: ARMCI_Free_local failed at line %d\n",__FILE__,__LINE__);
    status = ARMCI_Free_local(p_b);
    if(status != 0 ) printf("%s: ARMCI_Free_local failed at line %d\n",__FILE__,__LINE__);
    status = ARMCI_Free_local(p_a);
    if(status != 0 ) printf("%s: ARMCI_Free_local failed at line %d\n",__FILE__,__LINE__);
}
