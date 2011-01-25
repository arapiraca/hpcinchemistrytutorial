/***************************************************************************

                  COPYRIGHT

The following is a notice of limited availability of the code, and disclaimer
which must be included in the prologue of the code and in all source listings
of the code.

Copyright Notice
+2009 University of Chicago

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

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#ifdef _OPENMP
#include <omp.h>
#else
#warning OpenMP not enabled!
#endif

#include "mymath.h"

inline double gettime(void)
{
#ifdef _OPENMP
    return omp_get_wtime();
#else
    clock_t time;
    time = clock();
    return (double) (time/CLOCKS_PER_SEC);
//    return (double) time(NULL);
#endif
}

inline int imax(int a, int b)
{
    return ( (a>b) ? a : b );
}

/****** ESSL documentation ******
 *
 * syntax:
 * 
 * dgemm(transa, transb, 
 *       l, n, m, 
 *       alpha, 
 *       a, lda, 
 *       b, ldb, 
 *       beta, 
 *       c, ldc);
 *
 * usage:
 * 
 * dgemm_("n","n",
 *        &rowa,&colb,&cola,
 *        &alpha,
 *        p_a,&rowa,
 *        p_b,&rowb,
 *        &beta,
 *        p_d,&rowc);
 *
 * syntax:
 * 
 * dgemms(a, lda, transa, 
 *        b, ldb, transb, 
 *        c, ldc, 
 *        l, m, n, 
 *        aux, naux);
 * 
 * naux = max( n*l , 0.7*m*(l+n) )
 *
 * usage:
 *
 * dgemms_(p_a,&rowa,"n",
 *         p_b,&rowb,"n",
 *         p_e,&rowc,
 *         &rowa,&cola,&colb,
 *         aux,&naux);
 *
 * naux = max( rowa*colb,0.7*cola*(rowa+colb) )
 *
 ********************************/


int main(int argc, char **argv)
{
#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    fprintf(stdout,"using %d OpenMP threads\n",num_threads);
#endif

    if (argc!=6) fprintf(stderr,"./dgemm_performance.x <dim1> <dim2> <dim3> <timings> <check>\n");

    int dim1 = ( argc>1 ? atoi(argv[1]) : 50 );
    int dim2 = ( argc>2 ? atoi(argv[2]) : 50 );
    int dim3 = ( argc>3 ? atoi(argv[3]) : 50 );

    int count = ( argc>4 ? atoi(argv[4]) : 10 );
    int check = ( argc>5 ? atoi(argv[5]) : 0 );
    //if (check==1) count = 1; /* disable if checking for correctness, unnecessary for beta = 0.0 */

    fprintf(stdout,"dim1 = %d\n",dim1);
    fprintf(stdout,"dim2 = %d\n",dim2);
    fprintf(stdout,"dim3 = %d\n",dim3);
    fprintf(stdout,"number of timings = %d\n",count);
    fprintf(stdout,"validate against dmatmul = %d\n",check);

    if ((dim1<1) || (dim2<1) || (dim3<1)) return(1);
    if ((check!=0) && (check!=1)) return(1);

    int rowc = dim1;
    int rowa = dim1;
    int cola = dim3;
    int rowb = dim3;
    int colb = dim2;
    int colc = dim2;

    int i,j,k;
    int naux = imax( rowa*colb , 0.7*cola*(rowa+colb) );
    fprintf(stdout,"naux = %d\n",naux);

    double* p_a;
    double* p_b;
    double* p_c; /* output from dmatmul (loops)       */
    double* p_d; /* output from dgemm   (standard)    */
    double* p_e; /* output from dgemul  (alternative) */
    double* p_f; /* output from dgemms  (Strassen)    */
    double* aux; /* dgemms scratch space              */

    double start,finish;
    double t_loops = 0.0;
    double t_dgemm = 0.0;
    double t_dgemul = 0.0;
    double t_dgemms = 0.0;

    double error = 0.0;
    double alpha = 1.0;
    double beta = 0.0;

    double ngf;
    if (alpha==0.0)
        ngf = 2.0 * dim1 * dim2 * 1e-9;
    else
        ngf = 2.0 * dim1 * dim2 * dim3 * 1e-9;
    fprintf(stdout,"alpha = %30.14lf\n",alpha);
    fprintf(stdout,"beta  = %30.14lf\n",beta);
    fprintf(stdout,"2*M*N*K = %10.5lf gigaflops\n",ngf);
    fprintf(stdout,"\n");

    /* START UP */

    p_a = (double *) malloc(rowa*cola*sizeof(double));
    p_b = (double *) malloc(rowb*colb*sizeof(double));
    p_c = (double *) malloc(rowc*colc*sizeof(double));
    p_d = (double *) malloc(rowc*colc*sizeof(double));
    p_e = (double *) malloc(rowc*colc*sizeof(double));
    p_f = (double *) malloc(rowc*colc*sizeof(double));
    aux = (double *) malloc(naux*sizeof(double));

    start = gettime();
    dzero(rowa*cola,p_a); /* zero using threaded call to spread across memory controllers (hopefully) */
    dzero(rowb*colb,p_b); /* zero using threaded call to spread across memory controllers (hopefully) */
    dzero(rowc*colc,p_c); /* output from dmatmul (loops)    */
    dzero(rowc*colc,p_d); /* output from dgemm   (standard) */
    dzero(rowc*colc,p_e); /* output from dgemms  (Strassen) */
    finish = gettime();
    fprintf(stdout,"time for dzero (x5) = %30.14lf\n",finish-start);
    dzero(naux,aux);      /* dgemms scratch space           */

    start = gettime();
    drand(rowa*cola,p_a);
    drand(rowb*colb,p_b);
    finish = gettime();
    fprintf(stdout,"time for drand (x2) = %30.14lf\n",finish-start);

    printf("max val in p_a = %30.14lf\n",dmax(rowa*cola,p_a));
    printf("max val in p_b = %30.14lf\n",dmax(rowb*colb,p_b));

    fprintf(stdout,"\n");
    fprintf(stdout,"%20s %4s %4s %4s %14s %20s %20s\n","ESSL","dim1","dim2","dim3","seconds","gigaflop/s","error");

    /* PASS 1 */

    /* Loops */
    if (check==1){
        dzero(rowc*colc,p_c);
        if (beta==0.0) /* if C is not accumulated, no need to run more than once */
        {
            start = gettime();
            dmatmul(dim1,dim2,dim3,alpha,p_a,p_b,beta,p_c);
            finish = gettime();
            t_loops = (finish - start);
        }
        else
        {
            start = gettime();
            for (i=0;i<count;i++) dmatmul(dim1,dim2,dim3,alpha,p_a,p_b,beta,p_c);
            finish = gettime();
            t_loops = (finish - start)/count;
        }
        fprintf(stdout,"%20s %4d %4d %4d %20.14lf %20.14lf\n","dmatmul",dim1,dim2,dim3,t_loops,ngf/t_loops);
    }
    t_loops = 0.0;

    /* Standard */
    dzero(rowc*colc,p_d);
    start = gettime();
    for (i=0;i<count;i++) dgemm_("n","n",&rowa,&colb,&cola,&alpha,p_a,&rowa,p_b,&rowb,&beta,p_d,&rowc);
    finish = gettime();
    t_dgemm = (finish - start)/count;

    if (check==1)
    {
        error = ddiff(rowc*colc,p_c,p_d);
        fprintf(stdout,"%20s %4d %4d %4d %20.14lf %20.14lf %20.14lf\n","DGEMM",dim1,dim2,dim3,t_dgemm,ngf/t_dgemm,error);
        /*
        fprintf(stdout,"===================\n");
        printf("max val in p_c = %30.14lf\n",dmax(rowc*colc,p_c));
        printf("max val in p_d = %30.14lf\n",dmax(rowc*colc,p_d));
        dprint2(rowc*colc,p_c,p_d);
        fprintf(stdout,"===================\n");
        */
    }
    else
    {
        fprintf(stdout,"%20s %4d %4d %4d %20.14lf %20.14lf %20s\n","DGEMM",dim1,dim2,dim3,t_dgemm,ngf/t_dgemm,"");
    }
    t_dgemm = 0.0;

    /* Alternative */
    if (alpha==1.0 && beta==0.0)
    {
        dzero(rowc*colc,p_e);
        start = gettime();
        for (i=0;i<count;i++) dgemul_(p_a,&rowa,"n", p_b,&rowb,"n", p_e,&rowc, &rowa,&cola,&colb);
        finish = gettime();
        t_dgemul = (finish - start)/count;

        if (check==1)
        {
            error = ddiff(rowc*colc,p_c,p_d);
            fprintf(stdout,"%20s %4d %4d %4d %20.14lf %20.14lf %20.14lf\n","DGEMUL",dim1,dim2,dim3,t_dgemul,ngf/t_dgemul,error);
        }
        else
        {
            fprintf(stdout,"%20s %4d %4d %4d %20.14lf %20.14lf %20s\n","DGEMUL",dim1,dim2,dim3,t_dgemul,ngf/t_dgemul,"");
        }
        t_dgemul = 0.0;

        /* Strassen */
        dzero(rowc*colc,p_f);
        start = gettime();
        for (i=0;i<count;i++) dgemms_(p_a,&rowa,"n", p_b,&rowb,"n", p_f,&rowc, &rowa,&cola,&colb, aux,&naux);
        finish = gettime();
        t_dgemms = (finish - start)/count;

        if (check==1)
        {
            error = ddiff(rowc*colc,p_c,p_d);
            fprintf(stdout,"%20s %4d %4d %4d %20.14lf %20.14lf %20.14lf\n","DGEMMS",dim1,dim2,dim3,t_dgemms,ngf/t_dgemms,error);
        }
        else
        {
            fprintf(stdout,"%20s %4d %4d %4d %20.14lf %20.14lf %20s\n","DGEMMS",dim1,dim2,dim3,t_dgemms,ngf/t_dgemms,"");
        }
        t_dgemms = 0.0;
    }
    else
    {
        fprintf(stdout,"DGEMUL and DGEMMS only work for alpha=1.0 and beta=0.0!\n");
    }

    /* PASS 2 */

    /* CLEAN UP */

    free(p_f);
    free(p_e);
    free(p_d);
    free(p_c);
    free(p_b);
    free(p_a);

    return(0);
}
