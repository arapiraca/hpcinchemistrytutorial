/*==============================================================================

                COPYRIGHT

The following is a notice of limited availability of the code, and disclaimer
which must be included in the prologue of the code and in all source listings
of the code.

Copyright Notice
+ 2009 University of Chicago

Permission is hereby granted to use, reproduce, prepare derivative works, and
to redistribute to others.  This software was authored by:

A. Eugene DePrince
Center for Nanoscale Materials
Argonne National Laboratory
Argonne IL 60439 USA
phone: (630) 252-5418
e-mail: adeprince@anl.gov

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

==============================================================================*/

#include"gputiles.hpp"
int main(int argc,char*argv[]){
  if (argc<4){
     printf("\n  ERROR:  ENTER NPROCS AND DIMENSIONS, M AND N.\n\n");
     exit(0);
  }
  if (WARP==-1){
     printf("\n  ERROR:  PLEASE DEFINE WARP SIZE.\n\n");
     exit(0);
  }
  int i,j;
  long tilesizeM,tilesizeN,ntilesM,ntilesN,M,N,nprocs;
  long tilesizeM64,tilesizeN64,tilesizeMN,tilesizeMN64;


  // MATRIX DIMENSIONS:
  nprocs = atoi(argv[1]);
  M      = atoi(argv[2]);
  N      = atoi(argv[3]);

  //printf("\n  NPROCS = %10li\n",nprocs);
  //printf("       M = %10li\n",M);
  //printf("       N = %10li\n\n",N);

  // DETERMINE TILESIZE
  GPUTileSize64(nprocs,M,N,tilesizeM,tilesizeN,ntilesM,ntilesN);

  // PAD MATRIX WITH ZEROS:
  real*A,*B;
  A = (real*)malloc(sizeof(real)*M*N);
  for (i=0; i<M*N; i++) A[i] = real(i);
  B = GPUPaddedMatrix(A,M,N,tilesizeM,tilesizeN,ntilesM,ntilesN);

  printf("\n");
  printf("  tilesizeM          %15li\n",tilesizeM);
  printf("  tilesizeN          %15li\n",tilesizeN);
  printf("  ntilesM            %15li\n",ntilesM);
  printf("  ntilesN            %15li\n",ntilesN);
  printf("  n elements         %15li\n",M*N);
  printf("  n elements (tiled) %15li\n",tilesizeM*ntilesM*tilesizeN*ntilesN);
  printf("\n");
}
