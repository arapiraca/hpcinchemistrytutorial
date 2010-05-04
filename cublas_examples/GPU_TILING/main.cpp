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
  if (argc<2){
     printf("\n  ERROR:  ENTER NPROCS.\n\n");
     exit(0);
  }
  if (WARP==-1){
     printf("\n  ERROR:  PLEASE DEFINE WARP SIZE.\n\n");
     exit(0);
  }
  int i;
  long tilesizeMa,tilesizeNa,ntilesMa,ntilesNa,Ma,Na,nprocs;
  long tilesizeMb,tilesizeNb,ntilesMb,ntilesNb,Mb,Nb;

  // MATRIX DIMENSIONS:
  nprocs = atoi(argv[1]);

  //M      = atoi(argv[2]);
  //N      = atoi(argv[3]);

  Ma = 2048;
  Na = 768;
  Mb = 768;
  Nb = 2048;

  // DETERMINE TILESIZE
  GPUTileSize64(nprocs,Ma,Na,tilesizeMa,tilesizeNa,ntilesMa,ntilesNa);
  GPUTileSize64(nprocs,Mb,Nb,tilesizeMb,tilesizeNb,ntilesMb,ntilesNb);

  // FILL MATRICES WITH RANDOM NUMBERS:
  real*Ain,*Bin,*Cin;
  Ain = (real*)malloc(sizeof(real)*Ma*Na);
  Bin = (real*)malloc(sizeof(real)*Mb*Nb);
  Cin = (real*)malloc(sizeof(real)*Ma*Nb);
  srand(time(NULL));
  for (i=0; i<Ma*Na; i++) Ain[i] = (real)rand()/RAND_MAX;
  for (i=0; i<Mb*Nb; i++) Bin[i] = (real)rand()/RAND_MAX;
  for (i=0; i<Ma*Nb; i++) Cin[i] = 0.;

  // ALLOCATE TILED MATRICES:
  real**A,**B,**C;
  A = GPUPaddedTiles(Ain,Ma,Na,tilesizeMa,tilesizeNa,ntilesMa,ntilesNa);
  B = GPUPaddedTiles(Bin,Mb,Nb,tilesizeMb,tilesizeNb,ntilesMb,ntilesNb);

  // NEW MATRIX:  (MaxNa).(MbxNb) = (MaxNb)
  C = GPUPaddedTiles(Cin,Ma,Nb,tilesizeMa,tilesizeNb,ntilesMa,ntilesNb);

  printf("\n  Matrix A:\n");
  printf("  tilesizeM          %15li\n",tilesizeMa);
  printf("  tilesizeN          %15li\n",tilesizeNa);
  printf("  ntilesM            %15li\n",ntilesMa);
  printf("  ntilesN            %15li\n",ntilesNa);
  printf("  n elements         %15li\n",Ma*Na);
  printf("  n elements (tiled) %15li\n",tilesizeMa*ntilesMa*tilesizeNa*ntilesNa);
  printf("\n");

  printf("  Matrix B:\n");
  printf("  tilesizeM          %15li\n",tilesizeMb);
  printf("  tilesizeN          %15li\n",tilesizeNb);
  printf("  ntilesM            %15li\n",ntilesMb);
  printf("  ntilesN            %15li\n",ntilesNb);
  printf("  n elements         %15li\n",Mb*Nb);
  printf("  n elements (tiled) %15li\n",tilesizeMb*ntilesMb*tilesizeNb*ntilesNb);
  printf("\n");

  printf("  Matrix C:\n");
  printf("  tilesizeM          %15li\n",tilesizeMa);
  printf("  tilesizeN          %15li\n",tilesizeNb);
  printf("  ntilesM            %15li\n",ntilesMa);
  printf("  ntilesN            %15li\n",ntilesNb);
  printf("  n elements         %15li\n",Ma*Nb);
  printf("  n elements (tiled) %15li\n",tilesizeMa*ntilesMa*tilesizeNb*ntilesNb);
  printf("\n");

  // NORMAL MAT MULT
  printf("  Performing full matrix-matrix multiplication...\n");
  fflush(stdout);
  XGEMM(Ain,Bin,Cin,Ma,Na,Mb,Nb);

  // TILED MAT MULT
  printf("  Performing tiled matrix-matrix multiplication...\n");
  fflush(stdout);
  TiledXGEMM(A,B,C,Ma,Na,tilesizeMa,tilesizeNa,ntilesMa,ntilesNa,
                   Mb,Nb,tilesizeMb,tilesizeNb,ntilesMb,ntilesNb);

  // CHECK THE ANSWER:
  if (sizeof(real)==sizeof(float)){
     printf("  Checking solution with single precision tolerance of 1e-3...\n");
     fflush(stdout);
     CheckAnswer(Cin,C,Ma,Nb,tilesizeMa,tilesizeNb,ntilesMa,ntilesNb,1.e-3);
  }
  else{
     printf("  Checking solution with double precision tolerance of 1e-8...\n");
     fflush(stdout);
     CheckAnswer(Cin,C,Ma,Nb,tilesizeMa,tilesizeNb,ntilesMa,ntilesNb,1.e-8);
  }

  printf("\n  Successful termination!\n\n");
  fflush(stdout);
}
