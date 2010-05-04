/*=============================================================================

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

=============================================================================*/


/*=============================================================================

  gputiles.cpp 

  BY EUGENE DEPRINCE

  ROUTINES TO TILE MxN MATRIX INTO MULTIPLES OF
  WARPSIZE FOR GPU COMPUTING.  

  MxN ELEMENTS SPLIT OVER NPROCS.  SQRT OF THAT
  IS THE APPROXIMATE TILE DIMENSION.  NOTE THAT
  BECAUSE TILES ARE SQUARE, WHEN N<<M, WE WILL USE 
  FAR MORE MEMORY THAN NECESSARY IN OUR PADDED MAT.  

  FUNCIONS:

  GPUTileSize() * DEAD CODE *:
    COMPUTES TILE DIMENSION FOR MxN MATRIX.
    RETURNS THIS DIMENSION AS WELL AS THE
    NUMBER OF TILES IN M AND N.

  GPUTileSize64():
    COMPUTES TILE SIZE FOR MxN MATRIX SUCH THAT
    THE DIMENSION OF THE TILE IS DIVISIBLE BY WARPSIZE.
    RETURNS THIS DIMENSION AS WELL AS THE
    NUMBER OF TILES IN M AND N.

  GPUPaddedMatrix() * DEAD CODE *:
    RETURNS A TILED MATRIX THAT CONTAINS THE
    ORIGINAL MATRIX AND THE EXTRA ELEMENTS
    ARE PADDED WITH ZEROS.  THE PRECISION OF
    THE MATRIX IS CONTROLLED BY -DSINGLE OR
    -DDOUBLE AS A FLAG IN THE MAKEFILE

  GPUPaddedTiles():
    RETURNS A TILED MATRIX THAT CONTAINS THE
    ORIGINAL MATRIX AND THE EXTRA ELEMENTS
    ARE PADDED WITH ZEROS.  THE PRECISION OF
    THE MATRIX IS CONTROLLED BY -DSINGLE OR
    -DDOUBLE AS A FLAG IN THE MAKEFILE.  THE 
    TILED MATRIX IS AN ARRAY OF POINTERS THAT
    POINT TO EACH TILE (type real**)

  XGEMM():
    MATRIX-MATRIX MULTIPLICATION FOR SINGLE OR DOUBLE 
    PRECISION.  CAN BE REPLACED WITH LAPACK DGEMM OR SGEMM

  TiledXGEMM():
    TILED MATRIX-MATRIX MULTIPLICATION FOR SINGLE OR DOUBLE PRECISION.
    CALLS XGEMM FOR EACH TILE OF TARGET MATRIX.

  CheckAnswer():
    CHECK TO MAKE SURE XGEMM AND TiledXGEMM GIVE THE SAME
    RESULTS.  DIFFERENT TOLERANCES MAY BE USED FOR SINGLE OR 
    DOUBLE PRECISION.  IN PRACTICE, FOR VERY LARGE MATRICES,
    TiledXGEMM AND THE FULL XGEMM WILL GIVE QUITE DIFFERENT 
    RESULTS FOR WHEN USING SINGLE PRECISION.

=============================================================================*/

#include"gputiles.hpp"
void GPUTileSize(long nprocs,long M,long N,long&tilesizeM,long&tilesizeN,long&ntilesM,long&ntilesN){

  // TOTAL DIMENSION:
  long tilesize,dim = M*N;

  // NUMBER OF ELEMENTS PER PROCESSOR:
  long nelem_per_proc;
  if (dim%nprocs==0) nelem_per_proc = dim/nprocs;
  else               nelem_per_proc = (dim+nprocs-dim%nprocs)/nprocs;


  // TILE SIZE:
  real dum = sqrt(nelem_per_proc);
  if ((long)dum - dum < 0.) tilesize = (long)dum+1;
  else                      tilesize = (long)dum;

  // NUMBER OF TILES IN EACH DIMENSION:
  if (M/tilesize - (real)M/tilesize < 0.) ntilesM = M/tilesize + 1;
  else                                     ntilesM = M/tilesize;
  if (N/tilesize - (real)N/tilesize < 0.) ntilesN = N/tilesize + 1;
  else                                     ntilesN = N/tilesize;
  
  tilesizeM = tilesize;
  tilesizeN = tilesize;
}
void GPUTileSize64(long nprocs,long M,long N,long&tilesizeM,long&tilesizeN,long&ntilesM,long&ntilesN){
  // TOTAL DIMENSION:
  long tilesize,dim = M*N;

  // NUMBER OF ELEMENTS PER PROCESSOR:
  long nelem_per_proc; 
  if (dim%nprocs==0) nelem_per_proc = dim/nprocs;
  else               nelem_per_proc = (dim+nprocs-dim%nprocs)/nprocs;

  // TILE SIZE:
  real dum = sqrt(nelem_per_proc);
  if ((long)dum - dum < 0.) tilesize = (long)dum+1;
  else                      tilesize = (long)dum;

  // TILE SIZE, DIVISIBLE BY 64 (OR OPTIMAL WARP SIZE)
  if (tilesize%WARP!=0){
     tilesize = (tilesize+WARP-tilesize%WARP);
  }

  // NUMBER OF TILES IN EACH DIMENSION:
  if (M/tilesize - (real)M/tilesize < 0.)  ntilesM = M/tilesize + 1;
  else                                     ntilesM = M/tilesize;
  if (N/tilesize - (real)N/tilesize < 0.)  ntilesN = N/tilesize + 1;
  else                                     ntilesN = N/tilesize;
  
  tilesizeM = tilesize;
  tilesizeN = tilesize;
}

real*GPUPaddedMatrix(real*A,long dimM,long dimN,long tilesizeM,long tilesizeN,long ntilesM,long ntilesN){
  long n,m,ne,me,nt,mt,m_is_padded,n_is_padded,pad_shift,tilenumber,pos,pad_pos;
  long padN = ntilesN*tilesizeN;
  long padM = ntilesM*tilesizeM;

  real*B;
  B = (real*)malloc(padM*padN*sizeof(real));

  //m (down) x n (across)

  for (mt=0; mt<ntilesM; mt++){

      for (nt=0; nt<ntilesN; nt++){

          // TILE NUMBER AND SHIFT FOR POSITION IN PADDED MATRIX:
          tilenumber = mt * ntilesN + nt;
          pad_shift  = tilenumber * tilesizeN * tilesizeM;

          for (me=0; me<tilesizeM;me++){
              m_is_padded = 0;


              // M POSITION IN ORIGINAL MATRIX:
              m = tilesizeM*mt + me;
              if (m>=dimM) m_is_padded = 1;

              for (ne=0; ne<tilesizeN;ne++){
                  n_is_padded = 0;

                  // N POSITION IN ORIGINAL MATRIX:
                  n = tilesizeN*nt + ne;
                  if (n>=dimN) n_is_padded = 1;

                  // POSITION IN ORIGINAL MATRIX:
                  pos = m*dimN + n;

                  // POSITION IN PADDED MATRIX:
                  pad_pos = pad_shift + me * tilesizeN + ne;

                  if (m_is_padded || n_is_padded) B[pad_pos] = 0.;
                  else                            B[pad_pos] = A[pos];

              }
          }
      }
  }

  return B;
}
real**GPUPaddedTiles(real*A,long dimM,long dimN,long tilesizeM,long tilesizeN,long ntilesM,long ntilesN){
  long n,m,ne,me,nt,mt,m_is_padded,n_is_padded,pad_shift,tilenumber,pos,pad_pos;
  //long padN = ntilesN*tilesizeN;
  //long padM = ntilesM*tilesizeM;

  real**B;
  //B = (real**)malloc(padM*padN*sizeof(real));
  B = (real**)malloc(ntilesN*ntilesM*sizeof(real*));
  for (n=0; n<ntilesN*ntilesM; n++){
      B[n] = (real*)malloc(tilesizeM*tilesizeN*sizeof(real));
  }

  //m (down) x n (across)

  for (mt=0; mt<ntilesM; mt++){

      for (nt=0; nt<ntilesN; nt++){

          // TILE NUMBER AND SHIFT FOR POSITION IN PADDED MATRIX:
          tilenumber = mt * ntilesN + nt;
          pad_shift  = tilenumber * tilesizeN * tilesizeM;

          for (me=0; me<tilesizeM;me++){
              m_is_padded = 0;


              // M POSITION IN ORIGINAL MATRIX:
              m = tilesizeM*mt + me;
              if (m>=dimM) m_is_padded = 1;

              for (ne=0; ne<tilesizeN;ne++){
                  n_is_padded = 0;

                  // N POSITION IN ORIGINAL MATRIX:
                  n = tilesizeN*nt + ne;
                  if (n>=dimN) n_is_padded = 1;

                  // POSITION IN ORIGINAL MATRIX:
                  pos = m*dimN + n;

                  // POSITION IN PADDED MATRIX:
                  pad_pos = me * tilesizeN + ne;

                  if (m_is_padded || n_is_padded) B[tilenumber][pad_pos] = 0.;
                  else                            B[tilenumber][pad_pos] = A[pos];
              }
          }
      }
  }

  return B;
}


void TiledXGEMM(real**A,real**B,real**C,
  long dimMa,long dimNa,long tilesizeMa,long tilesizeNa,long ntilesMa,long ntilesNa,
  long dimMb,long dimNb,long tilesizeMb,long tilesizeNb,long ntilesMb,long ntilesNb){

  if (ntilesNa!=ntilesMb){
     printf("\n  ERROR: INNER TILE DIMENSIONS DIFFER.\n\n");
     exit(0);
  }
  int k,tm,tn,tc;
  long ntilesCommon = ntilesNa;
  real*temp;
  temp = (real*)malloc(tilesizeMa*tilesizeNb*sizeof(real));
  for (k=0; k<tilesizeMa*tilesizeNb; k++){
      temp[k] = 0.;
  }
  // LOOP OVER TILES OF TARGET MATRIX, C:
  for (tm=0; tm<ntilesMa; tm++){
      for (tn=0; tn<ntilesNb; tn++){

          for (tc=0; tc<ntilesCommon; tc++){

              XGEMM(A[tm*ntilesCommon+tc],B[tc*ntilesNb+tn],temp,
                   tilesizeMa,tilesizeNa,tilesizeMb,tilesizeNb);

              for (k=0; k<tilesizeMa*tilesizeNb; k++){
                  C[tm*ntilesNb+tn][k] += temp[k];
                  temp[k] = 0.;
              }

          }
          
      }
  }
  free(temp);

}

void XGEMM(real*A,real*B,real*C,long Ma,long Na,long Mb,long Nb){
  if (Na!=Mb){
     printf("\n  ERROR: INNER DIMENSIONS DIFFER.\n\n");
     exit(0);
  }
  int m,n,k;
  real sum;
  for (m=0; m<Ma; m++){
      for (n=0; n<Nb; n++){
          sum = 0.;
          for (k=0; k<Na; k++){
              sum += A[m*Na+k] * B[k*Nb+n];
          }
          C[m*Nb+n] = sum;
      }
  }
}



void CheckAnswer(real*Cin,real**C,long dimM,long dimN,long tilesizeM,long tilesizeN,long ntilesM,long ntilesN,real tol){
  long n,m,ne,me,nt,mt,m_is_padded,n_is_padded,pad_shift,tilenumber,pos,pad_pos;
  real diff;

  //m (down) x n (across)

  for (mt=0; mt<ntilesM; mt++){

      for (nt=0; nt<ntilesN; nt++){

          // TILE NUMBER AND SHIFT FOR POSITION IN PADDED MATRIX:
          tilenumber = mt * ntilesN + nt;
          pad_shift  = tilenumber * tilesizeN * tilesizeM;

          for (me=0; me<tilesizeM;me++){
              m_is_padded = 0;


              // M POSITION IN ORIGINAL MATRIX:
              m = tilesizeM*mt + me;
              if (m>=dimM) m_is_padded = 1;

              for (ne=0; ne<tilesizeN;ne++){
                  n_is_padded = 0;

                  // N POSITION IN ORIGINAL MATRIX:
                  n = tilesizeN*nt + ne;
                  if (n>=dimN) n_is_padded = 1;

                  // POSITION IN ORIGINAL MATRIX:
                  pos = m*dimN + n;

                  // POSITION IN PADDED MATRIX:
                  pad_pos = me * tilesizeN + ne;

                  if (m_is_padded || n_is_padded) {
                     if (C[tilenumber][pad_pos]!=0.){
                        printf("\n  ERROR: NONZERO PADDED ELEMENTS: %f\n\n",C[tilenumber][pad_pos]);
                        //exit(0);
                     }
                  }
                  else{
                      diff = fabs(C[tilenumber][pad_pos] - Cin[pos]);
                      if (diff>tol){
                        printf("\n  ERROR: ABS(Cpad-C) = %f\n\n",diff);
                        exit(0);
                      }
                  }
              }
          }
      }
  }
}
