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
  64 FOR GPU COMPUTING.  

  MxN ELEMENTS SPLIT OVER NPROCS.  SQRT OF THAT
  IS THE APPROXIMATE TILE DIMENSION.  NOTE THAT
  BECAUSE TILES ARE SQUARE, WHEN N<<M, WE WILL USE 
  FAR MORE MEMORY THAN NECESSARY IN OUR PADDED MAT.  

  FUNCIONS:

  GPUTileSize():
    COMPUTES TILE DIMENSION FOR MxN MATRIX.
    RETURNS THIS DIMENSION AS WELL AS THE
    NUMBER OF TILES IN M AND N.

  GPUTileSize64():
    COMPUTES TILE SIZE FOR MxN MATRIX SUCH THAT
    THE DIMENSION OF THE TILE IS DIVISIBLE BY 64.
    RETURNS THIS DIMENSION AS WELL AS THE
    NUMBER OF TILES IN M AND N.

  GPUPaddedMatrix():
    RETURNS A TILED MATRIX THAT CONTAINS THE
    ORIGINAL MATRIX AND THE EXTRA ELEMENTS
    ARE PADDED WITH ZEROS.  THE PRECISION OF
    THE MATRIX IS CONTROLLED BY -DSINGLE OR
    -DDOUBLE AS A FLAG IN THE MAKEFILE

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
