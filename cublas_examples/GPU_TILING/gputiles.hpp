#ifndef GPUTILES_HPP
#define GPUTILES_HPP
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

// DEFINE THE WARP SIZE IN THE MAKEFILE
#ifndef WARP
  #define WARP -1
#endif
// DEFINE THE PRECISION FOR CALCULATION:
#ifdef SINGLE
  typedef float real;
#endif
#ifdef DOUBLE
  typedef double real;
#endif

real**GPUPaddedTiles(real*A,long dimM,long dimN,long tilesizeM,long tilesizeN,long ntilesM,long ntilesN);
real*GPUPaddedMatrix(real*A,long dimM,long dimN,long tilesizeM,long tilesizeN,long ntilesM,long ntilesN);
void GPUTileSize(long n,long M,long N,long&tilesizeM,long&tilesizeN,long&ntilesM,long&ntilesN);
void GPUTileSize64(long n,long M,long N,long&tilesizeM,long&tilesizeN,long&ntilesM,long&ntilesN);
double*DGPUPaddedMatrix(double*A,long M,long N,long M64,long N64);
void XGEMM(real*A,real*B,real*C,long Ma,long Na,long Mb,long Nb);
void TiledXGEMM(real**A,real**B,real**C,
     long dimMa,long dimNa,long tilesizeMa,long tilesizeNa,long ntilesMa,long ntilesNa,
     long dimMb,long dimNb,long tilesizeMb,long tilesizeNb,long ntilesMb,long ntilesNb);
void CheckAnswer(real*Cin,real**C,long dimM,long dimN,long tilesizeM,long tilesizeN,long ntilesM,long ntilesN,real tol);


#endif
