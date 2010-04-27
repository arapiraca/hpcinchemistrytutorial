#ifndef GPUTILES_HPP
#define GPUTILES_HPP
#include<stdio.h>
#include<stdlib.h>
#include<math.h>



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

real*GPUPaddedMatrix(real*A,long dimM,long dimN,long tilesizeM,long tilesizeN,long ntilesM,long ntilesN);
void GPUTileSize(long n,long M,long N,long&tilesizeM,long&tilesizeN,long&ntilesM,long&ntilesN);
void GPUTileSize64(long n,long M,long N,long&tilesizeM,long&tilesizeN,long&ntilesM,long&ntilesN);
double*DGPUPaddedMatrix(double*A,long M,long N,long M64,long N64);
#endif
