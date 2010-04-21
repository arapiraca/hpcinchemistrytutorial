#ifndef GPUTILES_HPP
#define GPUTILES_HPP
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
void GPUTileSize(long n,long M,long N,long&tilesizeM,long&tilesizeN,long&ntilesM,long&ntilesN);
void GPUTileSize64(long n,long M,long N,long&tilesizeM,long&tilesizeN,long&ntilesM,long&ntilesN);
double*DGPUPaddedMatrix(double*A,long M,long N,long M64,long N64);
float*SGPUPaddedMatrix(float*A,long M,long N,long M64,long N64);
#endif
