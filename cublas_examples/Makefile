#############################################
#
#               Linux (64-bit)
#
#############################################

   # Fusion
   ifeq ($(HOSTNAME),flogin1.lcrc.anl.gov)
      OPT = -O3 -xSSE2,SSE3,SSSE3,SSE4.1,SSE4.2 -no-prec-div -funroll-loops
      MKL_DIR = /soft/mkl/10.2.2.025
      CUDA_DIR =
      LIB_BITS =
      CUDA = NO
      MKL = YES
   endif
   ifeq ($(HOSTNAME),flogin2.lcrc.anl.gov)
      OPT = -O3 -xSSE2,SSE3,SSSE3,SSE4.1,SSE4.2 -no-prec-div -funroll-loops
      MKL_DIR = /soft/mkl/10.2.2.025
      CUDA_DIR =
      LIB_BITS =
      CUDA = NO
      MKL = YES
   endif
   ifeq ($(HOSTNAME),flogin3.lcrc.anl.gov)
      OPT = -O3 -xSSE2,SSE3,SSSE3,SSE4.1,SSE4.2 -no-prec-div -funroll-loops
      MKL_DIR = /soft/mkl/10.2.2.025
      CUDA_DIR =
      LIB_BITS =
      CUDA = NO
      MKL = YES
   endif

   # Lincoln
   ifeq ($(HOSTNAME),honest1.ncsa.uiuc.edu)
      OPT = -O3 -xT -no-prec-div -funroll-loops
      MKL_DIR = /usr/local/intel/mkl/10.2.2
      CUDA_DIR = /usr/local/cuda-2.2/cuda
      LIB_BITS =
      CUDA = YES
      MKL = YES
   endif
   ifeq ($(HOSTNAME),honest2.ncsa.uiuc.edu)
      OPT = -O3 -xT -no-prec-div -funroll-loops
      MKL_DIR = /usr/local/intel/mkl/10.2.2
      CUDA_DIR = /usr/local/cuda-2.2/cuda
      LIB_BITS =
      CUDA = YES
      MKL = YES
   endif
   ifeq ($(HOSTNAME),honest3.ncsa.uiuc.edu)
      OPT = -O3 -xT -no-prec-div -funroll-loops
      MKL_DIR = /usr/local/intel/mkl/10.2.2
      CUDA_DIR = /usr/local/cuda-2.2/cuda
      LIB_BITS =
      CUDA = YES
      MKL = YES
   endif

   # Megatron
   ifeq ($(HOSTNAME),megatron)
      OPT = -O3 -xSSE2,SSE3,SSSE3,SSE4.1 -no-prec-div -funroll-loops
      MKL_DIR = /opt/intel/Compiler/11.1/059/mkl
      CUDA_DIR = /opt/cuda
      LIB_BITS = 64
      CUDA = YES
      MKL = YES
   endif

   DEFINES =
   INC = 

   EXTRAS = -lm -lpthread -openmp
   PLIB = $(EXTRAS)
   SLIB = $(EXTRAS)

   ifeq ($(MKL),YES)
      DEFINES += -DMKL
      PBLAS_LIB = -L${MKL_DIR}/lib/em64t -lguide -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lsvml
      SBLAS_LIB = -L${MKL_DIR}/lib/em64t -lguide -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lsvml
      INC += -I${MKL_DIR}/include
      PLIB += $(PBLAS_LIB)
      SLIB += $(SBLAS_LIB)
   endif

   ifeq ($(CUDA),YES)
      DEFINES += -DCUDA
      CUBLAS_LIB = -L${CUDA_DIR}/lib${LIB_BITS} -lcublas
      INC += -I${CUDA_DIR}/include
      PLIB += $(CUBLAS_LIB)
      SLIB += $(CUBLAS_LIB) 
   endif

   FC = ifort
   FOPT = $(OPT)
   FFLAGS = -g $(COPT) $(INC) $(DEFINES)

   CC = icc
   COPT = $(OPT)
   CFLAGS = -g -std=c99 $(COPT) $(INC) $(DEFINES)

   LD = $(CC)
   LDFLAGS = -g $(OPT)

#############################################
#
#               End of Targets
#
#############################################

all: simpleCUBLAS.x CUBLAS_sgemm.x CUBLAS_dgemm.x
#
# LEGACY TARGET
#
simpleCUBLAS.x: simpleCUBLAS.o getticks.o 
	$(LD) $(LDFLAGS) getticks.o simpleCUBLAS.o $(PLIB) -o simpleCUBLAS.x
#
# SERIAL/PARALLEL BLAS BINARIES
#
CUBLAS_sgemm.x: CUBLAS_sgemm.o getticks.o
	$(LD) $(LDFLAGS) getticks.o CUBLAS_sgemm.o $(SLIB) -o CUBLAS_sgemm_S.x
	$(LD) $(LDFLAGS) getticks.o CUBLAS_sgemm.o $(PLIB) -o CUBLAS_sgemm_P.x

CUBLAS_dgemm.x: CUBLAS_dgemm.o getticks.o
	$(LD) $(LDFLAGS) getticks.o CUBLAS_dgemm.o $(SLIB) -o CUBLAS_dgemm_S.x
	$(LD) $(LDFLAGS) getticks.o CUBLAS_dgemm.o $(PLIB) -o CUBLAS_dgemm_P.x
#
# OBJECTS
#
simpleCUBLAS.o: simpleCUBLAS.c
	$(CC) -c $(CFLAGS) $(INC) simpleCUBLAS.c

CUBLAS_sgemm.o: CUBLAS_sgemm.c
	$(CC) -c $(CFLAGS) $(INC) CUBLAS_sgemm.c

CUBLAS_dgemm.o: CUBLAS_dgemm.c
	$(CC) -c $(CFLAGS) $(INC) CUBLAS_dgemm.c

getticks.o: getticks.c
	gcc -c getticks.c
#
# UTILITY OPERATIONS
#
refresh: realclean all

clean:
	rm -f *.o

realclean: clean
	rm -f *.x *.a

