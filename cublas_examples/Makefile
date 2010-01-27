#############################################
#
#               Linux (64-bit)
#
#############################################

   # Fusion
   ifeq ($(HOSTNAME),flogin1.lcrc.anl.gov)
      OPT = -O3 -xSSE2,SSE3,SSSE3,SSE4.1,SSE4.2 -no-prec-div -funroll-loops
      INTEL_DIR = 
      MKL_DIR = /soft/mkl/10.2.2.025
      CUDA_DIR =
      LIB_BITS =
      CUDA = NO
      MKL = YES
      OPENMP = YES
   endif
   ifeq ($(HOSTNAME),flogin2.lcrc.anl.gov)
      OPT = -O3 -xSSE2,SSE3,SSSE3,SSE4.1,SSE4.2 -no-prec-div -funroll-loops
      INTEL_DIR = 
      MKL_DIR = /soft/mkl/10.2.2.025
      CUDA_DIR =
      LIB_BITS =
      CUDA = NO
      MKL = YES
      OPENMP = YES
   endif
   ifeq ($(HOSTNAME),flogin3.lcrc.anl.gov)
      OPT = -O3 -xSSE2,SSE3,SSSE3,SSE4.1,SSE4.2 -no-prec-div -funroll-loops
      INTEL_DIR = 
      MKL_DIR = /soft/mkl/10.2.2.025
      CUDA_DIR =
      LIB_BITS =
      CUDA = NO
      MKL = YES
      OPENMP = YES
   endif

   # Lincoln
   ifeq ($(HOSTNAME),honest1.ncsa.uiuc.edu)
      OPT = -O3 -xT -no-prec-div -funroll-loops
      INTEL_DIR = 
      MKL_DIR = /usr/local/intel/mkl/10.2.2
      CUDA_DIR = /usr/local/cuda-2.2/cuda
      LIB_BITS =
      CUDA = YES
      MKL = YES
      OPENMP = YES
   endif
   ifeq ($(HOSTNAME),honest2.ncsa.uiuc.edu)
      OPT = -O3 -xT -no-prec-div -funroll-loops
      INTEL_DIR = 
      MKL_DIR = /usr/local/intel/mkl/10.2.2
      CUDA_DIR = /usr/local/cuda-2.2/cuda
      LIB_BITS =
      CUDA = YES
      MKL = YES
      OPENMP = YES
   endif
   ifeq ($(HOSTNAME),honest3.ncsa.uiuc.edu)
      OPT = -O3 -xT -no-prec-div -funroll-loops
      INTEL_DIR = 
      MKL_DIR = /usr/local/intel/mkl/10.2.2
      CUDA_DIR = /usr/local/cuda-2.2/cuda
      LIB_BITS =
      CUDA = YES
      MKL = YES
      OPENMP = YES
   endif

   # Megatron
   ifeq ($(HOSTNAME),megatron)
      OPT = -O3 -xSSE2,SSE3,SSSE3,SSE4.1 -no-prec-div -funroll-loops
      INTEL_DIR = /opt/intel/Compiler/11.1/059
      MKL_DIR = /opt/intel/Compiler/11.1/059/mkl
      CUDA_DIR = /opt/cuda
      LIB_BITS = 64
      CUDA = YES
      MKL = YES
      OPENMP = YES
   endif

   DEFINES =
   INC =
   PLIB =
   SLIB =

   ifeq ($(OPENMP),YES)
      DEFINES += -DOPENMP
      INC += -I$(INTEL_DIR)/include
      PLIB += -openmp
      SLIB += -openmp
   endif

   ifeq ($(MKL),YES)
      PBLAS_LIB = -L${MKL_DIR}/lib/em64t -lguide -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lsvml
      SBLAS_LIB = -L${MKL_DIR}/lib/em64t -lguide -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lsvml
      DEFINES += -DMKL
      INC += -I${MKL_DIR}/include
      PLIB += $(PBLAS_LIB)
      SLIB += $(SBLAS_LIB)
   endif

   ifeq ($(CUDA),YES)
      CUBLAS_LIB = -L${CUDA_DIR}/lib${LIB_BITS} -lcublas
      DEFINES += -DCUDA
      INC += -I${CUDA_DIR}/include
      PLIB += $(CUBLAS_LIB)
      SLIB += $(CUBLAS_LIB)
   endif

   PLIB += #-lm -lpthread
   SLIB += #-lm -lpthread

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

OBJECTS = blas_gemm_test.o cublas_gemm_test.o blas_utils.o cublas_utils.o getticks.o

all: refresh

bin: test_driver.x
#
# SERIAL/PARALLEL BLAS BINARIES
#
test_driver.x: test_driver.o $(OBJECTS)
	$(LD) $(LDFLAGS) test_driver.o $(OBJECTS) $(SLIB) -o test_driver_S.x
	$(LD) $(LDFLAGS) test_driver.o $(OBJECTS) $(PLIB) -o test_driver_P.x
#
# OBJECTS
#
test_driver.o: test_driver.c blas_gemm_test.h
	$(CC) $(CFLAGS) $(INC) -c test_driver.c

blas_gemm_test.o: blas_gemm_test.c blas_utils.h
	$(CC) $(CFLAGS) $(INC) -c blas_gemm_test.c

cublas_gemm_test.o: cublas_gemm_test.c cublas_utils.h blas_utils.h
	$(CC) $(CFLAGS) $(INC) -c cublas_gemm_test.c

blas_utils.o: blas_utils.c blas_utils.h
	$(CC) $(CFLAGS) $(INC) -c blas_utils.c

cublas_utils.o: cublas_utils.c cublas_utils.h
	$(CC) $(CFLAGS) $(INC) -c cublas_utils.c

getticks.o: getticks.c
	gcc -c getticks.c

#
# UTILITY OPERATIONS
#
refresh: realclean bin

clean:
	rm -f *.o

realclean: clean
	rm -f *.x *.a

