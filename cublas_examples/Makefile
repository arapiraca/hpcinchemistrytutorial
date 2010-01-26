#############################################
#
#               Linux (64-bit)
#
#############################################

   # Lincoln
   ifeq ($(HOSTNAME),$(findstring $(TARGET),honest))
      MKL_DIR = /usr/local/intel/mkl/10.2.2
      CUDA_DIR = /usr/local/cuda-2.2/cuda
      LIB_BITS =
   endif

   # Megatron
   ifeq ($(HOSTNAME),megatron)
      MKL_DIR = /opt/intel/Compiler/11.1/059/mkl
      CUDA_DIR = /opt/cuda
      LIB_BITS = 64
   endif

   PBLAS_LIB = -L${MKL_DIR}/lib/em64t -lguide -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lsvml
   SBLAS_LIB = -L${MKL_DIR}/lib/em64t -lguide -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lsvml
   CUBLAS_LIB = -L${CUDA_DIR}/lib${LIB_BITS} -lcublas
   INC = -I${CUDA_DIR}/include -I${MKL_DIR}/include

   EXTRAS = -lm -lpthread -openmp

   PLIB = $(PBLAS_LIB) $(CUBLAS_LIB) $(EXTRAS)
   SLIB = $(SBLAS_LIB) $(CUBLAS_LIB) $(EXTRAS)

   OPT = -O3 -xT -no-prec-div -funroll-loops

   FC = ifort
   FOPT = $(OPT)
   FFLAGS = -g $(COPT) $(INC)

   CC = icc
   COPT = $(OPT)
   CFLAGS = -g -std=c99 $(COPT) $(INC)

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

