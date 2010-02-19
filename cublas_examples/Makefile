#############################################
#
#               Linux (64-bit)
#
#############################################

   # Gadzooks
   ifeq ($(HOSTNAME),login1.gadzooks.alcf.anl.gov)
      TARGET=LINUX64
      OPT = -O3 -xHost -static-intel
      INTEL_DIR = /soft/apps/intel-cc-11.0.074
      MKL_DIR = /soft/apps/intel-cc-11.0.074/mkl
      CUDA_DIR = /soft/apps/cuda-2.3/cuda
      LIB_BITS = 64
      CUDA = YES
      MKL = YES
      OPENMP = YES
      MPI = YES
      GA = YES
      BIGTESTS = YES
   endif

   # Fusion
   ifeq ($(HOSTNAME),flogin1.lcrc.anl.gov)
      FUSION = YES
   endif
   ifeq ($(HOSTNAME),flogin2.lcrc.anl.gov)
      FUSION = YES
   endif
   ifeq ($(HOSTNAME),flogin3.lcrc.anl.gov)
      FUSION = YES
   endif
   ifeq ($(HOSTNAME),flogin1)
      FUSION = YES
   endif
   ifeq ($(HOSTNAME),flogin2)
      FUSION = YES
   endif
   ifeq ($(HOSTNAME),flogin3)
      FUSION = YES
   endif
   ifeq ($(FUSION),YES)
      TARGET=LINUX64
      OPT = -O3 -xSSE2,SSE3,SSSE3,SSE4.1,SSE4.2
      INTEL_DIR = 
      MKL_DIR = /soft/mkl/10.2.2.025
      CUDA_DIR =
      LIB_BITS =
      CUDA = NO
      MKL = YES
      OPENMP = YES
      MPI = YES
      GA = YES
      BIGTESTS = YES
   endif

   # Lincoln
   ifeq ($(HOSTNAME),honest1.ncsa.uiuc.edu)
      LINCOLN=YES
   endif
   ifeq ($(HOSTNAME),honest2.ncsa.uiuc.edu)
      LINCOLN=YES
   endif
   ifeq ($(HOSTNAME),honest3.ncsa.uiuc.edu)
      LINCOLN=YES
   endif
   ifeq ($(HOSTNAME),honest4.ncsa.uiuc.edu)
      LINCOLN=YES
   endif
   ifeq ($(LINCOLN),YES)
      TARGET=LINUX64
      OPT = -O3 -xT
      INTEL_DIR = 
      MKL_DIR = /usr/local/intel/mkl/10.2.2
      CUDA_DIR = /usr/local/cuda-2.2/cuda
      LIB_BITS =
      CUDA = YES
      MKL = YES
      OPENMP = YES
      MPI = YES
      GA = YES
      BIGTESTS = YES
   endif

   # Megatron
   ifeq ($(HOSTNAME),megatron)
      TARGET=LINUX64
      OPT = -O3 -xSSE2,SSE3,SSSE3,SSE4.1
      INTEL_DIR = /opt/intel/Compiler/11.1/059
      MKL_DIR = /opt/intel/Compiler/11.1/059/mkl
      CUDA_DIR = /opt/cuda
      MPI_PREFIX=/software/open-mpi/gnu-build
      MPI_VENDOR=OMPI
      GA_PREFIX=/software/ga/ga-4-2
      LIB_BITS = 64
      CUDA = YES
      MKL = YES
      OPENMP = YES
      MPI = YES
      GA = YES
      BIGTESTS = NO
   endif

   DEFINES =
   INC =
   LIB =

   ifeq ($(BIGTESTS),YES)
      DEFINES += -DBIGTESTS
   endif 

   ifeq ($(OPENMP),YES)
      DEFINES += -DOPENMP
      INC += -I$(INTEL_DIR)/include
      LIB += -openmp
   endif

   ifeq ($(CUDA),YES)
      CUBLAS_LIB = -L${CUDA_DIR}/lib${LIB_BITS} -lcublas
      DEFINES += -DCUDA
      INC += -I${CUDA_DIR}/include
      LIB += $(CUBLAS_LIB)
   endif

   ifeq ($(GA),YES)
      DEFINES += -DGA -DARMCI
      INC += -I$(GA_PREFIX)/include
      LIB += -L$(GA_PREFIX)/lib/$(TARGET) -lglobal -lma -larmci -ltcgmsg-mpi
   endif

   ifeq ($(MPI),YES)
      DEFINES += -DMPI
      ifeq ($(MPI_VENDOR),OMPI)
         LIB += -L$(MPI_PREFIX)/lib -lmpi -lopen-rte -lopen-pal -ldl -lutil
      else
         LIB += -L$(MPI_PREFIX)/lib -lmpich  -lopa -lrt
      endif
      INC += -I$(MPI_PREFIX)/include
   endif

   ifeq ($(MKL),YES)
      DEFINES += -DMKL
      INC += -I${MKL_DIR}/include
      PBLAS += -L${MKL_DIR}/lib/em64t -lguide -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lsvml
      SBLAS += -L${MKL_DIR}/lib/em64t -lguide -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lsvml
   endif

   LIB += -lm -lpthread

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

OBJECTS = gagpu_gemm.o blas_gemm_test.o blas_gemm_test2.o cublas_gemm_test.o cublas_gemm_test2.o ga_utils.o blas_utils.o cublas_utils.o getticks.o

#all: refresh bin
all: bin

bin: test_driver.x armci_gpu_sgemm.x gemm_test2.x
#
# SERIAL/PARALLEL BLAS BINARIES
#
test_driver.x: test_driver.o $(OBJECTS)
	$(LD) $(LDFLAGS) test_driver.o $(OBJECTS) $(LIB) $(SBLAS) -o test_driver_S.x
	$(LD) $(LDFLAGS) test_driver.o $(OBJECTS) $(LIB) $(PBLAS) -o test_driver_P.x

dmp_gemm_driver.x: dmp_gemm_driver.o $(OBJECTS)
	$(LD) $(LDFLAGS) dmp_gemm_driver.o $(OBJECTS) $(LIB) $(PBLAS) -o dmp_gemm_driver.x

armci_gpu_sgemm.x: armci_gpu_sgemm.o $(OBJECTS)
	$(LD) $(LDFLAGS) armci_gpu_sgemm.o $(OBJECTS) $(LIB) $(PBLAS) -o armci_gpu_sgemm.x

gemm_test2.x: gemm_test2.o $(OBJECTS)
	$(LD) $(LDFLAGS) gemm_test2.o $(OBJECTS) $(LIB) $(PBLAS) -o gemm_test2.x
#
#
# OBJECTS
#
armci_gpu_sgemm.o: armci_gpu_sgemm.c
	$(CC) $(CFLAGS) $(INC) -c armci_gpu_sgemm.c

dmp_gemm_driver.o: dmp_gemm_driver.c blas_gemm_test.h
	$(CC) $(CFLAGS) $(INC) -c dmp_gemm_driver.c

test_driver.o: test_driver.c blas_gemm_test.h
	$(CC) $(CFLAGS) $(INC) -c test_driver.c

gagpu_gemm.o: gagpu_gemm.c ga_utils.h cublas_utils.h
	$(CC) $(CFLAGS) $(INC) -c gagpu_gemm.c

blas_gemm_test.o: blas_gemm_test.c blas_utils.h
	$(CC) $(CFLAGS) $(INC) -c blas_gemm_test.c

blas_gemm_test2.o: blas_gemm_test2.c blas_utils.h
	$(CC) $(CFLAGS) $(INC) -c blas_gemm_test2.c

cublas_gemm_test.o: cublas_gemm_test.c cublas_utils.h blas_utils.h
	$(CC) $(CFLAGS) $(INC) -c cublas_gemm_test.c

cublas_gemm_test2.o: cublas_gemm_test2.c cublas_utils.h blas_utils.h
	$(CC) $(CFLAGS) $(INC) -c cublas_gemm_test2.c

gemm_test2.o: gemm_test2.c
	$(CC) $(CFLAGS) $(INC) -c gemm_test2.c

ga_utils.o: ga_utils.c ga_utils.h
	$(CC) $(CFLAGS) $(INC) -c ga_utils.c

blas_utils.o: blas_utils.c blas_utils.h
	$(CC) $(CFLAGS) $(INC) -c blas_utils.c

cublas_utils.o: cublas_utils.c cublas_utils.h
	$(CC) $(CFLAGS) $(INC) -c cublas_utils.c

getticks.o: getticks.c
	gcc -c getticks.c

cpuid.o: cpuid.c
	gcc -c cpuid.c

#
# UTILITY OPERATIONS
#
refresh: realclean bin

clean:
	rm -f *.o

realclean: clean
	rm -f *.x *.a

