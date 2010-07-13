#############################################
#
#               Linux (64-bit)
#
#############################################

   ifeq ($(HOSTNAME),krakenpf11)
     MACHINE=KRAKEN
   endif
   ifeq ($(MACHINE),KRAKEN)
      TARGET=LINUX64
      CC=cc
      OPT=-O3
      MPI=NO
      GA=YES
      GA_PREFIX=/nics/b/home/hammond/ga-cvs-dev
   endif

   # Mist
   ifeq ($(HOSTNAME),mist.cs.uoregon.edu)
     MACHINE=MIST
   endif
   ifeq ($(MACHINE),MIST)
      TARGET=LINUX64
      CC=icc
      #CC=tau_cc.sh
      OPT=-O3 -g -std=c99 -xHost
      INTEL_DIR=/usr/local/packages/intel/Compiler/11.1/046
      CUDA=YES
      CUDA_DIR=/opt/cuda/
      LIB_BITS=
      MKL=YES
      MKL_DIR=/usr/local/packages/intel/Compiler/11.1/046/mkl
      MPI=YES
      MPI_VENDOR=OMPI
      MPI_PREFIX=/usr/local/packages/openmpi-1.3.3/intel-11.1.046-tm
      GA=YES
      GA_PREFIX=/mnt/netapp/home1/jhammond/ga-cvs-dev
   endif

   # Breadboard
   ifeq ($(HOSTNAME),bblogin)
      TARGET=LINUX64
      CC=icc
      OPT=-O3 -g -std=c99 -xHost
      INTEL_DIR=/soft/intel/11.0.081
      CUDA=YES
      CUDA_DIR=/usr/local/cuda/
      LIB_BITS=
      MKL=YES
      MKL_DIR=/soft/intel/11.0.081/mkl
      MPI=YES
      MPI_PREFIX=/soft/apps/mpich2-1.2.1
      GA=YES
      GA_PREFIX=/home/jhammond/gadzooks/ga-cvs-dev/
      OPENMP=YES
   endif

   # Eureka
   ifeq ($(HOSTNAME),login1.eureka.alcf.anl.gov)
      TARGET=LINUX64
      CC=icc
      OPT=-O3 -g -std=c99 -xHost -static-intel
      INTEL_DIR=/soft/apps/intel-cc-11.0.074
      CUDA=YES
      CUDA_DIR=/soft/apps/cuda-2.3/cuda
      LIB_BITS=64
      MKL=YES
      MKL_DIR=/soft/apps/intel-cc-11.0.074/mkl
      MPI=YES
      MPI_PREFIX=/soft/apps/mpich2-1.2.1
      GA=YES
      GA_PREFIX=/home/jhammond/eureka/ga-cvs-dev/
      OPENMP=YES
   endif

   # Gadzooks
   ifeq ($(HOSTNAME),login1.gadzooks.alcf.anl.gov)
      TARGET=LINUX64
      CC=icc
      OPT=-O3 -g -std=c99 -xHost -static-intel
      INTEL_DIR=/soft/apps/intel-cc-11.0.074
      CUDA=YES
      CUDA_DIR=/soft/apps/cuda-2.3/cuda
      LIB_BITS=64
      MKL=YES
      MKL_DIR=/soft/apps/intel-cc-11.0.074/mkl
      MPI=YES
      MPI_PREFIX=/soft/apps/mpich2-1.2.1
      GA=YES
      GA_PREFIX=/home/jhammond/gadzooks/ga-cvs-dev/
      OPENMP=YES
   endif

   # Fusion
   ifeq ($(HOSTNAME),flogin1.lcrc.anl.gov)
      FUSION=YES
   endif
   ifeq ($(HOSTNAME),flogin2.lcrc.anl.gov)
      FUSION=YES
   endif
   ifeq ($(HOSTNAME),flogin3.lcrc.anl.gov)
      FUSION=YES
   endif
   ifeq ($(HOSTNAME),flogin1)
      FUSION=YES
   endif
   ifeq ($(HOSTNAME),flogin2)
      FUSION=YES
   endif
   ifeq ($(HOSTNAME),flogin3)
      FUSION=YES
   endif
   ifeq ($(FUSION),YES)
      TARGET=LINUX64
      CC=icc
      OPT=-O3 -g -std=c99 -xHost
      INTEL_DIR=
      CUDA=NO
      MKL=YES
      MKL_DIR=/soft/mkl/10.2.2.025
      MPI=YES
      MPI_PREFIX=/soft/mvapich2/1.4-intel-11.1.059
      MPI_VENDOR=MVAPICH
      GA=YES
      GA_PREFIX=/homes/jhammond/ga-cvs-dev
      OPENMP=YES
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
      CC=icc
      OPT=-O3 -g -std=c99 -xHost -xT
      INTEL_DIR=
      CUDA=YES
      CUDA_DIR=/usr/local/cuda-2.2/cuda
      LIB_BITS=
      MKL=YES
      MKL_DIR=/usr/local/intel/mkl/10.2.2
      MPI=YES
      MPI_PREFIX=/usr/local/mvapich2-1.2-intel-ofed-1.2.5.5/
      GA=YES
      GA_PREFIX=/u/ac/jhammond/ga-cvs-dev/
      OPENMP=YES
   endif

   # Lincoln
   ifeq ($(HOSTNAME),cvrsvc01)
      MACHINE=DIRAC
   endif
   ifeq ($(HOSTNAME),cvrsvc02)
      MACHINE=DIRAC
   endif
   ifeq ($(HOSTNAME),cvrsvc03)
      MACHINE=DIRAC
   endif
   ifeq ($(HOSTNAME),cvrsvc04)
      MACHINE=DIRAC
   endif
   ifeq ($(HOSTNAME),cvrsvc05)
      MACHINE=DIRAC
   endif
   ifeq ($(MACHINE),DIRAC)
      TARGET=LINUX64
      CC=gcc
      OPT=-g -O3
      CUDA=YES
      #CUDA_DIR=/usr/common/usg/nvidia-driver-util/3.1b
      #CUDA_DIR=/usr/common/usg/nvidia-driver-util/3.0
      #LIB_BITS=64
      CUDA_DIR=/usr/common/usg/pgi/10.5/linux86-64/2010/cuda/3.0
      LIB_BITS=
      MKL=YES
      MKL_DIR=/usr/common/usg/mkl/10.2.2.025
      MPI=YES
      MPI_PREFIX=/usr/common/usg/openmpi/1.4.2/pgi
      MPI_VENDOR=OMPI
      GA=YES
      GA_PREFIX=/global/homes/j/jhammond/GA/ga-4-3
      OPENMP=NO
   endif

   # Megatron
   ifeq ($(HOSTNAME),megatron)
      TARGET=LINUX64
      CC=gcc
      OPT=-g -O3 -Wall
      OMP_VENDOR=GNU
      #OMP_DIR=/opt/intel/Compiler/11.1/059
      OMP_DIR=/usr/lib64/gcc/x86_64-suse-linux/4.3
      CUDA=YES
      CUDA_DIR=/opt/cuda
      LIB_BITS=64
      MKL=NO
      #MKL_DIR=/opt/intel/Compiler/11.1/059/mkl
      MPI=YES
      MPI_PREFIX=/software/mpich/mpich2-1.3a1-gnu-build
      MPI_VENDOR=MPICH
      GA=YES
      GA_PREFIX=/software/ga/ga-4-3
      OPENMP=YES
   endif

   DEFINES=
   INC=
   LIB=
   LDFLAGS=

   ifeq ($(BIGTESTS),YES)
      DEFINES+=-DBIGTESTS
   endif 

   ifeq ($(OPENMP),YES)
      ifeq ($(OMP_VENDOR),INTEL)
         LIB+=-openmp
      endif
      ifeq ($(OMP_VENDOR),GNU)
         LIB+=-L$(OMP_DIR) -lgomp
      endif
      DEFINES+=-DOPENMP
      INC+=-I$(OMP_DIR)/include
   endif

   ifeq ($(CUDA),YES)
      CUBLAS_LIB=-L$(CUDA_DIR)/lib$(LIB_BITS) -lcublas -lcudart
      DEFINES+=-DCUDA
      INC+=-I$(CUDA_DIR)/include 
      INC+=-I$(CUDA_DIR)/include/cuda
      LIB+=$(CUBLAS_LIB)
      LDFLAGS+=-Wl,-rpath $(CUDA_DIR)/lib$(LIB_BITS)
   endif

   ifeq ($(GA),YES)
      DEFINES+=-DGA -DARMCI # -DARMCI_MALLOC
      INC+=-I$(GA_PREFIX)/include
      LIB+=-L$(GA_PREFIX)/lib/$(TARGET) -lglobal -lma -larmci -ltcgmsg-mpi
   endif

   ifeq ($(MPI),YES)
      ifeq ($(MPI_VENDOR),OMPI)
         LIB+=-L$(MPI_PREFIX)/lib -lmpi -lopen-rte -lopen-pal -ldl -lutil
      endif
      ifeq ($(MPI_VENDOR),MVAPICH)
         LIB+=-L$(MPI_PREFIX)/lib -lmpich -lrdmacm -libverbs -libumad -L/usr/lib64/ -libverbs -lrt
         LDFLAGS+=-Wl,-rpath  $(MPI_PREFIX)/lib
      endif
      ifeq ($(MPI_VENDOR),HP)
            INC+=-I$(MPI_PREFIX)/include/64
            LIB+=-L$(MPI_PREFIX)/lib/linux_amd64 -lmpi -lmpio $(MPI_PREFIX)/lib/linux_amd64/hpmpautodbl_isi8.o $(MPI_PREFIX)/lib/linux_amd64/hpmpautodbl_isr8.o
      endif
      ifeq ($(MPI_VENDOR),MPICH)
            LIB+=-L$(MPI_PREFIX)/lib -lmpich -lmpl -lopa -lrt
            LDFLAGS+=-Wl,-rpath  $(MPI_PREFIX)/lib
      endif
      DEFINES+=-DMPI
      INC+=-I$(MPI_PREFIX)/include
   endif

   ifeq ($(MKL),YES)
      DEFINES+=-DMKL
      INC+=-I$(MKL_DIR)/include
      PBLAS+=-L$(MKL_DIR)/lib/em64t -lguide -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread
      SBLAS+=-L$(MKL_DIR)/lib/em64t -lguide -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread
      LDFLAGS+=-Wl,-rpath  $(MKL_DIR)/lib/em64t
      ifeq ($(CC),icc)
         PBLAS+=-lsvml
         SBLAS+=-lsvml
      endif
   else
      DEFINES+=-DNETLIB
      PBLAS+=-L/usr/lib64 -llapack -lblas -lpthread
      SBLAS+=-L/usr/lib64 -llapack -lblas -lpthread
   endif

   LIB+=-lm -lpthread

   CFLAGS=$(OPT) $(INC) $(DEFINES)
   LD=$(CC)
   LDFLAGS+=$(OPT) -Wl,--export-dynamic

#############################################
#
#               End of Targets
#
#############################################

BINARY=test_driver.x test_driver2.x gemm_test2.x ga_cpu_sgemm.x new_ga_cpu_sgemm.x new_armci_cpu_sgemm.x
ifeq ($(CUDA),YES)
BINARY+=mpicoll_gpu_sgemm.x armci_gpu_sgemm.x ga_gpu_sgemm.x ga_gpu_dgemm.x cuda_transfer_benchmark.x
endif
# BINARY=ga_gpu_sgemm.x

OBJECTS=blas_gemm_test.o blas_gemm_test2.o cublas_gemm_test.o cublas_gemm_test2.o ga_utils.o blas_utils.o cublas_utils.o getticks.o

#all: refresh bin
all: bin

bin: $(BINARY)
#
# SERIAL/PARALLEL BLAS BINARIES
#
test_driver.x: test_driver.o $(OBJECTS)
	$(LD) $(LDFLAGS) test_driver.o $(OBJECTS) $(LIB) $(SBLAS) -o test_driver_S.x
	$(LD) $(LDFLAGS) test_driver.o $(OBJECTS) $(LIB) $(PBLAS) -o test_driver_P.x

test_driver2.x: test_driver2.o $(OBJECTS)
	$(LD) $(LDFLAGS) test_driver2.o $(OBJECTS) $(LIB) $(SBLAS) -o test_driver2_S.x
	$(LD) $(LDFLAGS) test_driver2.o $(OBJECTS) $(LIB) $(PBLAS) -o test_driver2_P.x

dmp_gemm_driver.x: dmp_gemm_driver.o $(OBJECTS)
	$(LD) $(LDFLAGS) dmp_gemm_driver.o $(OBJECTS) $(LIB) $(PBLAS) -o dmp_gemm_driver.x

armci_gpu_sgemm.x: armci_gpu_sgemm.o $(OBJECTS)
	$(LD) $(LDFLAGS) armci_gpu_sgemm.o $(OBJECTS) $(LIB) $(PBLAS) -o armci_gpu_sgemm.x

ga_gpu_dgemm.x: ga_gpu_dgemm.o $(OBJECTS)
	$(LD) $(LDFLAGS) ga_gpu_dgemm.o $(OBJECTS) $(LIB) $(PBLAS) -o ga_gpu_dgemm.x

new_ga_cpu_sgemm.x: new_ga_cpu_sgemm.o $(OBJECTS)
	$(LD) $(LDFLAGS) new_ga_cpu_sgemm.o $(OBJECTS) $(LIB) $(SBLAS) -o new_ga_cpu_sgemm.x

new_armci_cpu_sgemm.x: new_armci_cpu_sgemm.o $(OBJECTS)
	$(LD) $(LDFLAGS) new_armci_cpu_sgemm.o $(OBJECTS) $(LIB) $(SBLAS) -o new_armci_cpu_sgemm.x

ga_cpu_sgemm.x: ga_cpu_sgemm.o $(OBJECTS)
	$(LD) $(LDFLAGS) ga_cpu_sgemm.o $(OBJECTS) $(LIB) $(SBLAS) -o ga_cpu_sgemm.x

ga_gpu_sgemm.x: ga_gpu_sgemm.o $(OBJECTS)
	$(LD) $(LDFLAGS) ga_gpu_sgemm.o $(OBJECTS) $(LIB) $(SBLAS) -o ga_gpu_sgemm.x

mpicoll_gpu_sgemm.x: mpicoll_gpu_sgemm.o $(OBJECTS)
	$(LD) $(LDFLAGS) mpicoll_gpu_sgemm.o $(OBJECTS) $(LIB) $(PBLAS) -o mpicoll_gpu_sgemm.x

cuda_transfer_benchmark.x: cuda_transfer_benchmark.o $(OBJECTS)
	$(LD) $(LDFLAGS) cuda_transfer_benchmark.o $(OBJECTS) $(LIB) $(PBLAS) -o cuda_transfer_benchmark.x

gemm_test2.x: gemm_test2.o blas_utils.o
	$(LD) $(LDFLAGS) gemm_test2.o blas_utils.o $(LIB) $(PBLAS) -o gemm_test2.x
#
#
# OBJECTS
#
armci_gpu_sgemm.o: armci_gpu_sgemm.c
	$(CC) $(CFLAGS) $(INC) -c armci_gpu_sgemm.c

ga_gpu_dgemm.o: ga_gpu_dgemm.c
	$(CC) $(CFLAGS) $(INC) -c ga_gpu_dgemm.c

ga_cpu_sgemm.o: ga_cpu_sgemm.c
	$(CC) $(CFLAGS) $(INC) -c ga_cpu_sgemm.c

ga_gpu_sgemm.o: ga_gpu_sgemm.c
	$(CC) $(CFLAGS) $(INC) -c ga_gpu_sgemm.c

mpicoll_gpu_sgemm.o: mpicoll_gpu_sgemm.c
	$(CC) $(CFLAGS) $(INC) -c mpicoll_gpu_sgemm.c

cuda_transfer_benchmark.o: cuda_transfer_benchmark.c
	$(CC) $(CFLAGS) $(INC) -c cuda_transfer_benchmark.c

dmp_gemm_driver.o: dmp_gemm_driver.c blas_gemm_test.h
	$(CC) $(CFLAGS) $(INC) -c dmp_gemm_driver.c

test_driver2.o: test_driver2.c blas_gemm_test.h
	$(CC) $(CFLAGS) $(INC) -c test_driver2.c

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

tauclean:
	rm -f *.pdb profile.* *.inst.*

realclean: clean tauclean
	rm -f *.x
