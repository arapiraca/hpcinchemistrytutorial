#############################################
#
#               Linux (64-bit)
#
#############################################

   # Breadboard
   ifeq ($(HOSTNAME),bblogin)
      TARGET=LINUX64
      OPT=-O3
      MKL=YES
      MKL_DIR= /soft/intel/11.0.081/mkl
      OPENMP=YES
   endif

   # Eureka
   ifeq ($(HOSTNAME),login1.eureka.alcf.anl.gov)
      TARGET=LINUX64
      OPT=-O3 -xHost -static-intel
      MKL=YES
      MKL_DIR=/soft/apps/intel-cc-11.0.074/mkl
      OPENMP=YES
   endif

   # Gadzooks
   ifeq ($(HOSTNAME),login1.gadzooks.alcf.anl.gov)
      TARGET=LINUX64
      OPT=-O3 -xHost -static-intel
      MKL=YES
      MKL_DIR=/soft/apps/intel-cc-11.0.074/mkl
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
      OPT=-O3 -xSSE2,SSE3,SSSE3,SSE4.1,SSE4.2 -static-intel
      MKL=YES
      MKL_DIR=/soft/intel/11.1.064/mkl
      GOTO=YES
      GOTO_DIR=/fusion/home/jhammond/goto
      GOTO_ARCH=
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
      OPT=-O3 -xT
      MKL=YES
      MKL_DIR=/usr/local/intel/mkl/10.2.2
      OPENMP=YES
   endif

   # Megatron
   ifeq ($(HOSTNAME),megatron)
      TARGET=LINUX64
      OPT=-O3 -xSSE2,SSE3,SSSE3,SSE4.1
      MKL=YES
      MKL_DIR=/opt/intel/Compiler/11.1/059/mkl
      OPENMP=YES
   endif

   DEFINES =
   INC =
   LIB =

   ifeq ($(OPENMP),YES)
      DEFINES+=-DOPENMP
      INC+=-I${MKL_DIR}/../include
      OPT+=-openmp
   endif

   ifeq ($(MKL),YES)
      MKL_INC+=-I${MKL_DIR}/include
      MT_MKL_LIB+=-L${MKL_DIR}/lib/em64t -lguide -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lsvml -lpthread
      ST_MKL_LIB+=-L${MKL_DIR}/lib/em64t -lguide -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lsvml -lpthread
   endif

   ifeq ($(GOTO),YES)
      MT_GOTO_LIB+=-L${GOTO_DIR} -lgoto2_nehalemp-r1.13 -lpthread
      ST_GOTO_LIB+=-L${GOTO_DIR} -lgoto2_nehalem-r1.13 -lpthread
   endif

   NETLIB_LIB=-lblas -lpthread

   FC=ifort
   FOPT=$(OPT)
   FFLAGS=-g $(COPT) $(INC) $(DEFINES)

   CC=icc
   COPT=$(OPT)
   CFLAGS=-g -std=c99 $(COPT) $(INC) $(DEFINES)

   LD=$(CC)
   LDFLAGS=-g -std=c99 $(OPT)

#############################################
#
#               End of Targets
#
#############################################

all: dgemm_perf_PMKL.x dgemm_perf_SMKL.x dgemm_perf_PGOTO.x dgemm_perf_SGOTO.x dgemm_perf_NETLIB.x

dgemm_perf_PMKL.x: dgemm_performance.c
	$(CC) $(CFLAGS) -DPMKL dgemm_performance.c $(MT_MKL_LIB) -o dgemm_perf_PMKL.x

dgemm_perf_SMKL.x: dgemm_performance.c
	$(CC) $(CFLAGS) -DSMKL dgemm_performance.c $(ST_MKL_LIB) -o dgemm_perf_SMKL.x

dgemm_perf_PGOTO.x: dgemm_performance.c
	$(CC) $(CFLAGS) -DPGOTO dgemm_performance.c $(MT_GOTO_LIB) -o dgemm_perf_PGOTO.x

dgemm_perf_SGOTO.x: dgemm_performance.c
	$(CC) $(CFLAGS) -DSGOTO dgemm_performance.c $(ST_GOTO_LIB) -o dgemm_perf_SGOTO.x

dgemm_perf_NETLIB.x: dgemm_performance.c
	$(CC) $(CFLAGS) -DNETLIB dgemm_performance.c $(NETLIB_LIB) -o dgemm_perf_NETLIB.x

refresh: clean bin

clean:
	rm -f *.o *.x *.a

