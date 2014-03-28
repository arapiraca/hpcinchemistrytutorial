TARGET=BGP

#############################################
#
#               Linux (64-bit)
#
#############################################

ifeq ($(TARGET),LINUX64)

   CPPFLAGS=

   RM = rm
   RMFLAGS = -f

   AR = ar
   ARFLAGS = 

   MPI_PREFIX=/software/open-mpi/gnu-build
   MPI_INC=-I$(MPI_PREFIX)/include
   MPI_LIB=-L$(MPI_PREFIX)/lib -lmpi

   GA_PREFIX=/software/ga/ga-trunk
   GA_INC=-I$(GA_PREFIX)/include
   GA_LIB=-L$(GA_PREFIX)/lib/LINUX64 -lglobal -lma -llinalg -larmci -ltcgmsg-mpi

   GSL_INC=-I/usr/include/gsl
   GSL_LIB=-L/usr/lib64 -lgslcblas -lgsl

   GOTO=-L/software/goto -lgoto
   BLAS=$(GOTO)

   OMP_LIB=-lgomp
   OMP_INC=

   EXTRAS=-lgfortran -lm -lpthread

   LIB=$(GA_LIB) $(MPI_LIB) $(BLAS) $(GSL_LIB) $(EXTRAS) $(OMP_LIB)
   INC=$(GA_INC) $(MPI_INC) $(GSL_INC) $(OMP_INC)

   OPT = -O3 -m64 -malign-double
   OPT+= -march=core2 -mtune=core2
   OPT+= -msse -msse2 -msse3 -mssse3
   OPT+= -fprefetch-loop-arrays -funroll-loops -fvariable-expansion-in-unroller
   OPT+= -ffast-math -mfpmath=sse
   OPT+= --param l2-cache-size=4096 --param l1-cache-size=64
   OPT+= -fopenmp

   CC=$(MPI_PREFIX)/bin/mpicc
   COPT = $(OPT)
   CFLAGS=-g $(COPT) -Wall
   CPP=$(CC) -E
   ifeq ($(DEBUG),YES)
      CPPFLAGS+=-DDEBUG
   endif

   CXX=$(MPI_PREFIX)/bin/mpicxx
   CXXOPT = $(OPT)
   CXXFLAGS=-g $(CXXOPT) -Wall
   CXXPP=$(CXX) -E
   CXXPPFLAGS = $(PPFLAGS)

   LD=$(CC)
   LDFLAGS=-g -O3 -m64 -march=core2 -mtune=core2

endif

#############################################
#
#               BlueGene/P
#
#############################################

ifeq ($(TARGET),BGP)

   CPPFLAGS=

   RM = rm
   RMFLAGS = -f

   BGP_SYS=/bgsys/drivers/ppcfloor

   MPI_PREFIX=$(BGP_SYS)/comm
   MPI_INC=-I$(MPI_PREFIX)/include
   MPI_LIB=-L$(MPI_PREFIX)/lib -lfmpich_.cnk -lmpich.cnk -ldcmf.cnk -ldcmfcoll.cnk -lrt -L$(BGP_SYS)/runtime/SPI -lSPI.cna

   GA_PREFIX=/home/jhammond/ga-4-2
   GA_INC=-I$(GA_PREFIX)/include
   GA_LIB=-L$(GA_PREFIX)/lib/$(TARGET) -lglobal -lma -llinalg -larmci -ltcgmsg-mpi

   ESSL=-L/soft/apps/ESSL-4.4/lib -lesslbg
   BLAS=$(ESSL)
   LAPACK=/soft/apps/LAPACK/liblapack_bgp.a

   OMP_LIB=/opt/ibmcmp/xlsmp/bg/1.7/lib/libxlsmp.a
   OMP_INC=-qsmp=omp

   EXTRAS=/opt/ibmcmp/xlf/bg/11.1/lib/libxlf90_r.a

   LIB=$(LAPACK) $(BLAS) $(EXTRAS) $(OMP_LIB)
   INC=$(OMP_INC)

   OPT=-O5 -qarch=440d -qtune=440 -qthreaded

   CC=$(MPI_PREFIX)/bin/mpixlc_r
   COPT=$(OPT)
   CFLAGS=$(INC) $(COPT)
   CPP=$(CC) -E
   ifeq ($(DEBUG),YES)
      CPPFLAGS+=-DDEBUG
   endif

   CXX=$(MPI_PREFIX)/bin/mpixlcxx_r
   CXXOPT=$(OPT)
   CXXFLAGS=$(INC) $(CXXOPT)
   CXXPP=$(CXX) -E
   CXXPPFLAGS = $(PPFLAGS)

   LD=$(CXX)
   LDFLAGS=-O3 -qarch=440d -qtune=440 -qthreaded

endif

#############################################
#
#               Apple
#
#############################################

ifeq ($(TARGET),APPLE)

   LIBS = /System/Library/Frameworks/vecLib.framework/vecLib

endif

#############################################
#
#               End of Targets
#
#############################################

scf: scf.o scfdata.o integ.o matrixutil.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIB) $(LDFLAGS)

scf.o: scfdata.h matrixutil.h

input.o: scfdata.h integ.h

integ.o: integ.h

matrixuil.o: matrixutil.h

clean:
	/bin/rm -f scf *.o *~
