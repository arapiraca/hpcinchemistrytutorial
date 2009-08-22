MY_ROOT=.
#DEBUG=YES
HPM_PROFILING=YES

#############################################
#
#               Linux (32-bit)
#
#############################################

ifeq ($(TARGET),LINUX32)

endif

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
   ARFLAGS = -r

   MPI_PREFIX=/software/open-mpi/gnu-build
   MPI_INC=-I$(MPI_PREFIX)/include
   MPI_LIB=-L$(MPI_PREFIX)/lib -lmpi

   GA_PREFIX=/software/ga/ga-trunk
   GA_INC=-I$(GA_PREFIX)/include
   #GA_LIB=-L$(GA_PREFIX)/lib/LINUX64 -lglobal -lma -llinalg -larmci -ltcgmsg-mpi
   GA_LIB=-L$(GA_PREFIX)/lib/LINUX64 -lglobal -lma -larmci -ltcgmsg-mpi

   GSL_INC=-I/usr/include/gsl
   GSL_LIB=-L/usr/lib64 -lgslcblas -lgsl

   GOTO=-L/software/goto -lgoto
   BLAS=$(GOTO)

   EXTRAS=-lgomp -lgfortran -lm -lpthread

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

   LIB=-L$(MY_ROOT) $(GA_LIB) $(MPI_LIB) $(GSL_LIB) $(BLAS) $(EXTRAS)
   INC=-I$(MY_ROOT) $(GA_INC) $(MPI_INC) $(GSL_INC)

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

   AR = /bgsys/drivers/ppcfloor/gnu-linux/bin/powerpc-bgp-linux-ar
   ARFLAGS = -r

   BGP_SYS=/bgsys/drivers/ppcfloor

   MPI_PREFIX=$(BGP_SYS)/comm
   MPI_INC=-I$(MPI_PREFIX)/include
   MPI_LIB=-L$(MPI_PREFIX)/lib -lfmpich_.cnk -lmpich.cnk -ldcmf.cnk -ldcmfcoll.cnk -lrt -L$(BGP_SYS)/runtime/SPI -lSPI.cna

   GA_PREFIX=/home/jhammond/ga-4-2
   GA_INC=-I$(GA_PREFIX)/include
   //GA_LIB=-L$(GA_PREFIX)/lib/$(TARGET) -lglobal -lma -llinalg -larmci -ltcgmsg-mpi
   GA_LIB=-L$(GA_PREFIX)/lib/$(TARGET) -lglobal -lma -larmci -ltcgmsg-mpi

   ESSL_PREFIX=/soft/apps/ESSL-4.4
   ESSL_INC=-I$(ESSL_PREFIX)/include
   ESSL_LIB=-L$(ESSL_PREFIX)/lib -lesslbg -lmass

   OMP=/opt/ibmcmp/xlsmp/bg/1.7/lib/libxlsmp.a

   EXTRAS=/opt/ibmcmp/xlf/bg/11.1/lib/libxlf90_r.a

   LIB=-L$(MY_ROOT) $(GA_LIB) $(MPI_LIB) $(ESSL_LIB) $(EXTRAS) $(OMP)
   INC=-I$(MY_ROOT) $(GA_INC) $(MPI_INC) $(ESSL_INC)

   CC=$(MPI_PREFIX)/bin/mpixlc_r
   COPT=-g -O5 -qarch=440d -qtune=440 -qthreaded
   CFLAGS=$(INC) $(COPT) $(CDEF)
   CPP=$(CC) -E
   ifeq ($(DEBUG),YES)
      CPPFLAGS+=-DDEBUG
   endif

   ifdef HPM_PROFILING
      EXTRAS += /soft/apps/UPC/lib/libhpm.a
      CPPFLAGS+=-DHPM_PROFILING
   endif

   LD=$(CC)
   LFLAGS=-g -O3 -qarch=440d -qtune=440 -qthreaded

endif

#############################################
#
#               Custom
#
#############################################

ifeq ($(TARGET),CUSTOM)

endif

#############################################
#
#               End of Targets
#
#############################################

OBJECTS = driver.o hello.o simple.o transpose.o matmul.o matmul2.o matvec.o gemm_test.o

all: driver.x

refresh: realclean all

driver.x: $(OBJECTS)
	$(LD) $(LDFLAGS) $(OBJECTS) $(LIB) -o driver.x

$%.o: %.c
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $(INC) $< -o $@

clean:
	$(RM) $(RMFLAGS) *.o

realclean: clean
	$(RM) $(RMFLAGS) *.x *.a



