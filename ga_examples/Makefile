MY_ROOT=.
DEBUG=YES

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

   OPT = -O3 -m64 -malign-double
   OPT+= -march=core2 -mtune=core2
   OPT+= -msse -msse2 -msse3 -mssse3
   OPT+= -fprefetch-loop-arrays -funroll-loops -fvariable-expansion-in-unroller
   OPT+= -ffast-math -mfpmath=sse
   OPT+= --param l2-cache-size=4096 --param l1-cache-size=64

   CC=$(MPI_PREFIX)/bin/mpicc
   COPT = $(OPT)

   CXX=$(MPI_PREFIX)/bin/mpicxx
   CXXOPT = $(OPT)

   ifeq ($(DEBUG),YES)
      PPFLAGS=-DDEBUG
   else
      PPFLAGS=
   endif

   CPP=$(CC) -E
   CPPFLAGS = $(PPFLAGS)

   CXXPP=$(CXX) -E
   CXXPPFLAGS = $(PPFLAGS)

   LD=$(CC)
   LDFLAGS=-O3 -m64 -march=core2 -mtune=core2

   GOTO=-L/software/goto -lgoto
   BLAS=$(GOTO)

   EXTRAS=-lgfortran -lm -lpthread

   LIB=-L$(MY_ROOT) $(GA_LIB) $(MPI_LIB) $(BLAS) $(EXTRAS)
   INC=-I$(MY_ROOT) $(GA_INC) $(MPI_INC)

   CFLAGS=$(INC) $(COPT) -Wall
   CXXFLAGS=$(INC) $(CXXOPT) -Wall
   LDFLAGS=$(LIB)

endif

#############################################
#
#               BlueGene/P
#
#############################################

ifeq ($(TARGET),BGP)

   RM = rm
   RMFLAGS = -f

   BGP_SYS=/bgsys/drivers/ppcfloor

   MPI_PREFIX=$(BGP_SYS)/comm
   MPI_INC=-I$(MPI_PREFIX)/include
   MPI_LIB=-L$(MPI_PREFIX)/lib -lfmpich_.cnk -lmpich.cnk -ldcmf.cnk -ldcmfcoll.cnk -lrt -L$(BGP_SYS)/runtime/SPI -lSPI.cna

   GA_PREFIX=$(BGP_SYS)/comm
   GA_INC=-I$(GA_PREFIX)/include
   GA_LIB=-L$(GA_PREFIX)/lib -lglobal -lma -llinalg -larmci -ltcgmsg-mpi

   CC=$(MPI_PREFIX)/bin/mpixlc_r
   COPT=-O5 -qarch=440d -qtune=440

   CPP=$(CC) -E
   ifeq ($(DEBUG),YES)
      CPPFLAGS=-DDEBUG
   else
      CPPFLAGS=
   endif

   LD=$(CC)
   LDFLAGS=-O5 -qstrict -qarch=440 -qtune=440

   ESSL=-L/soft/apps/ESSL-4.4/lib -lesslsmpbg
   BLAS=$(ESSL)

   EXTRAS=-lgfortran

   LIB=-L$(MY_ROOT) $(GA_LIB) $(MPI_LIB) $(BLAS) $(EXTRAS)
   INC=-I$(MY_ROOT) $(GA_INC) $(MPI_INC)

   CFLAGS=$(INC) $(COPT) $(CDEF)
   LDFLAGS=$(LIB)

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

all: driver.x

refresh: realclean all

driver.x: driver.o test1.o test2.o
	$(LD) $(LDFLAGS) driver.o libtests.a $(LDFLAGS) -o driver.x

test1.o: test1.c
	$(CC) -c $(CFLAGS) $(CPPFLAGS) test1.c -o test1.o
	$(AR) -r libtests.a test1.o
	$(RM) $(RMFLAGS) test1.o

test2.o: test2.c
	$(CC) -c $(CFLAGS) $(CPPFLAGS) test2.c -o test2.o
	$(AR) -r libtests.a test2.o
	$(RM) $(RMFLAGS) test2.o

clean:
	$(RM) $(RMFLAGS) *.o

realclean: clean
	$(RM) $(RMFLAGS) *.x *.a



