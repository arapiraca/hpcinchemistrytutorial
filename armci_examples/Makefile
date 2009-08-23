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
   MPI_INC=$(MPI_PREFIX)/include
   MPI_LIB=$(MPI_PREFIX)/lib -lmpi -lpthread

   ARMCI_PREFIX=/software/ga/ga-trunk
   ARMCI_INC=$(ARMCI_PREFIX)/include
   ARMCI_LIB=$(ARMCI_PREFIX)/lib/LINUX64 -larmci

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

   LIB=-L$(MY_ROOT) -L$(ARMCI_LIB) -L$(MPI_LIB)
   INC=-I$(MY_ROOT) -I$(ARMCI_INC) -I$(MPI_INC)


   CPPFLAGS+=$(INC)

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
   MPI_INC=$(MPI_PREFIX)/include
   MPI_LIB=$(MPI_PREFIX)/lib -lfmpich_.cnk -lmpich.cnk -ldcmf.cnk -ldcmfcoll.cnk -lrt -L$(BGP_SYS)/runtime/SPI -lSPI.cna

   ARMCI_PREFIX=/home/jhammond/ga-4-2
   ARMCI_INC=$(ARMCI_PREFIX)/include
   ARMCI_LIB=$(ARMCI_PREFIX)/lib/$(TARGET) -larmci

   LIB=-L$(MY_ROOT) -L$(ARMCI_LIB) -L$(MPI_LIB)
   INC=-I$(MY_ROOT) -I$(ARMCI_INC) -I$(MPI_INC)

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

   CPPFLAGS+=$(INC)

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

OBJECTS = driver.o simple_get.o simple_put.o simple_acc.o

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



