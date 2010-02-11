#############################################
#
#               Linux (64-bit)
#
#############################################

   ifeq ($(HOSTNAME),login1.surveyor.alcf.anl.gov)
      BGP=YES
   endif
   ifeq ($(HOSTNAME),login2.surveyor.alcf.anl.gov)
      BGP=YES
   endif
   ifeq ($(HOSTNAME),login3.surveyor.alcf.anl.gov)
      BGP=YES
   endif
   ifeq ($(HOSTNAME),login4.surveyor.alcf.anl.gov)
      BGP=YES
   endif
   ifeq ($(HOSTNAME),login5.surveyor.alcf.anl.gov)
      BGP=YES
   endif
   ifeq ($(HOSTNAME),login6.surveyor.alcf.anl.gov)
      BGP=YES
   endif
   ifeq ($(HOSTNAME),login7.surveyor.alcf.anl.gov)
      BGP=YES
   endif
   ifeq ($(HOSTNAME),login8.surveyor.alcf.anl.gov)
      BGP=YES
   endif
   ifeq ($(BGP),YES)
      TARGET=BGP
      CC=mpicc
      OPT = -O3
      MPI = NO # that is, LIB and INC not needed
      GA = YES
   endif

   # Gadzooks
   ifeq ($(HOSTNAME),login1.gadzooks.alcf.anl.gov)
      TARGET=LINUX64
      CC = gcc
      OPT = -O3
      MPI = YES
      GA = YES
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
      CC = gcc
      OPT = -O3
      MPI = YES
      GA = YES
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
      CC = gcc
      OPT = -O3
      MPI = YES
      GA = YES
   endif

   # Megatron
   ifeq ($(HOSTNAME),megatron)
      TARGET=LINUX64
      CC = gcc
      OPT = -O3
      MPI = YES
      GA = YES
      GA_PREFIX = /software/ga/ga-4-2
      MPI_PREFIX = /software/open-mpi/gnu-build
      MPI_VENDOR = OMPI
   endif

   DEFINES =
   INC =
   LIB =

   ifeq ($(GA),YES)
      DEFINES += -DGA
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

   ifeq ($(TARGET),LINUX64)
      GCC=gcc
   endif
   ifeq ($(TARGET),BGP)
      GCC = $(CC)
      DEFINES += USE_DCMF_TIMEBASE
   endif

   LIB += -lm -lpthread

   COPT = $(OPT)
   CFLAGS = -g -std=c99 $(COPT) $(INC) $(DEFINES)

   LD = $(CC)
   LDFLAGS = -g $(OPT)

#############################################
#
#               End of Targets
#
#############################################

all: refresh bin
# all: bin

bin: comm_bench.x

comm_bench.x: comm_bench.o getticks.o
	$(LD) $(LDFLAGS) comm_bench.o getticks.o $(LIB) -o comm_bench.x

comm_bench.o: comm_bench.c comm_bench.h
	$(CC) $(CFLAGS) $(INC) -c comm_bench.c

getticks.o: getticks.c
	$(GCC) -c getticks.c

refresh: realclean bin

clean:
	rm -f *.o

realclean: clean
	rm -f *.x

