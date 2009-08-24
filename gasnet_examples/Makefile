#############################################
#
#               Linux (64-bit)
#
#############################################

ifeq ($(TARGET),LINUX64)

   GASNET_CONDUIT=smp
   GASNET_PREFIX=/software/gasnet/debug
   GASNET_LIB=-L$(GASNET_PREFIX)/lib -lgasnet-$(GASNET_CONDUIT)-seq -lammpi
   GASNET_INC=-I$(GASNET_PREFIX)/include -I$(GASNET_PREFIX)/include/$(GASNET_CONDUIT)-conduit

   MPI_PREFIX=/software/open-mpi/gnu-build
   MPI_INC=-I$(MPI_PREFIX)/include
   MPI_LIB=-L$(MPI_PREFIX)/lib -lmpi

   OPT=-O0

   EXTRAS=-lm -lpthread

   INC=$(MPI_INC) $(GASNET_INC)

   CC=$(MPI_PREFIX)/bin/mpicc
   CFLAGS=-g $(OPT) $(INC) -Wall

   LD=$(CC)
   LDFLAGS=$(CFLAGS) $(MPI_LIB) $(GASNET_LIB) $(EXTRAS)

endif

#############################################
#
#               BlueGene/P
#
#############################################

ifeq ($(TARGET),BGP)

endif

#############################################
#
#               End of Targets
#
#############################################

OBJECTS=simple.o

simple.x: $(OBJECTS)
	$(LD) $(OBJECTS) $(LDFLAGS) -o simple.x


$%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) $(OBJECTS) simple.x
