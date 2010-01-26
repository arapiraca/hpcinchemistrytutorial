#############################################
#
#               Linux (64-bit)
#
#############################################

   #BLAS_LIB = -L/usr/local/intel/mkl/10.2.2/lib/em64t -lguide -lmkl_intel_ilp64 -lmkl_core -lmkl_intel_thread -lsvml
   BLAS_LIB = -L/usr/local/intel/mkl/10.2.2/lib/em64t -lguide -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lsvml
   CUBLAS_LIB = -L/usr/local/cuda-2.2/cuda/lib -lcublas

   EXTRAS = -lm -lpthread -openmp

   LIB = $(BLAS_LIB) $(CUBLAS_LIB) $(EXTRAS)
   INC = -I/usr/local/cuda-2.2/cuda/include -I/usr/local/intel/mkl/10.2.2/include

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

OBJECTS = simpleCUBLAS.o getticks.o 

all: simpleCUBLAS.x

refresh: realclean all

simpleCUBLAS.x: $(OBJECTS)
	$(LD) $(LDFLAGS) $(OBJECTS) $(LIB) -o simpleCUBLAS.x
	rm -f *.o

simpleCUBLAS.o: simpleCUBLAS.c
	$(CC) -c $(CFLAGS) $(INC) simpleCUBLAS.c

getticks.o: getticks.c
	gcc -c getticks.c

clean:
	rm -f *.o

realclean: clean
	rm -f *.x *.a
