# Choose compiler flags
#
# compilers flags for OpenMP support differ:
# GCC 4.4
OPENMPFLAGS = -fopenmp
# ICC
#OPENMPFLAGS = -openmp

# this should work for Intel and GNU compilers
CXXFLAGS = -g -O3 -ffast-math -Wall $(OPENMPFLAGS)
LDFLAGS += $(OPENMPFLAGS)

# Where are the libraries?
# 
# Default lapack and blas ... broken on most linuxen
#LIBS = -llapack -lblas -lgfortran -lm
# MKL intel library with 64-bit integers and pointers
#LIBS = -L/opt/intel/mkl/10.0.1.014/lib/em64t -lmkl -lguide -lpthread -lm
# Apple's vecLib
LIBS = /System/Library/Frameworks/vecLib.framework/vecLib

scf:	scf.o scfdata.o integ.o matrixutil.o
	$(CXX) -o $@ $^ $(LIBS) $(LDFLAGS)

scf.o:	scfdata.h matrixutil.h

input.o:	scfdata.h integ.h

integ.o:	integ.h

matrixuil.o:	matrixutil.h

clean:
	/bin/rm -f scf *.o *~
