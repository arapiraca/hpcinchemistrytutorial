
CXXFLAGS = -g -O3 -ffast-math -Wall

# Default lapack and blas ... broken on most linuxen
#LIBS = -llapack -lblas -lgfortran -lm

# MKL intel library with 64-bit integers and pointers
LIBS = -L/opt/intel/mkl/10.0.1.014/lib/em64t -lmkl -lguide -lpthread -lm


scf:	scf.o scfdata.o integ.o matrixutil.o
	$(CXX) -o $@ $^ $(LIBS)

scf.o:	scfdata.h matrixutil.h

input.o:	scfdata.h integ.h

integ.o:	integ.h

matrixuil.o:	matrixutil.h

clean:
	/bin/rm -f scf *.o *~
