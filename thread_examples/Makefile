#CC=gcc
#CFLAGS=-g -O2 -fopenmp
#LIBS=-lpthread -lgomp

CC=mpixlc_r
CFLAGS=-g -O3 -qstrict -qsmp=omp
LIBS=-lpthread 

#CC=/soft/apps/current/gcc-4.3.2/gnu-linux/bin/powerpc-bgp-linux-gcc
#CFLAGS=-g -O3 -fopenmp
#LIBS=-lpthread -lgomp

EXE=omp_pthread_test

$(EXE).x : $(EXE).o
	$(CC) $(CFLAGS) -o $(EXE).x $(EXE).o $(LIBS) 

$(EXE).o : $(EXE).c
	$(CC) $(CFLAGS) -c $(EXE).c

clean:
	rm -f *.o 

realclean: clean
	rm -f *.x

submit: $(EXE).x
	qsub --mode smp -n 1 -t 10 --env OMP_NUM_THREADS=4 $(EXE).x
	qsub --mode smp -n 1 -t 10 --env OMP_NUM_THREADS=2 $(EXE).x
	qsub --mode smp -n 1 -t 10 --env OMP_NUM_THREADS=1 $(EXE).x


