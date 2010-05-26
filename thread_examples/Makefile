#CC=gcc
#CFLAGS=-g -O2 -fopenmp
#LIBS=-lpthread -lgomp
CC=mpixlc_r
CFLAGS=-g -O3 -qstrict -qsmp=omp
LIBS=-lpthread 
EXE=omp_pthread_test

$(EXE) : $(EXE).o
	$(CC) $(CFLAGS) -o $(EXE) $(EXE).o $(LIBS) 

$(EXE).o : $(EXE).c
	$(CC) $(CFLAGS) -c $(EXE).c

clean:
	rm *.o 

realclean: clean
	rm $(EXE) 

submit: $(EXE)
	qsub --mode smp -n 1 -t 10 --env OMP_NUM_THREADS=4 $(EXE)


