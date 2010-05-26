CC=gcc
CFLAGS=-g -O2 -fopenmp # -Wall
LIBS=-lpthread -lgomp
EXE=omp_pthread_test

$(EXE) : $(EXE).o
	$(CC) $(CFLAGS) -o $(EXE) $(EXE).o $(LIBS) 

$(EXE).o : $(EXE).c
	$(CC) $(CFLAGS) -c $(EXE).c

clean:
	rm *.o 

realclean: clean
	rm $(EXE) 
