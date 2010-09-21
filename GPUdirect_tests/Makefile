CC      = gcc
COPT    = -g

MPI_PATH  = /usr/common/usg/openmpi/1.4.2/pgi
CUDA_PATH = /usr/common/usg/pgi/10.5/linux86-64/2010/cuda/3.0
INCLUDE = -I$(CUDA_PATH)/include -I$(MPI_PATH)/include
LIBS    = -L$(CUDA_PATH)/lib -lcudart -L$(MPI_PATH)/lib -lmpi -lopen-rte -lopen-pal -ldl -lutil -lm -lpthread 

CFLAGS  = $(COPT) $(INCLUDE)

LD      = $(CC)
LDFLAGS = $(COPT) -Wl,-rpath $(CUDA_PATH)/lib -Wl,--export-dynamic

all: allreduce_malloc.x allreduce_memalign.x allreduce_cuda.x

%.x: %.o
	$(LD) $(LDFLAGS) $< $(LIBS) -o $@

allreduce_malloc.o: allreduce.c
	$(CC) $(CFLAGS) -DUSE_MALLOC -c $< -o $@

allreduce_memalign.o: allreduce.c
	$(CC) $(CFLAGS) -DUSE_MEMALIGN -c $< -o $@

allreduce_cuda.o: allreduce.c
	$(CC) $(CFLAGS) -DUSE_CUDA -c $< -o $@

clean:
	$(RM) $(RMFLAGS) *.o

realclean: clean
	$(RM) $(RMFLAGS) *.x

