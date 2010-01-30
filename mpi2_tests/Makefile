QSUB     = cqsub -p Catalyst -t 20 -m smp -n 64
CC       = mpixlc_r
CFLAGS   = -g -O3 -qarch=450d -qtune=450
LDFLAGS  = -g -O3 -qarch=450d -qtune=450
SOURCES  = $(shell ls *.c)
OBJECTS  = $(shell ls *.c | sed s/"\.c"/"\.o"/g )
BINARIES = $(shell ls *.c | sed s/"\.c"/"\.x"/g )
SUBMITS  = $(shell ls *.c | sed s/"\.c"/"\.s"/g )

all: $(BINARIES)

submit: $(SUBMITS)
    
$(SUBMITS): $(BINARIES)
	$(QSUB) $< 

$(BINARIES): $(OBJECTS) 
	$(CC) $(CFLAGS) $< -o $@

$(OBJECTS): $(SOURCES) 
	$(CC) $(CFLAGS) -c $< -o $@

refresh: realclean all

clean:
	$(RM) $(RMFLAGS) *.o

realclean: clean
	$(RM) $(RMFLAGS) *.x

help:
	echo $(SOURCES)
	echo $(OBJECTS)
	echo $(BINARIES)

