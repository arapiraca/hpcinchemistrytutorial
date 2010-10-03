CC       = cc
CFLAGS   = 
LDFLAGS  =
BINARIES = $(shell ls *.c | sed s/"\.c"/"\.x"/g )

all: $(BINARIES)

%.x : %.o
	$(CC) $(CFLAGS) $< -o $@

%.o : %.c
	$(CC) $(CFLAGS) -c $< -o $@

refresh: realclean all

clean:
	$(RM) $(RMFLAGS) *.o

realclean: clean
	$(RM) $(RMFLAGS) *.x

