CC       = mpixlc_r
CFLAGS   = -g -O3 -qarch=450d -qtune=450
LDFLAGS  = -g -O3 -qarch=450d -qtune=450
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

