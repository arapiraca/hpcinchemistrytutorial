#############################################
#
#               BlueGene/P
#
#############################################

   #DEBUG=YES
   #HPM_PROFILING=YES

   BGP_SYS=/bgsys/drivers/ppcfloor

   RM = rm
   RMFLAGS = -f

   CC=mpixlc_r
   COPT=-g -O3 -qarch=450d -qtune=450 -qthreaded -qstrict
   CFLAGS=$(INC) $(COPT)

   ifeq ($(DEBUG),YES)
      CFLAGS+=-DDEBUG
   endif

   ifdef HPM_PROFILING
      LIB += /soft/apps/UPC/lib/libhpm.a
      CFLAGS+=-DHPM_PROFILING
   endif

#############################################
#
#               End of Targets
#
#############################################

all: driver.x rma_test.x am_test1.x

refresh: realclean all

driver.x: driver.o
	$(CC) $(CCFLAGS) driver.o $(LIB) -o driver.x

rma_test.x: rma_test.o
	$(CC) $(CCFLAGS) rma_test.o $(LIB) -o rma_test.x

am_test1.x: am_test1.o
	$(CC) $(CCFLAGS) am_test1.o $(LIB) -o am_test1.x

$%.o: %.c
	$(CC) -c $(CFLAGS) $(INC) $< -o $@

clean:
	$(RM) $(RMFLAGS) *.o

realclean: clean
	$(RM) $(RMFLAGS) *.x *.a

