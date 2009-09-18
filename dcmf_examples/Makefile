#############################################
#
#               BlueGene/P
#
#############################################

ifeq ($(TARGET),BGP)

   DEBUG=YES
   HPM_PROFILING=YES

   BGP_SYS=/bgsys/drivers/ppcfloor

   RM = rm
   RMFLAGS = -f

   DCMF_PREFIX=$(BGP_SYS)/comm
   DCMF_INC=-I$(DCMF_PREFIX)/include
   DCMF_LIB=-L$(DCMF_PREFIX)/lib -ldcmf.cnk -ldcmfcoll.cnk -lrt -L$(BGP_SYS)/runtime/SPI -lSPI.cna

   LIB=$(DCMF_LIB)
   INC=$(DCMF_INC)

   CC=bgxlc_r
   CPP=$(CC) -E
   LD=$(CC)
   COPT=-g -O0 -qarch=440d -qtune=440 -qthreaded -qstrict
   CFLAGS=$(INC) $(COPT)
   LFLAGS=-g -O0 -qarch=440d -qtune=440 -qthreaded

   ifeq ($(DEBUG),YES)
      CFLAGS+=-DDEBUG
   endif

   ifdef HPM_PROFILING
      LIB += /soft/apps/UPC/lib/libhpm.a
      CFLAGS+=-DHPM_PROFILING
   endif

endif

#############################################
#
#               End of Targets
#
#############################################

OBJECTS = driver.o

all: driver.x

refresh: realclean all

driver.x: $(OBJECTS)
	$(LD) $(LDFLAGS) $(OBJECTS) $(LIB) -o driver.x

$%.o: %.c
	$(CC) -c $(CFLAGS) $(INC) $< -o $@

clean:
	$(RM) $(RMFLAGS) *.o

realclean: clean
	$(RM) $(RMFLAGS) *.x *.a



