#########################################################
#
# Harvey
#
#CC=xlc_r
#CFLAGS=-O3 -qnoipa -qstrict -qtune=pwr7 -qarch=pwr7 -qsmp=omp
#COPT=-O3 -qtune=pwr7 -qarch=pwr7 -qsmp=omp
#
#LD=$(CC)
#LDFLAGS=-O3 -qnoipa -qstrict -qtune=pwr7 -qarch=pwr7 -qsmp=omp
#
#XL_DIR=/soft/apps/ibmcmp-nov2010
#
#ESSL_LIB=-L$(XL_DIR)/essl/5.1/lib64 -lesslsmp
#MASS_LIB=-L$(XL_DIR)/xlmass/6.1/lib64 -lmass_64
#XLF_LIB=-L$(XL_DIR)/xlf/13.1/lib64 -lxlf90_r
#
#LIBS=$(ESSL_LIB) $(MASS_LIB) $(XLF_LIB) $(OTHER_LIB)
#
#########################################################
#
# Megatron
#

GCCFLAGS=-O3 -m64 -march=core2 -mtune=core2 -msse -msse2 -msse3 -mssse3 -malign-double -funroll-loops -fprefetch-loop-arrays -fvariable-expansion-in-unroller -ffast-math -mfpmath=sse --param l2-cache-size=4096 --param l1-cache-size=64 -fopenmp
#GCCFLAGS=-g -Wall -m64 -O3 -march=core2 -mtune=core2 -ffast-math -mfpmath=sse -fopenmp

CC=gcc
CFLAGS=$(GCCFLAGS)
COPT=$(GCCFLAGS)

LD=$(CC)
LDFLAGS=$(GCCFLAGS)

GOTO_LIB=-L/software/goto -lgoto2_penrynp-r1.13_gnu_2threads -lgomp -lpthread
OTHER_LIB=-lm

LIBS=$(GOTO_LIB) $(OTHER_LIB)

#########################################################
#
# Intrepid or Surveyor
#
# CC=bgxlc_r
# CFLAGS=-O3 -qnoipa -qstrict -qtune=450 -qarch=450d -qsmp=omp 
# COPT=-O5 -qtune=450 -qarch=450d -qsmp=omp
#
# LD=$(CC)
# LDFLAGS=-O3 -qnoipa -qstrict -qtune=450 -qarch=450d -qsmp=omp
#
# ESSL_DIR=/soft/apps/ESSL-4.4.1-1
# XL_DIR=/soft/apps/ibmcmp-dec2010
#
# ESSL_LIB=-L$(ESSL_DIR)/lib -lesslsmpbg
# MASS_LIB=-L$(XL_DIR)/xlmass/bg/4.4/lib -lmass
# XLF_LIB=-L$(XL_DIR)/xlf/bg/11.1/lib -lxlf90_r
#
# OTHER_LIB=-L/bgsys/drivers/ppcfloor/gnu-linux/powerpc-bgp-linux/lib -lrt
#
#LIBS=$(ESSL_LIB) $(MASS_LIB) $(XLF_LIB) $(OTHER_LIB)
#
#########################################################

all: skeleton_ccd.x

%.x: %.o mymath.o
	$(LD) $(LDFLAGS) $< mymath.o $(LIBS) -o $@

%.o: %.c Makefile
	$(CC) $(CFLAGS) -c $< -o $@

mymath.o: mymath.c mymath.h Makefile
	$(CC) $(COPT) -c $< -o $@

refresh: clean all

clean:
	rm -f *.o *.x

