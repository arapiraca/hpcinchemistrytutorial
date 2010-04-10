#############################################
#
#               Linux (64-bit)
#
#############################################

   ifeq ($(HOSTNAME),mist.cs.uoregon.edu)
     MACHINE=MIST
   endif
   ifeq ($(MACHINE),MIST)
      TARGET=LINUX64
      CC=gcc
      OPT=-O3
      MPI=YES
      MPI_PREFIX=/opt/openmpi
      MPI_VENDOR=OMPI
      GA=YES
      GA_PREFIX=/mnt/netapp/home1/jhammond/ga-cvs-dev
   endif

   ifeq ($(HOSTNAME),jaguar14)
     MACHINE=JAGUAR
   endif
   ifeq ($(MACHINE),JAGUAR)
      TARGET=LINUX64
      CC=cc
      OPT=-O3
      MPI=NO
      GA=YES
      GA_PREFIX=/sw/xt5/globalarrays/4.2/cnl2.1_pgi8.0.3_portals/ga-4-2
   endif

   ifeq ($(HOSTNAME),krakenpf11)
     MACHINE=KRAKEN
   endif
   ifeq ($(MACHINE),KRAKEN)
      TARGET=LINUX64
      CC=cc
      OPT=-O3
      MPI=NO
      GA=YES
      GA_PREFIX=/nics/b/home/hammond/ga-cvs-dev
   endif

   ifeq ($(HOSTNAME),cu0login1)
      MACHINE=CHINOOK
   endif
   ifeq ($(HOSTNAME),cu0login2)
      MACHINE=CHINOOK
   endif
   ifeq ($(HOSTNAME),cu0login3)
      MACHINE=CHINOOK
   endif
   ifeq ($(HOSTNAME),cu0login4)
      MACHINE=CHINOOK
   endif
   ifeq ($(HOSTNAME),cu0login5)
      MACHINE=CHINOOK
   endif
   ifeq ($(MACHINE),CHINOOK)
      TARGET=LINUX64
      CC=pathcc
      OPT=-O3
      MPI=YES
      MPI_PREFIX=/apps/mpi/hp/2.03.00.00-20081112
      MPI_VENDOR=HP 
      GA=YES
      GA_PREFIX=/home/jhammond/ga-cvs-dev
   endif

   ifeq ($(HOSTNAME),login1.intrepid.alcf.anl.gov)
      MACHINE=BGP
   endif
   ifeq ($(HOSTNAME),login2.intrepid.alcf.anl.gov)
      MACHINE=BGP
   endif
   ifeq ($(HOSTNAME),login3.intrepid.alcf.anl.gov)
      MACHINE=BGP
   endif
   ifeq ($(HOSTNAME),login4.intrepid.alcf.anl.gov)
      MACHINE=BGP
   endif
   ifeq ($(HOSTNAME),login5.intrepid.alcf.anl.gov)
      MACHINE=BGP
   endif
   ifeq ($(HOSTNAME),login6.intrepid.alcf.anl.gov)
      MACHINE=BGP
   endif
   ifeq ($(HOSTNAME),login7.intrepid.alcf.anl.gov)
      MACHINE=BGP
   endif
   ifeq ($(HOSTNAME),login8.intrepid.alcf.anl.gov)
      MACHINE=BGP
   endif
   ifeq ($(HOSTNAME),login1.surveyor.alcf.anl.gov)
      MACHINE=BGP
   endif
   ifeq ($(HOSTNAME),login2.surveyor.alcf.anl.gov)
      MACHINE=BGP
   endif
   ifeq ($(HOSTNAME),login3.surveyor.alcf.anl.gov)
      MACHINE=BGP
   endif
   ifeq ($(HOSTNAME),login4.surveyor.alcf.anl.gov)
      MACHINE=BGP
   endif
   ifeq ($(HOSTNAME),login5.surveyor.alcf.anl.gov)
      MACHINE=BGP
   endif
   ifeq ($(HOSTNAME),login6.surveyor.alcf.anl.gov)
      MACHINE=BGP
   endif
   ifeq ($(HOSTNAME),login7.surveyor.alcf.anl.gov)
      MACHINE=BGP
   endif
   ifeq ($(HOSTNAME),login8.surveyor.alcf.anl.gov)
      MACHINE=BGP
   endif
   ifeq ($(MACHINE),BGP)
      TARGET=BGP
      CC=mpicc
      OPT=-O2
      MPI=BGP
      GA=YES
      #GA_PREFIX=/gpfs/home/jhammond/ga-cvs-dev
      #GA_PREFIX=/gpfs/home/jhammond/ga-dev-ibm
      GA_PREFIX=/gpfs/home/projects/nwchem/ga-4-3
      ARMCI_PREFIX=/gpfs/home/projects/nwchem/ga-4-3/armci
   endif

   # Eureka
   ifeq ($(HOSTNAME),login1.eureka.alcf.anl.gov)
      MACHINE=EUREKA
   endif
   ifeq ($(MACHINE),EUREKA)
      TARGET=LINUX64
      CC = gcc
      OPT = -O3
      MPI = YES
      MPI_PREFIX = /soft/apps/mpich2-1.2.1
      MPI_VENDOR = MPICH
      GA = YES
      GA_PREFIX = /home/jhammond/eureka/ga-cvs-dev
      EXTRA_LIBS = -L/soft/apps/intel-fc-11.0.074/lib/intel64 -lirc #-lmpl
   endif

   # Gadzooks
   ifeq ($(HOSTNAME),login1.gadzooks.alcf.anl.gov)
      MACHINE=GADZOOKS
   endif
   ifeq ($(MACHINE),GADZOOKS)
      TARGET=LINUX64
      CC=gcc
      OPT=-O3
      MPI=YES
      MPI_PREFIX=/soft/apps/mpich2-1.2.1
      #MPI_PREFIX=/home/jhammond/gadzooks
      MPI_VENDOR=MPICH
      GA=YES
      GA_PREFIX=/home/jhammond/gadzooks/ga-cvs-dev
      EXTRA_LIBS=-L/soft/apps/intel-fc-11.0.074/lib/intel64 -lirc #-lmpl
   endif

   # Fusion
   ifeq ($(HOSTNAME),flogin1.lcrc.anl.gov)
      MACHINE=FUSION
   endif
   ifeq ($(HOSTNAME),flogin2.lcrc.anl.gov)
      MACHINE=FUSION
   endif
   ifeq ($(HOSTNAME),flogin3.lcrc.anl.gov)
      MACHINE=FUSION
   endif
   ifeq ($(HOSTNAME),flogin1)
      MACHINE=FUSION
   endif
   ifeq ($(HOSTNAME),flogin2)
      MACHINE=FUSION
   endif
   ifeq ($(HOSTNAME),flogin3)
      MACHINE=FUSION
   endif
   ifeq ($(MACHINE),FUSION)
      TARGET=LINUX64
      CC=icc
      OPT=-O3
      MPI=YES
      MPI_PREFIX=/soft/mvapich2/1.4-intel-11.1.059
      MPI_VENDOR=MVAPICH
      GA=YES
      GA_PREFIX=/homes/jhammond/ga-cvs-dev
   endif

   # Lincoln
   ifeq ($(HOSTNAME),honest1.ncsa.uiuc.edu)
      MACHINE=LINCOLN
   endif
   ifeq ($(HOSTNAME),honest2.ncsa.uiuc.edu)
      MACHINE=LINCOLN
   endif
   ifeq ($(HOSTNAME),honest3.ncsa.uiuc.edu)
      MACHINE=LINCOLN
   endif
   ifeq ($(HOSTNAME),honest4.ncsa.uiuc.edu)
      MACHINE=LINCOLN
   endif
   ifeq ($(MACHINE),LINCOLN)
      TARGET=LINUX64
      CC=gcc
      OPT=-O3
      MPI=YES
      MPI_PREFIX=/usr/local/mvapich2-1.2-intel-ofed-1.2.5.5/
      MPI_VENDOR=MVAPICH
      EXTRA_LIBS=-L/usr/local/ofed-1.2.5.5-patch/lib64
      GA=YES
      GA_PREFIX=/u/ac/jhammond/ga-cvs-dev
   endif

   # Megatron
   ifeq ($(HOSTNAME),megatron)
      TARGET=LINUX64
      CC=gcc
      OPT=-O3
      MPI=YES
      MPI_PREFIX=/software/open-mpi/gnu-build
      MPI_VENDOR=OMPI
      GA=YES
      GA_PREFIX=/software/ga/ga-4-2
   endif

   DEFINES =
   INC =
   LIB =

   ifeq ($(ARMCI_PREFIX),)
      ARMCI_PREFIX=$(GA_PREFIX)
   endif

   ifeq ($(GA),YES)
      DEFINES+=-DGA
      INC+=-I$(GA_PREFIX)/include
      LIB+=-L$(GA_PREFIX)/lib/$(TARGET) -lglobal -lma -ltcgmsg-mpi
      LIB+=-L$(ARMCI_PREFIX)/lib/$(TARGET) -larmci
   endif

   ifeq ($(MPI),YES)
      DEFINES+=-DMPI
      ifeq ($(MPI_VENDOR),OMPI)
         LIB+=-L$(MPI_PREFIX)/lib -lmpi -lopen-rte -lopen-pal -ldl -lutil
      else
         ifeq ($(MPI_VENDOR),MVAPICH)
            LIB+=-L$(MPI_PREFIX)/lib -lmpich -lrdmacm -libverbs -libumad -L/usr/lib64/ -libverbs -lrt
         else
            ifeq ($(MPI_VENDOR),HP)
               INC+=-I$(MPI_PREFIX)/include/64
               LIB+=-L$(MPI_PREFIX)/lib/linux_amd64 -lmpi -lmpio $(MPI_PREFIX)/lib/linux_amd64/hpmpautodbl_isi8.o $(MPI_PREFIX)/lib/linux_amd64/hpmpautodbl_isr8.o
            else # MPICH
               LIB+=-L$(MPI_PREFIX)/lib -lmpich -lopa -lrt
            endif
         endif
      endif
      INC+=-I$(MPI_PREFIX)/include
   else
      ifeq ($(MPI),BGP)
         DEFINES+=-DMPI
      endif
   endif

   ifeq ($(TARGET),LINUX64)
      GCC=gcc
   endif

   ifeq ($(MACHINE),BGP)
      GCC=$(CC)
      DEFINES+=-DBGP
      DEFINES+=-DHAVE_DCMF_TIMEBASE
      DEFINES+=-DHAVE_BGP_PERSONALITY
      INC+=-I/bgsys/drivers/ppcfloor/arch/include
   endif

   ifeq ($(MACHINE),JAGUAR)
      DEFINES+=-DCRAYXT
      INC+=-I/opt/mpt/default/xt/pmi/include/
      INC+=-I/opt/xt-pe/default/include
      LIB+=-L/opt/xt-pe/default/lib -lportals
      LIB+=-L/opt/xt-pe/default/lib/snos64 -lrca
   endif

   LIB+=-lm -lpthread $(EXTRA_LIBS)

   OPT=-O3
   COPT=$(OPT)
   CFLAGS=-g $(COPT)
   ifeq ($(CC),mpixlc_r)
      CFLAGS+=-qlanglvl=extc99
   else
      ifeq ($(CC),cc)
         CFLAGS+=
      else
         CFLAGS+=-std=c99 
      endif
   endif
   CFLAGS+=$(INC) $(DEFINES)

   LD=$(CC)
   LDFLAGS=-g $(OPT)

#############################################
#
#               End of Targets
#
#############################################

#all: refresh bin
all: bin extra_bin comm_util_$(TARGET).o

bin: torture1_$(TARGET).x torture2_$(TARGET).x torture3_$(TARGET).x comm_bench1_$(TARGET).x comm_bench3_$(TARGET).x comm_bench4_$(TARGET).x mpi3_arg_test_$(TARGET).x

ifeq ($(MACHINE),BGP)
   ifeq ($(CC),mpixlc_r)
      extra_bin: comm_bench2_bgp_$(TARGET).x comm_bench5_$(TARGET).x comm_bench6_$(TARGET).x
   else
      extra_bin:
   endif
else
   ifeq ($(MACHINE),JAGUAR)
      extra_bin: comm_bench2_xt5_$(TARGET).x
   else
      extra_bin:
   endif
endif

torture1_$(TARGET).x: torture1_$(TARGET).o getticks_$(TARGET).o
	$(LD) $(LDFLAGS) torture1_$(TARGET).o getticks_$(TARGET).o $(LIB) -o torture1_$(TARGET).x

torture2_$(TARGET).x: torture2_$(TARGET).o getticks_$(TARGET).o
	$(LD) $(LDFLAGS) torture2_$(TARGET).o getticks_$(TARGET).o $(LIB) -o torture2_$(TARGET).x

torture3_$(TARGET).x: torture3_$(TARGET).o getticks_$(TARGET).o
	$(LD) $(LDFLAGS) torture3_$(TARGET).o getticks_$(TARGET).o $(LIB) -o torture3_$(TARGET).x

comm_bench1_$(TARGET).x: comm_bench1_$(TARGET).o getticks_$(TARGET).o
	$(LD) $(LDFLAGS) comm_bench1_$(TARGET).o getticks_$(TARGET).o $(LIB) -o comm_bench1_$(TARGET).x

comm_bench2_bgp_$(TARGET).x: comm_bench2_bgp_$(TARGET).o getticks_$(TARGET).o
	$(LD) $(LDFLAGS) comm_bench2_bgp_$(TARGET).o getticks_$(TARGET).o $(LIB) -o comm_bench2_bgp_$(TARGET).x

comm_bench2_xt5_$(TARGET).x: comm_bench2_xt5_$(TARGET).o getticks_$(TARGET).o
	$(LD) $(LDFLAGS) comm_bench2_xt5_$(TARGET).o getticks_$(TARGET).o $(LIB) -o comm_bench2_xt5_$(TARGET).x

comm_bench3_$(TARGET).x: comm_bench3_$(TARGET).o getticks_$(TARGET).o
	$(LD) $(LDFLAGS) comm_bench3_$(TARGET).o getticks_$(TARGET).o $(LIB) -o comm_bench3_$(TARGET).x

comm_bench4_$(TARGET).x: comm_bench4_$(TARGET).o getticks_$(TARGET).o
	$(LD) $(LDFLAGS) comm_bench4_$(TARGET).o getticks_$(TARGET).o $(LIB) -o comm_bench4_$(TARGET).x

comm_bench5_$(TARGET).x: comm_bench5_$(TARGET).o getticks_$(TARGET).o
	$(LD) $(LDFLAGS) comm_bench5_$(TARGET).o getticks_$(TARGET).o $(LIB) -o comm_bench5_$(TARGET).x

comm_bench6_$(TARGET).x: comm_bench6_$(TARGET).o getticks_$(TARGET).o
	$(LD) $(LDFLAGS) comm_bench6_$(TARGET).o getticks_$(TARGET).o $(LIB) -o comm_bench6_$(TARGET).x

mpi3_arg_test_$(TARGET).x: mpi3_arg_test_$(TARGET).o getticks_$(TARGET).o
	$(LD) $(LDFLAGS) mpi3_arg_test_$(TARGET).o getticks_$(TARGET).o $(LIB) -o mpi3_arg_test_$(TARGET).x

torture1_$(TARGET).o: torture1.c comm_bench.h
	$(CC) $(CFLAGS) $(INC) -c torture1.c -o torture1_$(TARGET).o

torture2_$(TARGET).o: torture2.c comm_bench.h
	$(CC) $(CFLAGS) $(INC) -c torture2.c -o torture2_$(TARGET).o

torture3_$(TARGET).o: torture3.c comm_bench.h
	$(CC) $(CFLAGS) $(INC) -c torture3.c -o torture3_$(TARGET).o

comm_bench1_$(TARGET).o: comm_bench1.c comm_bench.h
	$(CC) $(CFLAGS) $(INC) -c comm_bench1.c -o comm_bench1_$(TARGET).o

comm_bench2_bgp_$(TARGET).o: comm_bench2_bgp.c comm_bench.h
	$(CC) $(CFLAGS) $(INC) -c comm_bench2_bgp.c -o comm_bench2_bgp_$(TARGET).o

comm_bench2_xt5_$(TARGET).o: comm_bench2_xt5.c comm_bench.h
	$(CC) $(CFLAGS) $(INC) -c comm_bench2_xt5.c -o comm_bench2_xt5_$(TARGET).o

comm_bench3_$(TARGET).o: comm_bench3.c comm_bench.h
	$(CC) $(CFLAGS) $(INC) -c comm_bench3.c -o comm_bench3_$(TARGET).o

comm_bench4_$(TARGET).o: comm_bench4.c comm_bench.h
	$(CC) $(CFLAGS) $(INC) -c comm_bench4.c -o comm_bench4_$(TARGET).o

comm_bench5_$(TARGET).o: comm_bench5.c comm_bench.h
	$(CC) $(CFLAGS) $(INC) -c comm_bench5.c -o comm_bench5_$(TARGET).o

comm_bench6_$(TARGET).o: comm_bench6.c comm_bench.h
	$(CC) $(CFLAGS) $(INC) -c comm_bench6.c -o comm_bench6_$(TARGET).o

comm_util_$(TARGET).o: comm_util.c comm_util.h
	$(CC) $(CFLAGS) $(INC) -c comm_util.c -o comm_util_$(TARGET).o

mpi3_arg_test_$(TARGET).o: mpi3_arg_test.c 
	$(CC) $(CFLAGS) $(INC) -c mpi3_arg_test.c -o mpi3_arg_test_$(TARGET).o

getticks_$(TARGET).o: getticks.c
	$(GCC) -c getticks.c -o getticks_$(TARGET).o

hello_$(TARGET).x: hello.c
	mpicc hello.c -o hello_$(TARGET).x

refresh: realclean bin

clean:
	rm -f *_$(TARGET).o

realclean: clean
	rm -f *_$(TARGET).x

wipe:
ifeq ($(MACHINE),BGP)
	rm core.[01234567890]*
endif

