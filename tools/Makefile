#############################################
#
#               Linux (64-bit)
#
#############################################

   ifeq ($(HOSTNAME),krakenpf11)
      MACHINE=KRAKEN
   endif
   ifeq ($(MACHINE),KRAKEN)
      TARGET=LINUX64
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
   endif

   # Gadzooks
   ifeq ($(HOSTNAME),login1.gadzooks.alcf.anl.gov)
      MACHINE=GADZOOKS
   endif
   ifeq ($(MACHINE),GADZOOKS)
      TARGET=LINUX64
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
   endif

   # Megatron
   ifeq ($(HOSTNAME),megatron)
      TARGET=LINUX64
      DEFINES = -DLINUX64
   endif

   ifeq ($(TARGET),BGP)
      CC=powerpc-bgp-linux-gcc
      AR=powerpc-bgp-linux-ar
      DEFINES = -DHAVE_DCMF_TIMEBASE -DBGP
   else
      CC=gcc
      AR=ar
   endif

   FLAGS=-g -Wall -Werror

#############################################
#
#               End of Targets
#
#############################################

all: lib clean

lib: getticks_$(TARGET).o aligned_malloc_$(TARGET).o
	$(AR) -r libtools_$(TARGET).a getticks_$(TARGET).o aligned_malloc_$(TARGET).o

getticks_$(TARGET).o: getticks.c
	$(CC) $(FLAGS) -c getticks.c -o getticks_$(TARGET).o

aligned_malloc_$(TARGET).o: aligned_malloc.c
	$(CC) $(FLAGS) -c aligned_malloc.c -o aligned_malloc_$(TARGET).o

clean:
	rm -f *_$(TARGET).o

realclean: clean
	rm -f *_$(TARGET).a

