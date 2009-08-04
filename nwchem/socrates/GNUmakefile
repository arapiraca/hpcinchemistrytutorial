#
# If you put any source code in sub-directories, list them here.
#
SUBDIRS = 

#
# OBJ files are not optimized heavily.
#
# The effect of optimization upon the speed of compilation
# and correctness of code depends greatly on the platform
# and compiler used.
#
OBJ = socrates_input.o \
      socrates_driver.o

#
# OBJ_OPTIMIZE files are optimized heavily.  Any code with
# loops should go here.
#
OBJ_OPTIMIZE = socrates_util.o \
               socrates_scf_ga.o

#
# This is the archive files where the compiled code goes.
#
LIBRARY = libsocrates.a

#
# Any source files which call BLAS/LAPACK must be listed here.
#
USES_BLAS = 

#
# This is where you put any preprocessor directives.
#
LIB_DEFINES += -DDEBUG_PRINT -DDETAILED_FREE

#
# If you want to use headers that aren't in $(NWCHEM_TOP)/src/include 
# or $(NWCHEM_TOP)/src/tools/include, you need to list their directory
# location here.
#
HEADERS =

include ../config/makefile.h
include ../config/makelib.h

