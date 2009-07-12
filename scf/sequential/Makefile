
scf:	scf.o input.o integ.o
	$(CXX) -o $@ $^ -lm 

scf.o:	scfdata.h

input.o:	scfdata.h integ.h

integ.o:	integ.h

clean:
	/bin/rm -f scf *.o *~
