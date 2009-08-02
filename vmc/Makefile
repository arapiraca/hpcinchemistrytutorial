CXXFLAGS = -O3 -ffast-math -march=native 

vmc:	vmc.o
	$(CXX) -o $@ $^ 

vmc.o:	cmwcran.h

clean:
	/bin/rm -f *.o vmc *~ core*


