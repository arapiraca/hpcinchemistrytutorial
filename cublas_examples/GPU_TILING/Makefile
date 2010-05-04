

# PRECISION: VALID OPTIONS ARE SINGLE OR DOUBLE

PRECISION=DOUBLE

# WARP: WARP SIZE FOR OPTIMAL GPU COMPUTING ... gadzooks WARP=32

WARP=32

CFLAGS    = -O2  -D$(PRECISION) -DWARP=$(WARP) -Wall
EXEC      = gputiles.x 
CXX       = g++

$(EXEC): main.cpp gputiles.o 
	$(CXX) main.cpp gputiles.o $(CFLAGS) -o $(EXEC)

gputiles.o: gputiles.cpp gputiles.hpp
	$(CXX) -c gputiles.cpp $(CFLAGS)

clean:
	rm -f *.o
