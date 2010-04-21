CFLAGS    = -O2 
EXEC      = gputiles.x 
CXX       = g++

$(EXEC): main.cpp gputiles.o 
	$(CXX) main.cpp gputiles.o $(CFLAGS) -o $(EXEC)

gputiles.o: gputiles.cpp gputiles.hpp
	$(CXX) -c gputiles.cpp $(CFLAGS)

clean:
	rm -f *.o
