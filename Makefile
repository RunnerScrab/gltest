CXX = g++ -O3
CXXFLAGS = -mfma
LDLIBS = -lm -lGL -lglfw -lGLEW

gltest: gltest.o object.o camera.o graph.o
	${CXX} $^ -o gltest ${LDLIBS}

graph.o: graph.cc
gltest.o: gltest.cc
camera.o: camera.cc
object.o: object.cc

clean:
	rm -f gltest *.o
