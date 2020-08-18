CXX = g++ -g
CXXFLAGS = -mfma
LDLIBS = -lm -lGL -lglfw -lGLEW

gltest: gltest.o object.o camera.o graph.o ssemath.o
	${CXX} $^ -o gltest ${LDLIBS}

ssemath.o: ssemath.cc
graph.o: graph.cc
gltest.o: gltest.cc
camera.o: camera.cc
object.o: object.cc

clean:
	rm -f gltest *.o
