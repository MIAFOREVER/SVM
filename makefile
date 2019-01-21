all:a.out clean
a.out:main.o MYSVM.o
	g++ -o a.out main.o MYSVM.o `pkg-config --cflags --libs opencv`
main.o:main.cpp MYSVM.h
	g++ -c main.cpp `pkg-config --cflags --libs opencv`
MYSVM.o:MYSVM.cpp
	g++ -c MYSVM.cpp `pkg-config --cflags --libs opencv`
clean:
	rm -rf *.o