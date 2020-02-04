all:
	g++ -g -Wno-deprecated -fopenmp -Iheader -Ilib -lgomp src/* -o main

clean:
	rm -f *.o
	rm -f *.exe