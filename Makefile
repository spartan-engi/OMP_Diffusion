build:
	gcc main.c -o main.exe -g -lpthread -fopenmp

run:
	./main.exe