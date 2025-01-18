
compilerDir = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64"

build:
	gcc main.c -o main.exe -g -lpthread -fopenmp

cuda:
	nvcc main.c -o main.exe -g -D GPU -x cu -Xcompiler /openmp --compiler-bindir $(compilerDir)
# /showIncludes
