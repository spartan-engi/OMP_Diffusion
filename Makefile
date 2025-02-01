
MSVC_Dir   = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64"
MPI_IncDir = "C:\Program Files (x86)\Microsoft SDKs\MPI\Include"
MPI_LibDir = "C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64"
MPI_Runner = "C:\Program Files\Microsoft MPI\Bin\mpiexec.exe"

# include directory, library directory and use lib file
MPI_agruments = -I $(MPI_IncDir) -L $(MPI_LibDir) -lmsmpi

build:
	gcc main.c -o main.exe -g -fopenmp

cuda:
	nvcc main.c -o main.exe -g -D GPU -x cu $(MPI_agruments) -Xcompiler /openmp --compiler-bindir $(MSVC_Dir)
# /showIncludes

run_mpi:
	$(MPI_Runner) -np 2 main.exe
