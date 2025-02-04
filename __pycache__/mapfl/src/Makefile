FC = gfortran

ifeq ($(TARGET),)
  TARGET = mapfl
endif

FFLAGS = -O3 -march=native -fopenmp -I/opt/homebrew/Cellar/hdf5-mpi/1.14.3/include/

OBJS = psi_io.o mapfl.o

LDFLAGS = -L/opt/homebrew/Cellar/hdf5-mpi/1.14.3/lib/ -lhdf5_fortran -lhdf5_hl_fortran -lhdf5 -lhdf5_hl -ld_classic

all: $(OBJS)
	$(FC) $(FFLAGS) $(OBJS) $(LDFLAGS) -o mapfl
	rm *.o *.mod

clean:
	rm -f *.mod *.o *.so 2>/dev/null
	rm mapfl 2>/dev/null

psi_io.o: psi_io.f90
	$(FC) -c $(FFLAGS) $<

mapfl.o: mapfl.f
	$(FC) -c $(FFLAGS) $<

