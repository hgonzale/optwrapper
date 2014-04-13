all:
	gcc -c dummy.c
	python setup.py build_ext -i

clean: npsol-clean snopt-clean arrayWrapper-clean

npsol-clean:
	rm -rf build
	rm -f optwNpsol.c
	rm -f optwNpsol.so
	rm -f dummy.o

snopt-clean:
	rm -rf build
	rm -f optwSnopt.c
	rm -f optwSnopt.so
	rm -f dummy.o

arrayWrapper-clean:
	rm -rf build
	rm -f arrayWrapper.c
	rm -f arrayWrapper.so

test: npsol-test snopt-test

npsol-test:
	python -c "from optw_npsol import NpsolSolver;print 'NpsolSolver works fine.'"

snopt-test:
	python -c "from optw_snopt import SnoptSolver;print 'SnoptSolver works fine.'"

.PHONY: all clean npsol-clean snpopt-clean arrayWrapper-clean test npsol-test snopt-test
