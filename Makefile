all:
	gcc -c dummy.c
	python setup.py build_ext -i

clean: npsol-clean snopt-clean
	rm dummy.o

npsol-clean:
	rm -rf build
	rm -f optw_npsol.c
	rm -f optw_npsol.so

snopt-clean:
	rm -rf build
	rm -f optw_snopt.c
	rm -f optw_snopt.so

test: npsol-test snopt-test

npsol-test:
	python -c "from optw_npsol import NpsolSolver;print 'NpsolSolver works fine.'"

snopt-test:
	python -c "from optw_snopt import SnoptSolver;print 'SnoptSolver works fine.'"
