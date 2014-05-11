F2C = f2c
F2COPTS = -a -A
CFLAGS = -O2 -g -fPIC

modules = utils base npsol snopt

all: dummy.o filehandler.c filehandler.o
	python setup.py build_ext --inplace

clean:
	-rm -f $(modules:%=%.so) $(modules:%=%.c)
	-rm -f fort.*
	-rm -rf build
	-rm -f *.o

test:
	@$(foreach module,$(modules),python -c "from $(module) import *; print('$(module) works fine.')";)

%.c: %.f
	$(F2C) $(F2COPTS) $<

.PHONY: all clean test
