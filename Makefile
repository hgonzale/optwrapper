F2C = f2c
F2COPTS = -a -A
CFLAGS = -O2 -g -fPIC

modules = utils base npsol snopt
objects = dummy filehandler
f2ced = filehandler

all: $(f2ced:%=src/%.c) $(objects:%=src/%.o)
	python setup.py build

clean:
	-rm -f $(modules:%=src/%.c)
	-rm -f $(objects:%=src/%.o)
	-rm -rf build

test:
	@$(foreach module,$(modules),python -c "from optwrapper.$(module) import *; print('$(module) works fine.')";)

%.c: %.f
	$(F2C) $(F2COPTS) $<

.PHONY: all clean test
