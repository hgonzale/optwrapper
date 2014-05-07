modules = utils base npsol

all: dummy.o
	python setup.py build_ext --inplace

clean:
	-rm -f $(modules:%=%.so) $(modules:%=%.c)
	-rm -f fort.*
	-rm -rf build
	-rm -f *.o

test:
	@$(foreach module,$(modules),python -c "from $(module) import *; print('$(module) works fine.')";)

.PHONY: all clean test
