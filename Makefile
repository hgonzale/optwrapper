F2C = f2c
F2COPTS = -a -A
CFLAGS = -O2 -g -fPIC
PREFIX = /usr/local
INSTALL_MANIFEST = install_manifest.txt

modules = utils base npsol snopt
objects = dummy filehandler
f2ced = filehandler

all: $(f2ced:%=src/%.c) $(objects:%=src/%.o)
	python setup.py build

install: all
	python setup.py install --prefix=$(PREFIX) --record=$(INSTALL_MANIFEST)

uninstall:
	-if [ -r $(INSTALL_MANIFEST) ]; then \
     cat $(INSTALL_MANIFEST) | xargs -t rm -f; \
     rm $(INSTALL_MANIFEST); \
   fi;

clean:
	-rm -f $(modules:%=src/%.c)
	-rm -f $(objects:%=src/%.o)
	-rm -rf build

test:
	@$(foreach module,$(modules),python -c "from optwrapper.$(module) import *; print('$(module) works fine.')";)

%.c: %.f
	$(F2C) $(F2COPTS) $<

.PHONY: all clean test install uninstall
