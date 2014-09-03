F2C = f2c
F2COPTS = -a -A
INSTALL_MANIFEST = install_manifest.txt

modules = utils base npsol snopt
f2ced = filehandler

all: $(f2ced:%=src/%.c)
	python setup.py build

install: all
	python setup.py install --prefix=$(PREFIX) --record=$(INSTALL_MANIFEST)

uninstall:
	-if [ -r $(INSTALL_MANIFEST) ]; then \
     cat $(INSTALL_MANIFEST) | xargs -t rm -f; \
     rm $(INSTALL_MANIFEST); \
   fi;

clean:
	-rm -rf build

test:
	@$(foreach module,$(modules),python -c "from optwrapper.$(module) import *; print('$(module) works fine.')";)

%.c: %.f
	$(F2C) $(F2COPTS) $<

.PHONY: all clean test install uninstall
