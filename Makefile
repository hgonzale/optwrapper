INSTALL_MANIFEST = install_manifest.txt

modules = utils base lssol npsol snopt

all:
	python setup.py build

install: all
	python setup.py install --record=$(INSTALL_MANIFEST)

install_local: all
	python setup.py install --record=$(INSTALL_MANIFEST) --user

uninstall:
	-if [ -r $(INSTALL_MANIFEST) ]; then \
     cat $(INSTALL_MANIFEST) | xargs -t rm -f; \
     rm $(INSTALL_MANIFEST); \
   fi;

clean:
	-rm -rf build

test:
	@cd ${HOME}; $(foreach module,$(modules),python -c "from optwrapper.$(module) import *; print('$(module) works fine.')";)

.PHONY: all clean test install uninstall
