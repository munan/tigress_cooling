DIRS = bin lib

#-------------------------------------------------------------------------------
#  target all:
all:    dirs bin lib

help:
	@echo "all:       create ($(DIRS)) subdirectory and make lib and bin"
	@echo "dirs:      create ($(DIRS)) subdirectory"
	@echo "lib:	  compile the code and create dynamic library file in lib"
	@echo "bin:	  compile the code and create test c program file in bin"
	@echo "clean:     clean /src subdirectory"

.PHONY: depend clean bin lib
#-------------------------------------------------------------------------------
#  target dirs:
dirs:
	-@for i in $(DIRS) ; do \
	(if [ -d $$i ]; \
	then \
	    echo DIR $$i exists; \
	else \
	    echo DIR $$i created; \
	    mkdir $$i; \
	fi); done

#-------------------------------------------------------------------------------
#  target bin: runs make bin in /src
bin:
	(cd src; $(MAKE) bin)

#-------------------------------------------------------------------------------
#  target lib: runs make lib in /src
lib:
	(cd src; $(MAKE) lib)

#-------------------------------------------------------------------------------
#  target clean:
clean:
	(cd src; $(MAKE) clean)
