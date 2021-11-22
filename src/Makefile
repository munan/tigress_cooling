SHELL = /bin/sh

CC	= gcc
CXX	= g++

OS	:= $(shell uname)
ifeq ($(OS), Darwin)
	CLIBFLAGS	= -MP -fPIC
	DYNLIBFLAG	= -dynamiclib
	LIBEXT		= dylib
else ifeq ($(OS), Linux)
	CLIBFLAGS	= -fPIC
	DYNLIBFLAG	= -shared
	LIBEXT		= so
endif

BINDIR = ../bin/
BINNAME = test
BIN = ${BINDIR}${BINNAME}

DYNLIBDIR = ../lib/
DYNLIBNAME = cool_tigress
DYNLIB = ${DYNLIBDIR}${DYNLIBNAME}.$(LIBEXT)

#CFLAGS = -Wall -g3 -O0 -ggdb
CFLAGS 	= -Wall -O3
CXXLIBFLAGS = -lm -lstdc++
LIBS	= -lm

# -----------------------------------------------------------------------------------------

.PHONY: depend clean lib

all: bin lib

bin:
	$(CXX) $(CFLAGS) $(CLIBFLAGS) $(DYNLIBFLAG) linecool.cpp -o $(DYNLIBDIR)liblinecool.$(LIBEXT)
	$(CXX) $(CFLAGS) $(CLIBFLAGS) $(DYNLIBFLAG) -L$(DYNLIBDIR) -llinecool linecool_c_wrapper.cpp -o $(DYNLIBDIR)liblinecool_c_wrapper.$(LIBEXT)
	$(CC) $(CLIBFLAGS) $(DYNLIBFLAG) cool_tigress.c -L$(DYNLIBDIR) -llinecool_c_wrapper -o $(DYNLIB)
	$(CC) main.c -L$(DYNLIBDIR) -llinecool_c_wrapper -o $(BIN)

lib:
	$(CXX) -c $(CFLAGS) $(CLIBFLAGS) linecool.cpp
	$(CXX) -c $(CFLAGS) $(CLIBFLAGS) linecool_c_wrapper.cpp
	$(CC) -c $(CFLAGS) $(CLIBFLAGS) cool_tigress.c
	$(CXX) $(CLIBFLAGS) $(DYNLIBFLAG) $(CXXLIBFLAGS) -o $(DYNLIB) cool_tigress.o linecool_c_wrapper.o linecool.o

%.o : %.c
	$(CC) -c $(CFLAGS) $(CLIBFLAGS) $< -o $@

%.o : %.cpp
	$(CXX) -c $(CFLAGS) $(CLIBFLAGS) $< -o $@

clean:
	$(RM) *.o
	$(RM) .depend

depend: $(SRCS)
	makedepend  $^
# -----------------------------------------------------------------------------------------
