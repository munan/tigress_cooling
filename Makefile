SHELL = /bin/sh

CC	= gcc
CXX	= g++

#CFLAGS = -Wall -g3 -O0 -ggdb
CFLAGS 	= -Wall -O3
LIBS	= -lm
LIBNAME = cool_tigress

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

# -----------------------------------------------------------------------------------------

.PHONY: depend clean lib

all: main lib

main:
	$(CXX) $(CFLAGS) $(CLIBFLAGS) $(DYNLIBFLAG) linecool.cpp -o liblinecool.$(LIBEXT)
	$(CXX) $(CFLAGS) $(CLIBFLAGS) $(DYNLIBFLAG) -L. -llinecool linecool_c_wrapper.cpp -o liblinecool_c_wrapper.$(LIBEXT)
	$(CC) $(CLIBFLAGS) $(DYNLIBFLAG) cool_tigress.c -L. -llinecool_c_wrapper -o $(LIBNAME).$(LIBEXT)
	#$(CC) main.c -L. -llinecool_c_wrapper -o a

lib:
	$(CXX) -c $(CFLAGS) $(CLIBFLAGS) linecool.cpp
	$(CXX) -c $(CFLAGS) $(CLIBFLAGS) linecool_c_wrapper.cpp
	$(CC) -c $(CFLAGS) $(CLIBFLAGS) cool_tigress.c
	$(CC) $(CLIBFLAGS) $(DYNLIBFLAG) -lstdc++ -o $(LIBNAME).$(LIBEXT) cool_tigress.o linecool_c_wrapper.o linecool.o

%.o : %.c
	$(CC) -c $(CFLAGS) $(CLIBFLAGS) $< -o $@

%.o : %.cpp
	$(CXX) -c $(CFLAGS) $(CLIBFLAGS) $< -o $@

clean:
	$(RM) *.o *.$(LIBEXT)
	$(RM) .depend

depend: $(SRCS)
	makedepend  $^
# -----------------------------------------------------------------------------------------
