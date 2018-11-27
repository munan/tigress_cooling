prob = main.c
exe = main.exe
SHELL = /bin/sh

CC	= gcc
#CFLAGS = -Wall -g3 -O0 -ggdb
CFLAGS 	= -Wall -O3
LIBS	= -lm

OS	:= $(shell uname)
ifeq ($(OS), Darwin)
	CLIBFLAGS	= -MP
	DYNLIBFLAG	= -dynamiclib
	LIBEXT		= dylib
else ifeq ($(OS), Linux)
        CLIBFLAGS	= -fPIC
        DYNLIBFLAG	= -shared
        LIBEXT		= so
endif

#source files
SRCS = $(prob) cool_tigress.c

#object files
OBJS = $(SRCS:.cc=.o)

#executable
PROJ= $(exe)

# -----------------------------------------------------------------------------------------

.PHONY: depend clean

all: $(PROJ)

lib:	
	$(CC) $(CFLAGS) $(CLIBFLAGS) -c -o cool_tigress.o cool_tigress.c
	$(CC) $(CFLAGS) $(CLIBFLAGS) $(DYNLIBFLAG) -o cool_tigress.$(LIBEXT) cool_tigress.o

$(PROJ): $(OBJS)
	$(CC) -o $@ $^ $(CFLAGS) $(CLIBFLAGS) $(LIBS) 

.cc.o:
	$(CC) $(CFLAGS) $(CLIBFLAGS) -c $< -o $@

clean:
	$(RM) *.o *.dylib *.so
	$(RM) .depend
	$(RM) *.exe

depend: $(SRCS)
	makedepend  $^
# -----------------------------------------------------------------------------------------
