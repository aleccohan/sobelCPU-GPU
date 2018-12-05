.DEFAULT_GOAL := newSobel
CXX=g++
RM=rm -f
CPPFLAGS=-g -Wall

SRCS=newSobel.cpp
OBJS=$(subst .cpp,.o,$(SRCS))
NAME=newSobel

all: newSobel

newSobel: $(OBJS)
	$(CXX)	$(CPPFLAGS)	-o	$(NAME)	$(SRCS)

clean:
	$(RM)	$(OBJS)

distclean: clean
	$(RM)	newSobel
	$(RM)	newSobel