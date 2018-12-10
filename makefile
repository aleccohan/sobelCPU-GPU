.DEFAULT_GOAL := newSobel
CXX=nvcc
RM=rm -f
NVCCFLAGS=-arch=sm_52 

SRCS=newSobel.cu
OBJS=$(subst .cu,.o,$(SRCS))
NAME=newSobel

SHSRCS=newSobelShared.cu
SHOBJS=$(subst .cu,.o,$(SHSRCS))
SHNAME=sharedSobel

all: newSobel sharedSobel

newSobel: $(OBJS)
	$(CXX)	$(NVCCFLAGS)	-o	$(NAME)	$(SRCS)

sharedSobel:
	$(CXX)	$(NVCCFLAGS)	-o	$(SHNAME)	$(SHSRCS)

clean:
	$(RM)	$(OBJS)
	$(RM)	$(SHOBJS)

distclean: clean
	$(RM)	newSobel
	$(RM)	sharedSobel