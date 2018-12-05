.DEFAULT_GOAL := newSobel
CXX=nvcc
RM=rm -f
NVCCFLAGS=-arch=sm_52 

SRCS=newSobel.cu
OBJS=$(subst .cu,.o,$(SRCS))
NAME=newSobel

all: newSobel

newSobel: $(OBJS)
	$(CXX)	$(NVCCFLAGS)	-o	$(NAME)	$(SRCS)

clean:
	$(RM)	$(OBJS)

distclean: clean
	$(RM)	newSobel
	$(RM)	newSobel