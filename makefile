#
#  make FN=<filename without the .cu>
#  make clean FN=<filename without the .cu>
#  
#  make check_reg FN=<filename without the .cu>
#  make debug FN=<filename without the .cu>
#
FN=
NVCC=nvcc

NVCCFLAGS=-arch=sm_52 
EXTRAS=-cubin -Xptxas="-v" --ptxas-options=-v
$(FN):	$(FN).cu
		$(NVCC) $(FN).cu $(NVCCFLAGS) -o $(FN)
clean:
		rm *.o *~ $(FN)	
check_reg:
	$(NVCC) $(FN).cu $(NVCCFLAGS) $(EXTRAS) -o $(FN)
debug:
	$(NVCC) $(FN).cu -DDEBUG $(NVCCFLAGS) -o $(FN)
