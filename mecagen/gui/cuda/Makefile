all:vbo.o

vbo.o:vbo.cu vbo.h kernel.cu kernel.cuh
	$(CUDACC) -DGRN -I../includes -I../../common/includes -I../../model/includes -c vbo.cu -o vbo.o

clean:
	rm -f vbo.o

propper:
	clean
