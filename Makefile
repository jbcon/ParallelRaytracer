debug:
	nvcc -o parallelrenderer.out raytracer.cu

all:
	nvcc -o parallelrenderer.out --compiler-options -Wall -O3 raytracer.cu

run:
	./parallelrenderer.out 640 480
