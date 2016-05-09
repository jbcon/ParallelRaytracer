threaded:
	g++ -pthread -o threadedrenderer.out -Wall -O5 raytrace_pthread.cpp

debug:
	nvcc -o parallelrenderer.out raytracer.cu

all:
	nvcc -o parallelrenderer.out --compiler-options -Wall -O3 raytracer.cu

run:
	./parallelrenderer.out 640 480
