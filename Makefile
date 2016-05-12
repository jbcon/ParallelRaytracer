threaded:
	g++ -pthread -o threadedrenderer.out -Wall -O5 raytrace_pthread.cpp

serial:
	g++ -o serialrenderer.out -Wall -O5 serialraytracer.cpp

debug:
	nvcc -o threadedrenderer.out raytrace_pthread.cpp

all:
	nvcc -o parallelrenderer.out --compiler-options -Wall -O3 raytracer.cu

run:
	./threadedrenderer.out 3840 2160
