# Parallel Raytracer

This is a Pthread implementation for a parallel raytracer as described on http://scratchapixel.com, developed as our final project for RPI's CSCI-4320: Parallel Programming course.

Build instructions:
 - Provided in Makefile
 - ```g++ raytrace_pthread.cpp -pthread -Wall -O5```

Usage:
 ```./threadedrenderer.out <width> <height> <#spheres> <max_radius> <#threads> <#max_ray_depth>```

Developed by:
 - John Conover
 - Andrew Eng
 - Lucas Volle
