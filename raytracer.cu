#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <fstream>

#include "raytracer_utils.h"

#define NUM_SPHERES 10
// ================================================================
__global__ void trace(const Vec3d * ray_origins,
    const Vec3d * ray_dirs,
    const int * depth,
    Vec3d * image){

}

int main(int argc, char* argv[]){

    // srand48(13);
    // command line args are width and height
    if (argc < 3){
        std::cerr << "ERROR: invalid command line arguments\nUsage: ./parallelrenderer.out <width> <height>" << std::endl;
        return 1;
    }

    unsigned int width = atoi(argv[1]);
    unsigned int height = atoi(argv[2]);

    // init data
    Sphere * d_spheres;
    Sphere * h_spheres = new Sphere[NUM_SPHERES];

    // create spheres
    cudaMalloc(&d_spheres, NUM_SPHERES * sizeof(Sphere));

    // start render function

    // create image
    Vec3d * image = new Vec3d[width * height];

    // write to file

    delete [] image;
    // clean up
    cudaFree(d_spheres);

    return 0;
}
