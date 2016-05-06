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
        int index = threadIdx.x + blockIdx.x * blockDim.x;

}

int main(int argc, char* argv[]){

    srand48(13);
    // command line args are width and height
    if (argc < 3){
        std::cerr << "ERROR: invalid command line arguments\nUsage: ./parallelrenderer.out <width> <height>" << std::endl;
        return 1;
    }

    unsigned int width = atoi(argv[1]);
    unsigned int height = atoi(argv[2]);

    // init data
    Sphere * d_spheres;
    std::vector<Sphere> h_spheres;

    // create spheres
    h_spheres.push_back(Sphere(Vec3d( 0.0, -10004, -20), 10000, Vec3d(0.20, 0.20, 0.20), 0, 0.0));
    h_spheres.push_back(Sphere(Vec3d( 0.0,      0, -20),     4, Vec3d(1.00, 0.32, 0.36), 1, 0.5));
    h_spheres.push_back(Sphere(Vec3d( 5.0,     -1, -15),     2, Vec3d(0.90, 0.76, 0.46), 1.0, 0.7));
    h_spheres.push_back(Sphere(Vec3d( 5.0,      0, -25),     3, Vec3d(0.65, 0.77, 0.97), 1, 0.2));
    h_spheres.push_back(Sphere(Vec3d( 5.0,      -10, -25),     3, Vec3d(0.65, 0.77, 0.97), 0.5, 0.0));
    h_spheres.push_back(Sphere(Vec3d(-5.5,      0, -15),     3, Vec3d(0.90, 0.90, 0.90), 0, 0.0));
    // light
    h_spheres.push_back(Sphere(Vec3d( 0.0,     20, -15),     3, Vec3d(0.00, 0.00, 0.00), 1, 0.0, Vec3d(3)));

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
