#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <fstream>

#include "raytracer_utils.h"
#if defined __linux__ || defined __APPLE__
// "Compiled for Linux
#else
// Windows doesn't define these values by default, Linux does
#define M_PI 3.141592653589793
#define INFINITY 1e8
#endif

#define NUM_SPHERES 10
// ================================================================
__global__ void trace(const Vec3d * ray_origin
    const Vec3d * ray_dir,
    const int * depth,
    const Sphere * spheres,
    Vec3d * image){
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        Vec3d thisRayOrigin = ray_origins[index];
        Vec3d thisRayDir = ray_dirs[index];
        // determine closest sphere
        double nearest = INFINITY;
        const Sphere * sphere = NULL;
        for(int i = 0; i < NUM_SPHERES; i++){
            double t0 = INFINITY, t1 = INFINITY;
            if (spheres[i].intersect(thisRayOrigin, thisRayDir, t0, t1)) {
                if (t0 < 0) t0 = t1;
                if (t0 < nearest) {
                    nearest = t0;
                    sphere = &spheres[i];
                }
            }
        }



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
    cudaMemcpy(d_spheres, h_spheres.data(), NUM_SPHERES, cudaMemcpyHostToDevice);
    // create image
    Vec3d * image = new Vec3d[width * height];

    // start render function
    render<<<1,1>>>(Vec3d())

    // write to file

    delete [] image;
    // clean up
    cudaFree(d_spheres);

    return 0;
}
