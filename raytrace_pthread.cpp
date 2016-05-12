#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>
#include <iostream>
#include <ctime>
#include <cassert>
#include <pthread.h>

#include "rt_classes.h"

#define MAX_RAY_DEPTH 5
#define NUM_THREADS 50

unsigned int width = 640, height = 480;
struct thread_data{
	int thread_id;
	std::vector<Sphere> spheres;
	Vec3f *image;
};

float mix(const float &a, const float &b, const float &mix)
{
    return b * mix + a * (1 - mix);
}

Vec3f trace(
	const Vec3f &rayorig,
	const Vec3f &raydir,
	const std::vector<Sphere> &spheres,
	const int &depth)
{
	//if (raydir.length() != 1) std::cerr << "Error " << raydir << std::endl;
	float tnear = INFINITY;
	const Sphere* sphere = NULL;
	// find intersection of this ray with the sphere in the scene
	for (unsigned i = 0; i < spheres.size(); ++i) {
		float t0 = INFINITY, t1 = INFINITY;
		if (spheres[i].intersect(rayorig, raydir, t0, t1)) {
			if (t0 < 0) t0 = t1;
			if (t0 < tnear) {
				tnear = t0;
				sphere = &spheres[i];
			}
		}
	}
	// if there's no intersection return black or background color
	if (!sphere) return Vec3f(2);
	Vec3f surfaceColor = 0; // color of the ray/surfaceof the object intersected by the ray
	Vec3f phit = rayorig + raydir * tnear; // point of intersection
	Vec3f nhit = phit - sphere->center; // normal at the intersection point
	nhit.normalize(); // normalize normal direction
	// If the normal and the view direction are not opposite to each other
	// reverse the normal direction. That also means we are inside the sphere so set
	// the inside bool to true. Finally reverse the sign of IdotN which we want
	// positive.
	float bias = 1e-4; // add some bias to the point from which we will be tracing
	bool inside = false;
	if (raydir.dot(nhit) > 0) nhit = -nhit, inside = true;
	if ((sphere->transparency > 0 || sphere->reflection > 0) && depth < MAX_RAY_DEPTH) {
		float facingratio = -raydir.dot(nhit);
		// change the mix value to tweak the effect
		float fresneleffect = mix(pow(1 - facingratio, 3), 1, 0.1);
		// compute reflection direction (not need to normalize because all vectors
		// are already normalized)
		Vec3f refldir = raydir - nhit * 2 * raydir.dot(nhit);
		refldir.normalize();
		Vec3f reflection = trace(phit + nhit * bias, refldir, spheres, depth + 1);
		Vec3f refraction = 0;
		// if the sphere is also transparent compute refraction ray (transmission)
		if (sphere->transparency) {
			float ior = 1.1, eta = (inside) ? ior : 1 / ior; // are we inside or outside the surface?
			float cosi = -nhit.dot(raydir);
			float k = 1 - eta * eta * (1 - cosi * cosi);
			Vec3f refrdir = raydir * eta + nhit * (eta * cosi - sqrt(k));
			refrdir.normalize();
			refraction = trace(phit - nhit * bias, refrdir, spheres, depth + 1);
		}
		// the result is a mix of reflection and refraction (if the sphere is transparent)
		surfaceColor = (
			reflection * fresneleffect +
			refraction * (1 - fresneleffect) * sphere->transparency) * sphere->surfaceColor;
	}
	else {
		// it's a diffuse object, no need to raytrace any further
		for (unsigned i = 0; i < spheres.size(); ++i) {
			if (spheres[i].emissionColor.x > 0) {
				// this is a light
				Vec3f transmission = 1;
				Vec3f lightDirection = spheres[i].center - phit;
				lightDirection.normalize();
				for (unsigned j = 0; j < spheres.size(); ++j) {
					if (i != j) {
						float t0, t1;
						if (spheres[j].intersect(phit + nhit * bias, lightDirection, t0, t1)) {
							transmission = 0;
							break;
						}
					}
				}
				surfaceColor += sphere->surfaceColor * transmission *
				std::max(float(0), nhit.dot(lightDirection)) * spheres[i].emissionColor;
			}
		}
	}

	return surfaceColor + sphere->emissionColor;
}

void *render_thread(void *threadarg){
	struct thread_data *data;
	data = (struct thread_data *) threadarg;
	float invWidth = 1 / float(width), invHeight = 1 / float(height);
	float fov = 30, aspectratio = width / float(height);
	float angle = tan(M_PI * 0.5 * fov / 180.);
	// Trace rays
	data->image += width*data->thread_id;
	for (unsigned y = 0; y < height; ++y) {
		if ((int)y%NUM_THREADS == data->thread_id){
			for (unsigned x = 0; x < width; ++x, ++(data->image)) {
				float xx = (2 * ((x + 0.5) * invWidth) - 1) * angle * aspectratio;
				float yy = (1 - 2 * ((y + 0.5) * invHeight)) * angle;
				Vec3f raydir(xx, yy, -1);
				raydir.normalize();
				*(data->image) = trace(Vec3f(0), raydir, data->spheres, 0);
			}
			data->image += width*(NUM_THREADS-1);
		}
	}
	pthread_exit(NULL);
}

void random_spheres(std::vector<Sphere> &spheres, int n){
    for (int i = 0; i < n; i++){
        spheres.push_back(Sphere(Vec3f( drand48() * 15 -7, drand48() * 3 - 1, -drand48() * 40 - 3), drand48(), Vec3f(drand48(), drand48(), drand48()), drand48(), drand48()));
    }
}


int main(int argc, char **argv)
{

    if (argc == 3){
        width = atoi(argv[1]);
        height = atoi(argv[2]);
    }

	srand48(13);
	std::vector<Sphere> spheres;
	// position, radius, surface color, reflectivity, transparency, emission color
	spheres.push_back(Sphere(Vec3f( 0.0, -10004, -20), 10000, Vec3f(0.20, 0.20, 0.20), 0, 0.0));
	// spheres.push_back(Sphere(Vec3f( 0.0, 0, -40), 4, Vec3f(1.00, 0.32, 0.36), 1, 0.5));
	// spheres.push_back(Sphere(Vec3f( 5.0, -1, -15), 2, Vec3f(0.90, 0.76, 0.46), 1, 0.0));
	// spheres.push_back(Sphere(Vec3f( 5.0, 0, -25), 3, Vec3f(0.65, 0.77, 0.97), 1, 0.0));
	// spheres.push_back(Sphere(Vec3f(-5.5, 0, -15), 3, Vec3f(0.90, 0.90, 0.90), 1, 0.0));

    random_spheres(spheres, 50);
    // light
	spheres.push_back(Sphere(Vec3f( 0.0, 20, -30), 3, Vec3f(0.00, 0.00, 0.00), 0, 0.0, Vec3f(3)));
	//*
	pthread_t threads[NUM_THREADS];
	struct thread_data td[NUM_THREADS];
	int rc, i;
	Vec3f *image = new Vec3f[width * height];

    // start timing
    clock_t start, end;
    start = clock();

    std::cout << "Starting compute..." << std::endl;

    for(i = 0; i < NUM_THREADS; i++){
		td[i].thread_id = i;
		td[i].spheres = spheres;
		td[i].image = image;
		rc = pthread_create(&threads[i], NULL, render_thread, (void*) &td[i]);
		if(rc){
			fprintf(stderr,"Error: unable to create thread");
			exit(-1);
		}
	}
	//wait for all threads to finish
	for (i = 0; i < NUM_THREADS; i++){
		pthread_join(threads[i], NULL);
	}
    end = clock();
    std::cout << "Compute finished in " << ((float)end-start)/CLOCKS_PER_SEC << " seconds" << std::endl;

	// Save result to a PPM image (keep these flags if you compile under Windows)

    char filename[128];
    sprintf(filename, "image%ux%u.ppm", width, height);

	std::ofstream ofs(filename, std::ios::out | std::ios::binary);
	ofs << "P6\n" << width << " " << height << "\n255\n";
	for (unsigned i = 0; i < width * height; ++i) {
		ofs << (unsigned char)(std::min(float(1), image[i].x) * 255) <<
			(unsigned char)(std::min(float(1), image[i].y) * 255) <<
			(unsigned char)(std::min(float(1), image[i].z) * 255);
	}
	ofs.close();
	delete [] image;
	//*/
	//render(spheres);
	return 0;
}
