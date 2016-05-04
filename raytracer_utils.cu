template<typename T>
struct Vec3 {
    T x, y, z;
    __host__ __device__ Vec3() : x(T(0)), y(T(0)), z(T(0)) {}
    __host__ __device__ Vec3(T xx) : x(xx), y(xx), z(xx) {}
    __host__ __device__ Vec3(T xx, T yy, T zz) : x(xx), y(yy), z(zz) {}
    __host__ __device__ Vec3& normalize()
    {
        T nor2 = length2();
        if (nor2 > 0) {
            T invNor = 1 / sqrt(nor2);
            x *= invNor, y *= invNor, z *= invNor;
        }
        return *this;
    }
    __host__ __device__ Vec3<T> operator * (const T &f) const { return Vec3<T>(x * f, y * f, z * f); }
    __host__ __device__ Vec3<T> operator * (const Vec3<T> &v) const { return Vec3<T>(x * v.x, y * v.y, z * v.z); }
    __host__ __device__ T dot(const Vec3<T> &v) const { return x * v.x + y * v.y + z * v.z; }
    __host__ __device__ Vec3<T> operator - (const Vec3<T> &v) const { return Vec3<T>(x - v.x, y - v.y, z - v.z); }
    __host__ __device__ Vec3<T> operator + (const Vec3<T> &v) const { return Vec3<T>(x + v.x, y + v.y, z + v.z); }
    __host__ __device__ Vec3<T>& operator += (const Vec3<T> &v) { x += v.x, y += v.y, z += v.z; return *this; }
    __host__ __device__ Vec3<T>& operator *= (const Vec3<T> &v) { x *= v.x, y *= v.y, z *= v.z; return *this; }
    __host__ __device__ Vec3<T> operator - () const { return Vec3<T>(-x, -y, -z); }
    __host__ __device__ T length2() const { return x * x + y * y + z * z; }
    __host__ __device__ T length() const { return sqrt(length2()); }
    __host__ __device__ friend std::ostream & operator << (std::ostream &os, const Vec3<T> &v)
    {
        os << "[" << v.x << " " << v.y << " " << v.z << "]";
        return os;
    }
};
typedef Vec3<double> Vec3d;

struct Sphere {
    Vec3d center;
    double radius, radius2;
    Vec3d surfaceColor, emissionColor;
    double transparency, reflection;
    Sphere(
        const Vec3d &c,
        const double &r,
        const Vec3d &sc,
        const double &refl = 0,
        const double &transp = 0,
        const Vec3d &ec = 0) :
        center(c), radius(r), radius2(r * r), surfaceColor(sc), emissionColor(ec),
        transparency(transp), reflection(refl)
    { /* empty */ }
    bool intersect(const Vec3d &rayorig, const Vec3d &raydir, double &t0, double &t1) const
    {
        Vec3d l = center - rayorig;
        double tca = l.dot(raydir);
        if (tca < 0) return false;
        double d2 = l.dot(l) - tca * tca;
        if (d2 > radius2) return false;
        double thc = sqrt(radius2 - d2);
        t0 = tca - thc;
        t1 = tca + thc;

        return true;
    }
};
