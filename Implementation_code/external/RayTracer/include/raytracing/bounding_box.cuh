#pragma once

#include <raytracing/common.h>
#include <raytracing/triangle.cuh>

namespace raytracing {

template <int N_POINTS>
__host__ __device__ inline void project(Eigen::Vector3f points[N_POINTS], const Eigen::Vector3f& axis, float& min, float& max) {
    min = std::numeric_limits<float>::infinity();
    max = -std::numeric_limits<float>::infinity();

    #pragma unroll
    for (uint32_t i = 0; i < N_POINTS; ++i) {
        float val = axis.dot(points[i]);

        if (val < min) {
            min = val;
        }

        if (val > max) {
            max = val;
        }
    }
}

struct BoundingBox {

    Eigen::Vector3f min = Eigen::Vector3f::Constant(std::numeric_limits<float>::infinity());
    Eigen::Vector3f max = Eigen::Vector3f::Constant(-std::numeric_limits<float>::infinity());

    __host__ __device__ BoundingBox() {}

    __host__ __device__ BoundingBox(const Eigen::Vector3f& a, const Eigen::Vector3f& b) : min{a}, max{b} {}

    __host__ __device__ explicit BoundingBox(const Triangle& tri) {
        min = max = tri.a;
        enlarge(tri.b);
        enlarge(tri.c);
    }

    BoundingBox(std::vector<Triangle>::iterator begin, std::vector<Triangle>::iterator end) {
        min = max = begin->a;
        for (auto it = begin; it != end; ++it) {
            enlarge(*it);
        }
    }

    __host__ __device__ void enlarge(const BoundingBox& other) {
        min = min.cwiseMin(other.min);
        max = max.cwiseMax(other.max);
    }

    __host__ __device__ void enlarge(const Triangle& tri) {
        enlarge(tri.a);
        enlarge(tri.b);
        enlarge(tri.c);
    }

    __host__ __device__ void enlarge(const Eigen::Vector3f& point) {
        min = min.cwiseMin(point);
        max = max.cwiseMax(point);
    }

    __host__ __device__ void inflate(float amount) {
        min -= Eigen::Vector3f::Constant(amount);
        max += Eigen::Vector3f::Constant(amount);
    }

    __host__ __device__ Eigen::Vector3f diag() const {
        return max - min;
    }

    __host__ __device__ Eigen::Vector3f relative_pos(const Eigen::Vector3f& pos) const {
        return (pos - min).cwiseQuotient(diag());
    }

    __host__ __device__ Eigen::Vector3f center() const {
        return 0.5f * (max + min);
    }

    __host__ __device__ BoundingBox intersection(const BoundingBox& other) const {
        BoundingBox result = *this;
        result.min = result.min.cwiseMax(other.min);
        result.max = result.max.cwiseMin(other.max);
        return result;
    }

    __host__ __device__ bool intersects(const BoundingBox& other) const {
        return !intersection(other).is_empty();
    }

    // Based on the separating axis theorem
    // (https://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/tribox_tam.pdf)
    // Code adapted from a C# implementation at stack overflow
    // https://stackoverflow.com/a/17503268
    __host__ __device__ bool intersects(const Triangle& triangle) const {
        float triangle_min, triangle_max;
        float box_min, box_max;

        // Test the box normals (x-, y- and z-axes)
        Eigen::Vector3f box_normals[3] = {
            Eigen::Vector3f{1.0f, 0.0f, 0.0f},
            Eigen::Vector3f{0.0f, 1.0f, 0.0f},
            Eigen::Vector3f{0.0f, 0.0f, 1.0f},
        };

        Eigen::Vector3f triangle_normal = triangle.normal();
        Eigen::Vector3f triangle_verts[3];
        triangle.get_vertices(triangle_verts);

        for (int i = 0; i < 3; i++) {
            project<3>(triangle_verts, box_normals[i], triangle_min, triangle_max);
            if (triangle_max < min[i] || triangle_min > max[i]) {
                return false; // No intersection possible.
            }
        }

        Eigen::Vector3f verts[8];
        get_vertices(verts);

        // Test the triangle normal
        float triangle_offset = triangle_normal.dot(triangle.a);
        project<8>(verts, triangle_normal, box_min, box_max);
        if (box_max < triangle_offset || box_min > triangle_offset) {
            return false; // No intersection possible.
        }

        // Test the nine edge cross-products
        Eigen::Vector3f edges[3] = {
            triangle.a - triangle.b,
            triangle.a - triangle.c,
            triangle.b - triangle.c,
        };

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                // The box normals are the same as it's edge tangents
                Eigen::Vector3f axis = edges[i].cross(box_normals[j]);
                project<8>(verts, axis, box_min, box_max);
                project<3>(triangle_verts, axis, triangle_min, triangle_max);
                if (box_max < triangle_min || box_min > triangle_max)
                    return false; // No intersection possible
            }
        }

        // No separating axis found.
        return true;
    }

    __host__ __device__ Eigen::Vector2f ray_intersect(Eigen::Ref<const Eigen::Vector3f> pos, Eigen::Ref<const Eigen::Vector3f> dir) const {
        float tmin = (min.x() - pos.x()) / dir.x();
        float tmax = (max.x() - pos.x()) / dir.x();

        if (tmin > tmax) {
            host_device_swap(tmin, tmax);
        }

        float tymin = (min.y() - pos.y()) / dir.y();
        float tymax = (max.y() - pos.y()) / dir.y();

        if (tymin > tymax) {
            host_device_swap(tymin, tymax);
        }

        if (tmin > tymax || tymin > tmax) {
            return { std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
        }

        if (tymin > tmin) {
            tmin = tymin;
        }

        if (tymax < tmax) {
            tmax = tymax;
        }

        float tzmin = (min.z() - pos.z()) / dir.z();
        float tzmax = (max.z() - pos.z()) / dir.z();

        if (tzmin > tzmax) {
            host_device_swap(tzmin, tzmax);
        }

        if (tmin > tzmax || tzmin > tmax) {
            return { std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
        }

        if (tzmin > tmin) {
            tmin = tzmin;
        }

        if (tzmax < tmax) {
            tmax = tzmax;
        }

        return { tmin, tmax };
    }

    __host__ __device__ bool is_empty() const {
        return (max.array() < min.array()).any();
    }

    __host__ __device__ bool contains(const Eigen::Vector3f& p) const {
        return
            p.x() >= min.x() && p.x() <= max.x() &&
            p.y() >= min.y() && p.y() <= max.y() &&
            p.z() >= min.z() && p.z() <= max.z();
    }

    /// Calculate the squared point-AABB distance
    __host__ __device__ float distance(const Eigen::Vector3f& p) const {
        return sqrt(distance_sq(p));
    }

    __host__ __device__ float distance_sq(const Eigen::Vector3f& p) const {
        return (min - p).cwiseMax(p - max).cwiseMax(0.0f).squaredNorm();
    }

    __host__ __device__ float signed_distance(const Eigen::Vector3f& p) const {
        Eigen::Vector3f q = (p - min).cwiseAbs() - diag();
        return q.cwiseMax(0.0f).norm() + std::min(std::max(q.x(), std::max(q.y(), q.z())), 0.0f);
    }

    __host__ __device__ void get_vertices(Eigen::Vector3f v[8]) const {
        v[0] = {min.x(), min.y(), min.z()};
        v[1] = {min.x(), min.y(), max.z()};
        v[2] = {min.x(), max.y(), min.z()};
        v[3] = {min.x(), max.y(), max.z()};
        v[4] = {max.x(), min.y(), min.z()};
        v[5] = {max.x(), min.y(), max.z()};
        v[6] = {max.x(), max.y(), min.z()};
        v[7] = {max.x(), max.y(), max.z()};
    }

};

inline std::ostream& operator<< (std::ostream& os, const raytracing::BoundingBox& bb) {
    os << "[";
    os << "min=[" << bb.min.x() << "," << bb.min.y() << "," << bb.min.z() << "], ";
    os << "max=[" << bb.max.x() << "," << bb.max.y() << "," << bb.max.z() << "]";
    os << "]";
    return os;
}

}