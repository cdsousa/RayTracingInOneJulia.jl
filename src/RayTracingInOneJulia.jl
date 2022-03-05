module RayTracingInOneJulia

using LinearAlgebra
using LinearAlgebra: norm_sqr
using StaticArrays
using Images
using CUDA, KernelAbstractions, CUDAKernels
using Tullio
using ChangePrecision

CUDA.allowscalar(false)


export Vec3, Point3, Camera, Sphere
export render!


const Vec3 = SVector{3, T} where T
const Point3 = Vec3


struct Camera{T}
    origin::Point3{T}
    lower_left_corner::Point3{T}
    horizontal::Vec3{T}
    vertical::Vec3{T}
end

struct Ray{T}
    orig::Point3{T}
    dir::Vec3{T}
end
at(r::Ray, t) = r.orig + t * r.dir

struct HitRecord{T}
    p::Point3{T}
    normal::Vec3{T}
    t::T
    front_face::Bool
end
function HitRecord(r::Ray, p, t, outward_normal)
    front_face = r.dir ⋅ outward_normal < 0
    normal = front_face ? outward_normal : -outward_normal
    return HitRecord(p, normal, t, front_face)
end

abstract type Hittable end

    struct Sphere{T} <: Hittable
        center::Point3{T}
        radius::T
    end


function Camera{T}() where T
    aspect_ratio = T(16) / T(9)
    viewport_height = T(2)
    viewport_width = aspect_ratio * viewport_height
    focal_length = T(1)

    origin = Point3{T}(0, 0, 0)
    horizontal = Vec3{T}(viewport_width, 0, 0)
    vertical = Vec3{T}(0, viewport_height, 0)
    lower_left_corner = origin - horizontal/T(2) - vertical/T(2) - Vec3{T}(0, 0, focal_length)

    return Camera(origin, lower_left_corner, horizontal, vertical)
end
function get_ray(cam, u, v)
    return Ray(cam.origin, cam.lower_left_corner + u*cam.horizontal + v*cam.vertical - cam.origin)
end

function hit(s::Sphere, r::Ray, t_min, t_max)::Union{HitRecord, Nothing}
    oc = r.orig - s.center
    a = norm_sqr(r.dir)
    half_b = oc ⋅ r.dir
    c = norm_sqr(oc) - s.radius^2
    discriminant = half_b^2 - a*c
    if discriminant <= 0
        return nothing
    end
    sqrtd = sqrt(discriminant)

    # Find the nearest root that lies in the acceptable range.
    root = (-half_b - sqrtd) / a
    if root < t_min || t_max < root
        root = (-half_b + sqrtd) / a
        if root < t_min || t_max < root
            return nothing
        end
    end

    t = root
    p = at(r, t)
    outward_normal = (p - s.center) / s.radius

    return HitRecord(r, p, t, outward_normal)
end

function hit(hitables::AbstractArray, r::Ray, t_min, t_max)
    rec = nothing
    closest_so_far = t_max

    for object in hitables
        temp_rec = hit(object, r, t_min, closest_so_far)
        if !isnothing(temp_rec)
            closest_so_far = temp_rec.t
            rec = temp_rec
        end
    end

    return rec
end

function ray_color(r::Ray{T}, world) where T
    rec = hit(world, r, T(0), T(Inf))
    if !isnothing(rec)
        return T(0.5) * (RGB{T}(rec.normal[1], rec.normal[2], rec.normal[3]) + RGB{T}(1))
    end
    unit_direction = normalize(r.dir)
    t = T(0.5) * (unit_direction.y + T(1))
    return (T(1.0)-t) * RGB{T}(1, 1, 1) + t * RGB{T}(0.5, 0.7, 1)
end

function render_pixel(::Type{T}, i, j, world, cam, image_width, image_height, samples_per_pixel) where T
    pixel_color = RGB{T}(0)
    for _=Int32(1):samples_per_pixel
        u = (j-1+rand(T)) / (image_width-1)
        v = (image_height-1 - (i-1+rand(T))) / (image_height-1)
        r = get_ray(cam, u, v)
        pixel_color  += ray_color(r, world)
    end
    return clamp01(pixel_color/T(samples_per_pixel))
end

function render!(image::AbstractArray{RGB{T}}, world, cam, image_width, image_height, samples_per_pixel) where T
    @tullio image[i,j] = render_pixel(T, T(i), T(j), world, cam, T(image_width), T(image_height), Int32(samples_per_pixel))
    # @kernel function k(image, world, cam, image_width, image_height, samples_per_pixel)
    #     i, j = @index(Global, NTuple)
    #     image[i, j] = render_pixel(T, T(i), T(j), world, cam, T(image_width), T(image_height), Int32(samples_per_pixel))
    # end
    # if isa(image, CuArray)
    #     wait(k(CUDADevice())(image, world, cam, image_width, image_height, samples_per_pixel, ndrange=size(image)) )
    # else
    #     wait(k(CPU())(image, world, cam, image_width, image_height, samples_per_pixel, ndrange=size(image)) )
    # end
    # image
end

end
