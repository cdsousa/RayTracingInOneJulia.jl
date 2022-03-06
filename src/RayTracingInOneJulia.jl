module RayTracingInOneJulia


using LinearAlgebra
using StaticArrays
using Images
using CUDA, KernelAbstractions, CUDAKernels
using Tullio

CUDA.allowscalar(false)


export Vec3, Point3

const Vec3 = SVector{3, T} where T
const Point3 = Vec3


include("ray.jl")
include("hittable.jl")
include("sphere.jl")
include("camera.jl")


export render!

function ray_color(r::Ray{T}, world, max_depth) where T
    depth = max_depth
    attenuation = T(1)
    while true
        if depth <= 0
            return RGB{T}(0)
        end
        rec = hit(world, r, T(0.001), T(Inf))
        if !isnothing(rec)
            # d = rec.normal + normalize(randn(Vec3{T}))
            d = randn(Vec3{T}) ; d = d ⋅ rec.normal > 0 ? d : -d
            target = rec.p + d
            r = Ray{T}(rec.p, target - rec.p)
            attenuation = attenuation * T(0.5)
            depth -= one(depth)
        else
            unit_direction = normalize(r.dir)
            t = T(0.5) * (unit_direction.y + T(1))
            return attenuation * ((T(1.0)-t) * RGB{T}(1, 1, 1) + t * RGB{T}(0.5, 0.7, 1))
        end
    end
end

function write_color(pixel_color, samples_per_pixel)
    pixel_color = (1/samples_per_pixel) * pixel_color
    pixel_color = RGB(sqrt(pixel_color.r), sqrt(pixel_color.g), sqrt(pixel_color.b))
    return clamp01(pixel_color)
end

function render_pixel(::Type{T}, i, j, world, cam, image_width, image_height, samples_per_pixel, max_depth) where T
    pixel_color = RGB{T}(0)
    for _ in Base.OneTo(samples_per_pixel)
        u = (j-1+rand(T)) / (image_width-1)
        v = (image_height-i+rand(T)) / (image_height-1)
        r = get_ray(cam, u, v)
        pixel_color += ray_color(r, world, max_depth)
    end
    return write_color(pixel_color, samples_per_pixel)
end

function render!(image::AbstractArray{RGB{T}}, world, cam, image_width, image_height, samples_per_pixel, max_depth) where T
    # @tullio image[i,j] = render_pixel(T, T(i), T(j), world, cam, T(image_width), T(image_height), Int32(samples_per_pixel), Int32(max_depth))
    @kernel function k(image, world, cam, image_width, image_height, samples_per_pixel, max_depth)
        i, j = @index(Global, NTuple)
        image[i, j] = render_pixel(T, T(i), T(j), world, cam, T(image_width), T(image_height), Int32(samples_per_pixel), Int32(max_depth))
    end
    if isa(image, CuArray)
        wait(k(CUDADevice())(image, world, cam, image_width, image_height, samples_per_pixel, max_depth, ndrange=size(image)) )
    else
        wait(k(CPU())(image, world, cam, image_width, image_height, samples_per_pixel, max_depth, ndrange=size(image)) )
    end
    image
end


end # module RayTracingInOneJulia
