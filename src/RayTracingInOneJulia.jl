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
include("material.jl")
include("sphere.jl")
include("camera.jl")


export render!

function ray_color(r::Ray{T}, world, max_depth) where T
    depth = max_depth
    attenuation = RGB{T}(1)
    while true
        if depth <= 0
            return RGB{T}(0)
        end
        rec = hit(world, r, T(0.001), T(Inf))
        if !isnothing(rec)
            scat_rec = scatter(rec.material, r, rec)
            if !isnothing(scat_rec)
                attenuation = attenuation ⊙ scat_rec.attenuation
                r = scat_rec.scattered
                depth -= one(depth)
            else
                return RGB{T}(0)
            end
        else
            unit_direction = normalize(r.dir)
            t = T(0.5) * (unit_direction.y + T(1))
            return attenuation ⊙ ((T(1.0)-t) * RGB{T}(1, 1, 1) + t * RGB{T}(0.5, 0.7, 1))
        end
    end
end

function write_color(pixel_color, samples_per_pixel)
    pixel_color = (1/samples_per_pixel) * pixel_color
    pixel_color = RGB(sqrt(pixel_color.r), sqrt(pixel_color.g), sqrt(pixel_color.b))
    return clamp01(pixel_color)
end

function render_pixel(::Type{T}, i, j, world, cam, image_width, image_height, samples_per_pixel, max_depth) where T
    (i, j, image_width, image_height) = T.((i, j, image_width, image_height))
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
    @tullio image[i,j] = render_pixel(T, i, j, $world, cam, image_width, image_height, Int32(samples_per_pixel), Int32(max_depth))
end


end # module RayTracingInOneJulia
