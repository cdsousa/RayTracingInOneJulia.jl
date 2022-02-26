using LinearAlgebra
using StaticArrays
using Images
using CUDA, KernelAbstractions, CUDAKernels
using Tullio
using ChangePrecision

const Vec3 = SVector{3, T} where T
const Point3 = Vec3

struct Ray{T}
    orig::Point3{T}
    dir::Vec3{T}
end
at(r::Ray, t) = r.orig + t * r.dir


function ray_color(r::Ray)
    @changeprecision Float32 begin
        unit_direction = normalize(r.dir)
        t = 0.5 * (unit_direction.y + 1.0)
        return (1.0-t) * RGB(1.0, 1.0, 1.0) + t * RGB(0.5, 0.7, 1.0)
    end
end


function main(use_cuda=true)
    @changeprecision Float32 begin

        # Image

        aspect_ratio = 16 / 9
        image_width = 400
        image_height = floor(Int, image_width / aspect_ratio)

        if use_cuda[]
            ArrType = CuArray
        else
            ArrType = Array
        end

        image = ArrType{RGB{Float32}}(undef, image_height, image_width)

        # Camera

        viewport_height = 2.0
        viewport_width = aspect_ratio * viewport_height
        focal_length = 1.0

        origin = Point3(0.0, 0.0, 0.0)
        horizontal = Vec3(viewport_width, 0.0, 0.0)
        vertical = Vec3(0, viewport_height, 0.0)
        lower_left_corner = origin - horizontal/2 - vertical/2 - Vec3(0.0, 0.0, focal_length)

        # Render

        function render(i, j)
            u = (j-1) / (image_width-1)
            v = (image_height-i) / (image_height-1)
            r = Ray(origin, lower_left_corner + u*horizontal + v*vertical - origin)
            ray_color(r)
        end

        @time begin
            @tullio image[i,j] = render(i,j)
            ArrType==CuArray && CUDA.synchronize()
            image = ArrType!=Array ? Array(image) : image
        end

        # Save

        save("image.png", image)

        image

    end
end


main(false)
main(true)

main(false)
main(true)
