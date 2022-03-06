using Revise
using Images
using StaticArrays
using ChangePrecision
using BenchmarkTools
using CUDA

using RayTracingInOneJulia


function main(use_cuda=true)
    @changeprecision Float32 begin
        T = typeof(0.0)

        # Image

        aspect_ratio = 16 / 9
        image_width = 400
        image_height = floor(Int, image_width / aspect_ratio)
        samples_per_pixel = 100
        max_depth = 50

        if use_cuda
            ArrType = CuArray
        else
            ArrType = Array
        end

        image = ArrType{RGB{T}}(undef, image_height, image_width)

        # World

        world = ArrType([Sphere(Point3(0.0,0.0,-1.0), 0.5), Sphere(Point3(0.0,-100.5,-1.0), 100.0)])

        # Camera

        cam = Camera{T}()

        # Render

        @time begin
            render!(image, world, cam, image_width, image_height, samples_per_pixel, max_depth)
            ArrType==CuArray && CUDA.synchronize()
        end
        # @btime begin
        #     render!($image, $world, $cam, $image_width, $image_height, $samples_per_pixel, $max_depth)
        #     $ArrType==CuArray && CUDA.synchronize()
        # end

        image = ArrType != Array ? Array(image) : image


        # Save

        save("image.png", image)

        image

    end # @changeprecision
end


main(false)
main(true)
# #

main(false)
main(false)
main(false)
main(true)
main(true)
main(true)
