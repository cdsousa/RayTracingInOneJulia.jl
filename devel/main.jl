using Revise
using Images
using StaticArrays
using ChangePrecision
using BenchmarkTools
using Setfield
using CUDA

using RayTracingInOneJulia

import RayTracingInOneJulia.scatter

# Union material approach
UnionMaterial = Union{Lambertian{T}, Metal{T}} where T
struct UnionMaterialWrapper{T}; _::UnionMaterial{T}; end
scatter(r_in::Ray{T}, rec::HitRecord{T, UnionMaterialWrapper{T}}) where {T} = scatter(r_in, @set rec.material = rec.material._)

# Indexed union-split vector approach
scatter(r_in::Ray{T}, rec::HitRecord{T, <:NamedTuple{(:list,:idx,)}}) where {T} = scatter(r_in, @set rec.material = rec.material.list[rec.material.idx])

# United material type approach
@enum UnitedMaterialType  UnitedMaterialLambertian UnitedMaterialMetal
struct UnitedMaterial{T}; typ::UnitedMaterialType; albedo::RGB{T}; fuzz::T; end
UnitedMaterial(mat::Lambertian{T}) where T = UnitedMaterial{T}(UnitedMaterialLambertian, mat.albedo, T(0))
UnitedMaterial(mat::Metal{T}) where T = UnitedMaterial{T}(UnitedMaterialMetal, mat.albedo, mat.fuzz)
function specialize_on_material(f, mat::UnitedMaterial)
    if mat.typ == UnitedMaterialLambertian
        return f(Lambertian(mat.albedo))
    elseif mat.typ == UnitedMaterialMetal
        return f(Metal(mat.albedo, mat.fuzz))
    end
end
scatter(r_in::Ray, rec::HitRecord{<:Any, <:UnitedMaterial}) = specialize_on_material(m->scatter(r_in, @set rec.material=m), rec.material)


function main(use_cuda, material_approach=1)
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

        material_ground = Lambertian(RGB(0.8, 0.8, 0.0))
        material_center = Lambertian(RGB(0.7, 0.3, 0.3))
        material_left   = Metal(RGB(0.8, 0.8, 0.8), 0.3)
        material_right  = Metal(RGB(0.8, 0.6, 0.2), 1.0)
        if material_approach == 1
            world = ArrType([
                Sphere(Point3( 0.0, -100.5, -1.0), 100.0, UnionMaterialWrapper(material_ground)),
                Sphere(Point3( 0.0,    0.0, -1.0),   0.5, UnionMaterialWrapper(material_center)),
                Sphere(Point3(-1.0,    0.0, -1.0),   0.5, UnionMaterialWrapper(material_left)),
                Sphere(Point3( 1.0,    0.0, -1.0),   0.5, UnionMaterialWrapper(material_right)),
            ])
        elseif material_approach == 2
            materials = ArrType{UnionMaterial{T}}([material_ground, material_center, material_left, material_right])
            materials_dev = ArrType == CuArray ? cudaconvert(materials) : materials
            world = ArrType([
                Sphere(Point3( 0.0, -100.5, -1.0), 100.0, (;list=materials_dev, idx=1)),
                Sphere(Point3( 0.0,    0.0, -1.0),   0.5, (;list=materials_dev, idx=2)),
                Sphere(Point3(-1.0,    0.0, -1.0),   0.5, (;list=materials_dev, idx=3)),
                Sphere(Point3( 1.0,    0.0, -1.0),   0.5, (;list=materials_dev, idx=4)),
            ])
        elseif material_approach == 3
            world = ArrType([
                Sphere(Point3( 0.0, -100.5, -1.0), 100.0, UnitedMaterial(material_ground)),
                Sphere(Point3( 0.0,    0.0, -1.0),   0.5, UnitedMaterial(material_center)),
                Sphere(Point3(-1.0,    0.0, -1.0),   0.5, UnitedMaterial(material_left)),
                Sphere(Point3( 1.0,    0.0, -1.0),   0.5, UnitedMaterial(material_right)),
            ])
        end


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

# #

matapproach = 1

main(false, matapproach)
main(true, matapproach)

##

main(false, matapproach)
main(false, matapproach)
main(false, matapproach)
main(true, matapproach)
main(true, matapproach)
main(true, matapproach)
