module RayTracingInOneJulia

    using LinearAlgebra
    using LinearAlgebra: norm_sqr
    using StaticArrays
    using Images
    using CUDA, KernelAbstractions, CUDAKernels
    using Tullio
    using ChangePrecision

    CUDA.allowscalar(false)


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

    @changeprecision Float32 begin

        function Camera()
            aspect_ratio = 16.0 / 9.0
            viewport_height = 2.0
            viewport_width = aspect_ratio * viewport_height
            focal_length = 1.0

            origin = Point3(0.0, 0.0, 0.0)
            horizontal = Vec3(viewport_width, 0.0, 0.0)
            vertical = Vec3(0.0, viewport_height, 0.0)
            lower_left_corner = origin - horizontal/2 - vertical/2 - Vec3(0.0, 0.0, focal_length)

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

        function ray_color(r::Ray, world)
            rec = hit(world, r, 0.0, Inf)
            if !isnothing(rec)
                return 0.5 * (RGB(rec.normal[1], rec.normal[2], rec.normal[3]) + RGB(1.0))
            end
            unit_direction = normalize(r.dir)
            t = 0.5 * (unit_direction.y + 1.0)
            return (1.0-t) * RGB(1.0, 1.0, 1.0) + t * RGB(0.5, 0.7, 1.0)
        end


        function main(use_cuda=true)

            # Image

            aspect_ratio = 16 / 9
            image_width = 400
            image_height = floor(Int, image_width / aspect_ratio)
            samples_per_pixel = 100

            if use_cuda
                ArrType = CuArray
            else
                ArrType = Array
            end

            image = ArrType{RGB{Float32}}(undef, image_height, image_width)

            # World

            world = ArrType([Sphere(Point3(0.0,0.0,-1.0), 0.5), Sphere(Point3(0.0,-100.5,-1.0), 100.0)])

            # Camera

            cam = Camera()

            # Render

            function render_pixel(i, j)
                pixel_color = RGB(0.0)
                for _=1:samples_per_pixel
                    u = (j-1+rand()) / (image_width-1)
                    v = (image_height-1 - (i-1+rand())) / (image_height-1)
                    r = get_ray(cam, u, v)
                    pixel_color  += ray_color(r, world)
                end
                return clamp01(pixel_color/samples_per_pixel)
            end

            @time begin
                @tullio image[i,j] = render_pixel(i, j)
                ArrType==CuArray && CUDA.synchronize()
                image = ArrType != Array ? Array(image) : image
            end

            # Save

            save("image.png", image)

            image

        end

    end # @changeprecision

end # module RayTracingInOneJulia

# import RayTracingInOneJulia.main
main = RayTracingInOneJulia.main

main(false)
main(true)

main(false)
main(false)
main(false)
main(true)
main(true)
main(true)
