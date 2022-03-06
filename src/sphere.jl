using LinearAlgebra: norm_sqr

export Sphere

struct Sphere{T} <: Hittable
    center::Point3{T}
    radius::T
end

function hit(s::Sphere, r::Ray, t_min, t_max)::Union{HitRecord, Nothing}
    oc = r.orig - s.center
    a = norm_sqr(r.dir)
    half_b = oc ⋅ r.dir
    c = norm_sqr(oc) - s.radius^2
    discriminant = half_b^2 - a*c
    if discriminant < 0
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
