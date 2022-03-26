
export Material, Lambertian, Metal, Dielectric, scatter


function random_unit_vector(::Type{T}) where T
    # normalize(randn(Vec3{T}))
    u = T(2) * rand(T) - T(1)
    ϕ = T(2) * T(π) * rand(T)
    v = sqrt(T(1) - u^2)
    Vec3(cos(ϕ) * v, sin(ϕ) * v, u)
end

random_in_unit_sphere(::Type{T}) where {T} = cbrt(rand(T)) * random_unit_vector(T)


abstract type Material end

struct Lambertian{T} <: Material
    albedo::RGB{T}
end

function scatter(r_in::Ray{T}, rec::HitRecord{T, Lambertian{T}}) where T
    scatter_direction = rec.normal + random_unit_vector(T)
    # Catch degenerate scatter direction
    if iszero(scatter_direction)
        scatter_direction = rec.normal
    end
    return (attenuation=rec.material.albedo, scattered=Ray(rec.p, scatter_direction))
end

struct Metal{T} <: Material
    albedo::RGB{T}
    fuzz::T
end

reflect(v, n) = v - 2*(v⋅n)*n

function scatter(r_in::Ray{T}, rec::HitRecord{T, Metal{T}}) where T
    reflected = reflect(normalize(r_in.dir), rec.normal)
    scattered = Ray(rec.p, reflected + rec.material.fuzz * random_in_unit_sphere(T))
    if scattered.dir ⋅ rec.normal > 0
        return (;attenuation=rec.material.albedo, scattered)
    else
        return nothing
    end
end

struct Dielectric{T} <: Material
    ir::T  # Index of Refraction
end

function refract(uv::Vec3{T}, n, etai_over_etat) where T
    cos_theta = min(-uv ⋅ n, T(1))
    r_out_perp =  etai_over_etat * (uv + cos_theta*n)
    r_out_parallel = -sqrt(abs(T(1) - norm_sqr(r_out_perp))) * n
    return r_out_perp + r_out_parallel
end


function scatter(r_in::Ray{T}, rec::HitRecord{T, Dielectric{T}}) where T
    attenuation = RGB(T(1))
    refraction_ratio = rec.front_face ? (T(1)/rec.material.ir) : rec.material.ir
    unit_direction = normalize(r_in.dir)
    refracted = refract(unit_direction, rec.normal, refraction_ratio)
    scattered = Ray(rec.p, refracted)
    return (;attenuation, scattered)
end