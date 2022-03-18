
export Material, Lambertian, Metal, scatter


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