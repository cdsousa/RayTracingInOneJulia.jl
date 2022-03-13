
export Lambertian, Metal


abstract type Material end

struct Lambertian{T} <: Material
    albedo::RGB{T}
end

function scatter(r_in::Ray{T}, rec::HitRecord{T, Lambertian{T}}) where T
    scatter_direction = rec.normal + normalize(randn(Vec3{T}))
    # Catch degenerate scatter direction
    if iszero(scatter_direction)
        scatter_direction = rec.normal
    end
    return (attenuation=rec.material.albedo, scattered=Ray(rec.p, scatter_direction))
end

struct Metal{T} <: Material
    albedo::RGB{T}
end

reflect(v, n) = v - 2*(v⋅n)*n

function scatter(r_in::Ray{T}, rec::HitRecord{T, Metal{T}}) where T
    reflected = reflect(normalize(r_in.dir), rec.normal)
    scattered=Ray(rec.p, reflected)
    if scattered.dir ⋅ rec.normal > 0
        return (;attenuation=rec.material.albedo, scattered)
    else
        return nothing
    end
end