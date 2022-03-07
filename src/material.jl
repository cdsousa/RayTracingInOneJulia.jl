
export Lambertian

abstract type Material end

struct Lambertian{T} <: Material
    albedo::RGB{T}
end

function scatter(lambertian::Lambertian, ray::Ray{T}, rec::HitRecord) where T
    scatter_direction = rec.normal + normalize(randn(Vec3{T}))
    # Catch degenerate scatter direction
    if iszero(scatter_direction)
        scatter_direction = rec.normal
    end
    return (attenuation=lambertian.albedo, scattered=Ray(rec.p, scatter_direction))
end
