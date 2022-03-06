
export Ray

struct Ray{T}
    orig::Point3{T}
    dir::Vec3{T}
end
at(r::Ray, t) = r.orig + t * r.dir