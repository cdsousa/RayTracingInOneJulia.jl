
abstract type Hittable end

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

function hit(hittables::AbstractArray, r::Ray, t_min, t_max)
    rec = nothing
    closest_so_far = t_max

    for object in hittables
        temp_rec = hit(object, r, t_min, closest_so_far)
        if !isnothing(temp_rec)
            closest_so_far = temp_rec.t
            rec = temp_rec
        end
    end

    return rec
end