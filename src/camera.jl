
export Camera

struct Camera{T}
    origin::Point3{T}
    lower_left_corner::Point3{T}
    horizontal::Vec3{T}
    vertical::Vec3{T}
end

function Camera{T}() where T
    aspect_ratio = T(16) / T(9)
    viewport_height = T(2)
    viewport_width = aspect_ratio * viewport_height
    focal_length = T(1)

    origin = Point3{T}(0, 0, 0)
    horizontal = Vec3{T}(viewport_width, 0, 0)
    vertical = Vec3{T}(0, viewport_height, 0)
    lower_left_corner = origin - horizontal/T(2) - vertical/T(2) - Vec3{T}(0, 0, focal_length)

    return Camera(origin, lower_left_corner, horizontal, vertical)
end

function get_ray(cam, u, v)
    return Ray(cam.origin, cam.lower_left_corner + u*cam.horizontal + v*cam.vertical - cam.origin)
end
