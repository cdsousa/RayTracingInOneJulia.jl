using Images
using CUDA, KernelAbstractions, CUDAKernels
using Tullio
using ChangePrecision

function main()

    # Image

    image_width = 256
    image_height = 256

    # ArrType = Array
    ArrType = CuArray

    image = ArrType{RGB{Float32}}(undef, image_height, image_width)

    # Render

    @changeprecision Float32 begin
        function render(i, j)
            r = (j-1) / (image_width-1)
            g = (image_height-i) / (image_height-1)
            b = 0.25
            RGB(r,g,b)
        end
    end

    @time begin
        @tullio image[i,j] = render(i,j)
        image = ArrType!=Array ? Array(image) : image
    end

    # Save

    save("image.png", image)

    image
end

main()