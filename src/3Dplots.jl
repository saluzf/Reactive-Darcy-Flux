using MAT
using GLMakie
using Plots
for it = 1:41
    if it == 1
        file        = matopen("viz3D_out/ReactiveSalineAqui_$(it)_updatedk.mat"); C = read(file, "C"); T = read(file, "T"); close(file)

    else
        file        = matopen("viz3D_out/ReactiveSalineAqui_$(50*(it-1))_updatedk.mat"); C = read(file, "C"); T = read(file, "T"); close(file)
    end 

    lx, ly, lz  = 40, 20.0, 20.0
    nx          = size(C,1)
    ny          = size(C,2)
    nz          = size(C,3)

    xc, yc, zc  = LinRange(0, lx, nx), LinRange(0, ly, ny), LinRange(0, lz, nz)

    fig         = Figure(resolution=(1600, 1000), fontsize=24)
    ax          = Axis3(fig[1, 1]; aspect=(1, 1, 0.5), title="Temperature")

    surf_wCH4 = GLMakie.contour!(ax, xc, yc, zc, T; alpha=0.05, colormap=:turbo, labels=true, transparency =true)
    #Colorbar(fig[1, 1][1, 2],surf_wCH4)
    if it <10
        save("PDF/00000$(it).png", fig)
    else
        save("PDF/0000$(it).png", fig)
    end
    
end








