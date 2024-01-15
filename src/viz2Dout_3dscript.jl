# Visualisation script for the 2D MPI solver
using Plots, MAT


@views function vizme2D_mpi()
    #firstplot
    anim = @animate for i in 1:21
        if i == 1
            file        = matopen("viz3D_out/SalineAqui_$(50*i).mat"); C = read(file, "C"); T = read(file, "T"); close(file)

        else
            file        = matopen("viz3D_out/SalineAqui_$(50*i).mat"); C = read(file, "C"); T = read(file, "T"); close(file)
        end 
            fontsize    = 12
            lx, ly, lz = 40.0, 20.0, 20.0
            nx, ny ,nz = size(T,1), size(T,2), size(T,3)
            xc, yc, zc = LinRange(0, lx, nx), LinRange(0, ly, ny), LinRange(0, lz, nz)
            p1 = heatmap(xc, zc, C[:, ceil(Int, ny / 2), :]'; xlims=(xc[1], xc[end]), ylims=(zc[1], zc[end]), aspect_ratio=1, c=:turbo)
            png(p1, "viz3D_out/Saline_C_$(50*i)_new.png")
            p2 = heatmap(xc, zc, T[:, ceil(Int, ny / 2), :]'; xlims=(xc[1], xc[end]), ylims=(zc[1], zc[end]), aspect_ratio=1, c=:turbo)
            png(p2, "viz3D_out/Saline_T_$(50*i)_new.png")

    end
    gif(anim,"viz3D_out/SalineAquifier_C.gif";fps=6)
    anim2 = @animate for i in 1:10
        file        = matopen("viz3D_out/SalineAqui_$(50*i).mat"); C = read(file, "C"); T = read(file, "T"); close(file)
        fontsize    = 12
        lx, ly, lz = 40.0, 20.0, 20.0
        nx, ny ,nz = size(T,1), size(T,2), size(T,3)
        xc, yc, zc = LinRange(0, lx, nx), LinRange(0, ly, ny), LinRange(0, lz, nz)
        p2 = heatmap(xc, zc, T[:, ceil(Int, ny / 2), :]'; xlims=(xc[1], xc[end]), ylims=(zc[1], zc[end]), aspect_ratio=1, c=:turbo)
        png(p2, "viz3D_out/Saline_T_$(50*i)_new.png")
    end
    gif(anim2,"viz3D_out/SalineAquifier_T.gif";fps=6)
    return
end

vizme2D_mpi()