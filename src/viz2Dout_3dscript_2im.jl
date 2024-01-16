# Visualisation script for the 2D MPI solver
using Plots, MAT


@views function vizme2D_mpi()
    #firstplot
    anim = @animate for i in 1:41
        if i == 1
            file        = matopen("viz3D_out/SalineAqui_$(i).mat"); C = read(file, "C"); T = read(file, "T"); close(file)

        else
            file        = matopen("viz3D_out/SalineAqui_$(50*(i-1)).mat"); C = read(file, "C"); T = read(file, "T"); close(file)
        end 
        fontsize    = 12
        lx, ly, lz = 40.0, 20.0, 20.0
        nx, ny ,nz = size(T,1), size(T,2), size(T,3)
        xc, yc, zc = LinRange(0, lx, nx), LinRange(0, ly, ny), LinRange(0, lz, nz)
        p1 = plot(heatmap(xc, zc, C[:, ceil(Int, ny / 2), :]'; xlims=(xc[1], xc[end]), ylims=(zc[1], zc[end]), aspect_ratio=1, c=:turbo,
        title="Concentration"), heatmap(xc, zc, T[:, ceil(Int, ny / 2), :]'; xlims=(xc[1], xc[end]), ylims=(zc[1], zc[end]), aspect_ratio=1, c=:turbo,
        title="temperature"), size=(720,230))
        # ,clim=(-151,151)
        png(p1, "viz3D_out/IsoSalineAqui_$(50*i)_2im.png")

    end
    gif(anim,"viz3D_out/IsothermalSalineAqui_daint.gif";fps=2)

    return
end

vizme2D_mpi()