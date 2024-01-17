using MAT
using GLMakie
using Plots
import Printf.@sprintf
import Plots:Animation, buildanimation  

"""
Invoked after runing the Plot_Results function if satisfied with the result. Generates an animation from the results.
"""
function GenerateGif()
# allows to generate an animation of 3d plots generated with 3Dplots.jl
nframes     = 41            
fnames      = [@sprintf("%06d.png", k) for k  in 1:nframes] 
print(fnames)  
anim        = Animation("PDF", fnames); #PDF is the folder name which contains the pngs
buildanimation(anim, "T_ML05_3fps.gif", fps = 3, show_msg=false) #set a suitable fps
end

"""
Function reads in the Solver solution from the Saline Aquifier files, plots the 3D result and saves it in a form that allows a 3D animation. Make sure to change the File name and the number of files for the for loop.
"""
function Plot_Results()
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
end



Plot_Results()




