import Printf.@sprintf
import Plots:Animation, buildanimation  

# allows to generate an animation of 3d plots generated with 3Dplots.jl
nframes     = 41            
fnames      = [@sprintf("%06d.png", k) for k  in 1:nframes] 
print(fnames)  
anim        = Animation("PDF", fnames); #PDF is the folder name which contains the pngs
buildanimation(anim, "T_ML05_3fps.gif", fps = 3, show_msg=false) #set a suitable fps
