# Final Project

[![Run tests](https://github.com/saluzf/PDE-on-GPU-FabioSaluz-FinalProject/actions/workflows/CI.yml/badge.svg)](https://github.com/saluzf/PDE-on-GPU-FabioSaluz-FinalProject/actions/workflows/CI.yml)



## Probelm Description

Four increasingly difficult problems are solved all based on the same problem. Each new problem introduces one additional chemical or physical problem. 
The start for the analysis is  built upon the previously build [porous convection solver](https://github.com/saluzf/pde-on-gpu-Fabio-Saluz), designed in the first project. The isothermal saline aquifier is an alternative approach to the porous convection. The underlying darcy flux formulation and the density can be written dependent on the concentration instead of temperature. Both the first and second project of the isothermal saline aquifier and the thermal porous convection solver are then combined to a saline aquifier where both temperature and concentration changes are considered. The temperature is more the driving force of the flow dragging along the concentration. Finaly the reactive saline aquifier is the final and most advanced version of the [porous convection solver](https://github.com/saluzf/pde-on-gpu-Fabio-Saluz). Additionaly to the saline aquifier setup a reactive term is included. It simulates the dissolution of SiO2 out of calcium into an aqueous solution. Intialy there is no SiO2 in the fluid. The reaction is temperature dependent and warm streams increase the amount of SiO2 that dissolves into the water while the cold stream decreases it. Higher temperature increase the reaction and the salt concentration increases. Increasesed salt concentration leads to higher density countering the lower density caused by the increased temperature. The concentration profile evolves based on the hot and cold streams. The transport of the species becomes a dominating factor for the salt dissolution process. The dissolution reaction is very slow. This allows to neglect influences of the reactions on the temperature and to assume the porosity to be unafected by the reaction process.


## Methods 

### Physics and Numerics

The continuity equation for the saline aquifier reads as shown in equation 1. With the porosity $\theta$ and the velocity vector $v$ 

![Alt text](./docs/Eq1.png)  

We define the product of the porosity and the velocity as the darcy flux and express it as a function of   
  
![Alt text](./docs/Eq2.png)  

By plugging in the Darcy's flux expression in the continuity equation we obtain: 
  
![Alt text](./docs/Eq3.png)  
  
Density formulation for both temperature and concentration for aqueous solution as in equation 4: We consider fluid and dissolved salt = 2 species. Formulation from Chapter 5.4 in Bortoli, De AL, Greice Andreis, and Felipe Pereira. Modeling and simulation of reactive flows. Elsevier, 2015.
High salt concentration in the aqueous solution increases the density of the fluid while high temperature decreases the density. Therefore the thermal expansion coefficient is usually positive while the concentraiton expansion coefficient is negative. 

![Alt text](./docs/Eq4.png)   
  
By solving for 1 concentration one obtains the other from mass conservation. With
the mass fraction balance for 2 species 1 = y1 + y2 and y = C·MW where y is
the mass fraction and MW the molecular weight we obtain:  

![Alt text](./docs/Eq5.png)  

By plugging in the density formulation in equation 4 into the darcy flux formulation of equation 2 we obtain equation 6 now depending on both concentration and temperature. 

![Alt text](./docs/Eq6.png)  

The dissolution of salt from the solid is a very slow reaction. Therefore the heat produced is very low and neglected for the current analysis. This allows to reuse the energy balance derived in the porous convection project. 

![Alt text](./docs/Eq7.png)  

We can formulate the concentration transport equation based on advection diffusion and a potential reactive term.   

![Alt text](./docs/Eq8.png)  

Combining all the equations yields equation 9. 

![Alt text](./docs/Eq9.png)  

As in the porous convection project the system of equation is solved with a pseudo transient method. Each system is expanded by a pseudo physical term depending on the pseudo time $\tau$.

![Alt text](./docs/Eq10.png)  


### Chemistry  

For simplicity the fluid is assumed to be an aqueous solution of SiO2. The solid consists of mineral quartz and dissolves slowly into the water according to SiO2(s) <=> SiO2(aq). The reaction constants are taken from: Chapter 5.4 in Bortoli, De AL, Greice Andreis, and Felipe Pereira. Modeling and simulation of reactive flows. Elsevier, 2015. They found log(k)=-13.4 and Ea=90.4kJ/mol. The chemistry is not added to the concentration transport equation. An operator split is applied and the temperature and concentration in each time step is solved first and the reaction rate is computed afterwards. 

### Computations  

The simulations where performed on the [Piz Daint Supercomputer of ETHZ](https://www.cscs.ch/computers/piz-daint) using 4 Nvidia P100 16GB PCIe GPU for each simulation. The following script depicts the settings used for the batch job:
```
#!/bin/bash -l
#SBATCH --job-name="diff2D"
#SBATCH --output=diff2D.%j.o
#SBATCH --error=diff2D.%j.e
#SBATCH --time=08:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account class04


module load daint-gpu
module load Julia/1.9.3-CrayGNU-21.09-cuda


export MPICH_RDMA_ENABLED_CUDA=0
export IGG_CUDAAWARE_MPI=0

srun -n4 bash -c 'LD_PRELOAD="/usr/lib64/libcuda.so:/usr/local/cuda/lib64/libcudart.so" julia --project -O3 --check-bounds=no ReactiveSalineAquifier_3D_multixpu.jl'
```

### Initial and Boundary Conditions

The 

![Alt text](./docs/ReactiveSalineAqui_50_2im.png)

## Results

## Isothermal Saline Aquifier
 
The underlying code can be found in [Isothermal Saline Aquifier](src/IsothermalSalineAquifier_3D_multixpu.jl). The isothermal saline aquifier assumes constant temperature in the whole domain. Changes in the density arise only from concentration changes. 

![Alt text](docs/IsothermalSalineAquifier.gif)

## Non-Isothermal Saline Aquifier

The underlying code can be found in [Non-Isothermal Saline Aquifier](src/SalineAquifier_3D_multixpu.jl). Instead of only considering temperature influences on the porous transport I now added a concentration dependence. Darcy's law which was used to approximate the porous flow and to compute the density can be expanded by a concentration dependance according to equation (1). The temperature profile including initial and boundary conditions from the porous convection solver remains the same. The concentration intials conditions are selected similarly for simplicty. In the beginning the concentration is 0 everywhere except for two hotspots. The hotspots are not in the same position as the temperature intial hotspot. 
 
 The same intial and boundary conditions was simulated 3 times for different volumetric expansion coefficients of the fluid:
 
 The first solution was obtained for $\alpha=10\beta$ The rising warm and the sinking cold fluid dominate the transport of the concentrations and drag along the concentrations. Nevertheless if there is no temperature gradient one can clearly see patches of increased concentration to get advected downwards. 

![Alt text](docs/SalineAquifier.gif)

## Reactive Saline Aquifier

The underlying code can be found in [Reactive Saline Aquifier](src/ReactiveSalineAquifier_3D_multixpu.jl). In a final step the reaction is added to the system. One can clearly see that the temperature is dominating the density changes. The reaction rate is too small to have a significant effect. Therefore temperature is the driving force for the reaction. Because of the temperature dependence of the rection hot streams cause more salt to be edged out into the  

![Alt text](docs/ReactiveSalineAquifier.gif)  

## Outlook

## Literature

1. Bortoli, De AL, Greice Andreis, and Felipe Pereira. Modeling and simulation of reactive flows. Elsevier, 2015.


2. Susan L. Brantley James D. Kubicki Art F. White. Kinetics of Water-Rock Interaction. Springer, 2008.


3. Ludovic Räss,   Mauro Werder,   Samuel Omlin & Ivan Utkin, Fall 2023, [PDE on GPU Course-Webpage](https://pde-on-gpu.vaw.ethz.ch), last accessed: 16.01.2024, (https://pde-on-gpu.vaw.ethz.ch).