# Final Project

[![Run tests](https://github.com/saluzf/PDE-on-GPU-FabioSaluz-FinalProject/actions/workflows/CI.yml/badge.svg)](https://github.com/saluzf/PDE-on-GPU-FabioSaluz-FinalProject/actions/workflows/CI.yml)

# Saline Aquifier

The final project is built upon the previously build [porous convection solver](https://github.com/saluzf/pde-on-gpu-Fabio-Saluz). Instead of only considering temperature influences on the porous transport I now added a concentration dependence. Darcy's law which was used to approximate the porous flow and to compute the density can be expanded by a concentration dependance according to equation (1). The temperature profile including initial and boundary conditions from the porous convection solver remains the same. The concentration intials conditions are selected similarly for simplicty. In the beginning the concentration is 0 everywhere except for two hotspots. The hotspots are not in the same position as the temperature intial hotspot. The rising warm and the sinking cold fluid dominate the transport of the concentrations and drag along the concentrations.  


![Alt text](docs/SalineAquifier_C_2im.gif)

## Methods 

### Physics and Equations

The continuity equation for the saline aquifier reads as shown in equation 1. With the porosity $\theta$ and the velocity vector $v$ 
![Alt text](docs/Eq1.png)  

We define the product of the porosity and the velocity as the darcy flux and express it as a function of   
  
![Alt text](docs/Eq2.png)  

By plugging in the Darcy's flux expression in the continuity equation we obtain: 
![Alt text](docs/Eq3.png)  

Density formulation for both temperature and concentration for aqueous solution as in equation 4: We consider fluid and dissolved salt = 2 species. Formulation from Chapter 5.4 in Bortoli, De AL, Greice Andreis, and Felipe Pereira. Modeling and simulation of reactive flows. Elsevier, 2015.
![Alt text](docs/Eq4.png)  
By solving for 1 concentration one obtains the other from mass conservation. With
the mass fraction balance for 2 species 1 = y1 + y2 and y = C·MW where y is
the mass fraction and MW the molecular weight we obtain:
![Alt text](docs/Eq5.png) 
### Initial and Boundary Conditions

The 
![Alt text](docs/ReactiveSalineAqui_50_2im.png)

## Results


# Reactive Saline Aquifier

The reactive saline aquifier is the final and most advanced version of the [porous convection solver](https://github.com/saluzf/pde-on-gpu-Fabio-Saluz). Additionaly to the saline aquifier setup a reactive term is included. It simulates the dissolution of SiO2 out of calcium into an aqueous solution. Intialy there is no SiO2 in the fluid. The reaction is temperature dependent and warm streams increase the amount of SiO2 that dissolves into the water while the cold stream decreases it. Therefore the concentration profile evolves based on the hot and cold streams. The transport of the species becomes a dominating factor for the salt dissolution process. The dissolution reaction is very slow. This allows to neglect influences of the reactions on the temperature and to assume the porosity to be unafected by the reaction process.

## Methods 

### Chemistry and Equations

### Initial and Boundary Conditions


## Results

