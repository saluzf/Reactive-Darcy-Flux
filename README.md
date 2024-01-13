# Final Project

[![Run tests](https://github.com/saluzf/PDE-on-GPU-FabioSaluz-FinalProject/actions/workflows/CI.yml/badge.svg)](https://github.com/saluzf/PDE-on-GPU-FabioSaluz-FinalProject/actions/workflows/CI.yml)

# Saline Aquifier

The final project expands the previously build [porous convection solver](https://github.com/saluzf/pde-on-gpu-Fabio-Saluz). Instead of only considering temperature influences on the porous transport I now added a concentration dependence. Darcy's law which was used to approximate the porous flow and to compute the density can be expanded by a concentration dependance according to equation (1). The temperature profile including initial and boundary conditions from the porous convection solver remains the same. The concentration intial condition is selected similarly for simplicty. In the beginning the concentration is 0 everywhere except for two hotspots. The hotspots are not in the same position as the temperature intial hotspot. The rising warm and the sinking cold fluid dominate the transport of the concentrations. 


![Alt text](docs/SalineAquifier_C_2im.gif)