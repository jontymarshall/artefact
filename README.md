# Analytical Radiative Transfer Engine Focused on the Architectures of Circumstellar Things

This is a small piece of code to calculate the continuum emission from a dust disc around a star in the optically thin limit. 

The repository contains the artefact code along with a worked example including the necessary data files. The code can take input stellar models from a variety of sources including direct user defined input, a blackbody assumption, VOSA stellar models (https://svo2.cab.inta-csic.es/theory/newov2/), and from SDB JSON files (https://sdb.drgmk.com). Disc architecture is likewise flexibly defined with 1- and 2- power law models, a Gaussian belt, or user defined distribution. Dust grain scattering and emission is calculated from optical constants using Mie theory with a power law size distribution of dust grains in conjunction with the belt architecture.

If you use artefact in your research, please cite Marshall et al. (2023) (https://ui.adsabs.harvard.edu/abs/2023MNRAS.521.5940M). This code was written by Jonathan Marshall in collaboration with Robin Young and Shane Hengst. Artefact makes use of several Python packages for its calculations, most notably AstroPy, Numpy, SciPy, and MiePython. In the examples given here I have used 

```
.
├── artefact.py					<- The artefact radiative transfer code
├── artefact_parameters.txt		<- Default parameters file for artefact
├── artefact_script.py			<- Default script file for artefact
├── artefact_visual.py			<- Visualisation tools for use with artefact
├── data
│	├── optical_constants		
│	│	└── astrosil.lnk		<- Astronomical silicate n,k values
│	└── stellar_models		
│		├── A0V_spectrum.txt 	<- A star stellar photosphere model
│		└── G2V_spectrum.txt	<- G star stellar photosphere model
├── example						
│	├── HD104860_example.py   	<- Example script for HD104860
│	├── HD104860_photometry.txt <- Input data for model fitting, obtained from Vizier using PySED
│	└── HD104860_parameters.txt	<- Input parameters to artefact model
├── .gitignore					
├── LICENCE
├── README.md					<- This file
├── requirements.txt			<- List of dependencies for artefact
└── robin						<- Scripts written by Robin Young
```
