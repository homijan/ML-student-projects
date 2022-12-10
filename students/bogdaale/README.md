# NN-driven heat flux
Jupyter notebook uses training data generated from Impact kinetic simulation. Go to `NN-examples`
 and generate the data by `python3 generate_QimpactTrainingData.py`. The jupyter notebook `ptl-NNheatflux.ipynb` can be opened via jupyter-notebook server started on the HPC cluster at FJFI `ssh -L 8888:localhost:8888 username@q3b.fjfi.cvut.cz` (once started you can open jupyter in your browser at `http://localhost:8888/?token=...`).
 
# Gadolinium-He hohlraum plasma profile
1. try plotting the profiles of temperature `Te`, electron density `ne`, and ionization `Z` in `./gd-profiles` by calling `python3 plot-gd-profiles.py`
2. add a figure plotting the Knudsen number `Kn` from `Q_gdhohlraum_cm_10ps_c7b-bgk-Wcm2-clogCHIC.txt`
3. evaluate the heat flux constant `k` from the formula `q = - k / Z (Z + 0.24) / (Z + 4.2) T^(2.5) dTdx` by matching the `Qloc` profile.
4. evaluate "nonlocal" fitting constants `alphaC` and `alphaN` from the formula `q = - alphaC / Z (Z + 0.24) / (Z + 4.2) T^alphaN dTdx` by matching `Qimpact` profile.
5. evaluate "nonlocal" fitting constants `alphaC` and `alphaN` above subintervals `(x0, x0+deltax), (x0+deltax, x0+2*deltax), .., (x0+(N-1)*deltax, x1)`, where `x0`, `x1`, and `deltax` are `min(x)`, `max(x)`, and `(max(x)-min(x))/N` from the task 4. Use N = 1, 2, 3, 4. Finally, find a x-dependent fit of `alphaC(x)` and `alphaN(x)` and plot `q = - alphaC(x) / Z (Z + 0.24) / (Z + 4.2) T^alphaN(x) dTdx` for all given values of `N`.
6. Generalize the concept of subintervals to a concept of a *sliding* interval `(xc - deltax/2, xc + deltax/2)` centered around point `xc` of size `deltax` (for example 50 microns). Prepare a function `getAlphas(xi, Zi, Ti, Qi)` returning the value of `alphaC(xc)` and `alphaN(xc)` where `xi, Zi, Ti, Qi` are discrete values restricted to the `deltax`-inerval around `xc`. **Finally, plot `alphaC(x)` and `alphaN(x)` for different fit functions as defined in `Qimpact_MH.py`**
7. Prepare regression data based on the concept of *sliding* interval. For a given point `xc` create a `3N+1` long vector `datapoint = [Zi, Ti, gradTi, Qc]`, where `N` is the number of discrete points in the `deltax`-interval and `Qc` is the value of Qimpact at `xc`. **For each discrete position `xc` in `(min(x) + deltax/2, max(x) - deltax/2)` store  the `datapoint` vector as a row to the data matrix `Qdata`.**
8. Familiarize yourself with the `How it works - Bike Share Regression PyTorch Lightning.ipynb` jupyter notebook from `https://github.com/shotleft/how-to-python` and then apply the same notebook for *heat flux regression* using `Qdata` instead of 'bike_sharing_hourly.csv'.

# Zadani prace pro Alexe

Nazev prace (anglicky):       Machine learning-driven nonlocal hydrodynamics for thermonuclear fusion modeling

Pokyny pro vypracovani:

V ramci bakalarske prace provedte nasledujici ukoly:

1) Get acquainted with the state-of-the-art of the inertial confinement fusion (ICF) research and the importance of the physical phenomena of transport [1, 2, 3].

2) Research hydrodynamic models used in ICF with focus on nonlocal electron transport [4]. 

3) Process kinetic modeling data provided by Lawrence Livermore National Laboratory.

4) Teach a deep neural network (DNN) to learn the process of nonlocal electron transport based on physically motivated loss function [5, 6, 7].

5) Compare the DNN model with the classical heat flux limiter model used in ICF [8].

[1] H. Abu-Shawareb et al. (Indirect Drive ICF Collaboration), "Lawson Criterion for Ignition Exceeded in an Inertial Fusion Experiment", Physical Review Letters 129, 075001 (2022).

[2] D. T. Casey, et al., "Evidence of Three-Dimensional Asymmetries Seeded by High-Density Carbon-Ablator Nonuniformity in Experiments at the National Ignition Facility," Physical Review Letters 126, 025002 (2021). 

[3] M. D. Rosen, et al., "The role of a detailed configuration accounting (DCA) atomic physics package in explaining the energy balance in ignition-scale hohlraums," High Energy Density Physics 7 (3), 180-190 (2011).

[4] M. Holec, J. Nikl and S. Weber, "Nonlocal transport hydrodynamic model for laser heated plasmas," Physics of Plasmas 25, 032704 (2018).

[5] PyTorch Lightning Tutorial, https://becominghuman.ai/pytorch-lightning-tutorial-1-getting-started-5f82e06503f6

[6] Introduction to PyTorch Lightning, https://pytorch-lightning.readthedocs.io/en/stable/

[7] Regression using PyTorch Lightning, "Bike Share Regression PyTorch Lightning.ipynb", https://github.com/shotleft/how-to-python.git

[8] D. A. Chapman, et al., "A preliminary assessment of the sensitivity of uniaxially driven fusion targets to flux-limited thermal conduction modeling", Physics of Plasmas 28, 072702 (2021).



Nazev prace (cesky):          Aplikace strojoveho uceni pri nelokalnim hydrodynamickem modelovani plazmatu termojaderne fuze 

Nazev prace (anglicky):       Machine learning-driven nonlocal hydrodynamics for thermonuclear fusion modeling

Pokyny pro vypracovani:

V ramci bakalarske prace provedte nasledujici ukoly:

1) Seznamte se s nejnovejsimi poznatky a smerovanim k uspesnemu dosazeni inercialni termojaderne fuze [1, 2].

2) Seznamte se s modernimi modely pro simulovani fuznich experimentu, predevsim s modely elektronoveho transportu [3, 4, 5].

3) Zpracujte data ziskana z kinetickych modelu Lawrence Livermore National Laboratory [6].

4) Naucte neuronovou sit proces nelokalniho transportu na zaklade spravne definovane objektivni funkce [7].

5) Porovnejte model neuronove site s klasickym modelem limiteru tepelne vodivosti.
