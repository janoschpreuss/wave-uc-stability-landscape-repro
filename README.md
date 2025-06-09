# wave-uc-stability-landscape-repro
This repository contains software and instructions to reproduce the numerical experiments in the paper
> "Unique continuation for the wave equation: the stability landscape
>
> * authors: Erik Burman(1), Lauri Oksanen(2), Janosch Preuss(3) and  Ziyao Zhao(2)
> * (1): University College London
> * (2): University of Helsinki
> * (3): Inria Bordeaux (project team: Makutu)

# <a name="repro"></a> How to reproduce
The `python` scripts for runnings the numerical experiments are located in the folder `scripts`.
To run an experiment we change to this folder and run the corresponding file.
After execution has finished the produced data will be available in the folder `data`.
For the purpose of comparison, the folder `data_save` contains a copy of the data which has been used for the plots in the paper.
The data in both folders should be identical.

To generate the plots as shown in the article from the data just produced we change to the folder `plots`
and compile the corresponding `latex` file.
Below we decribe the above process for each of the figures in the article in detail.
For viewing the generated pdf file, say `figure.pdf`, the figure has to be copied to the host machine.
This can be done by executing the following commands in a new terminal window (not the one in which `docker` is run):

    CONTAINER_ID=$(sudo docker ps -alq)
    sudo docker cp $CONTAINER_ID:/home/app/wave-uc-stability-landscape-repro/plots/figure.pdf \
    /path/on/host/machine/figure.pdf

Here, `/path/on/host/machine/` has to be adapted according to the file structure on the host machine.
The file `figure.pdf` can then be found at the designated path on the host machine and inspected with a common pdf viewer.
(The command above assumes that the reproduction image is the latest docker image to be started on the machine).
Alternatively, if a recent latex distribution is available on the host machine it is also possible to copy data and tex files to the latter and
compile the figures there.


## Figures 2, 3 & 4 
Change to directory `scripts`. Run

    python3 Solve-cylinder-clean-data.py

The following data files will be generated: 

* "Cylinder--q__i__-qstar__j__-k__n__-kstar__m__-msol2.dat" where __i__,__j__,__n__ and __m__ represent the polynomial degrees q, qstar, k and kstar, 
respectively. This data corresponds to Fig. 2 and 3. Addtionally, there are vtk files "2D-cylinder-reflvl1-q2_X.vtk" for X in [0,..7] created which 
correspond to the plots of the difference in B and Q \ B shown in Figure 3.
* "Cylinder--q1-qstar1-k1-kstar1-msol2alpha-__a__.dat" where __a__ in [one, 3quarter, half, quarter] corresponds to the value of the parameter kappa 
in [1,3/4,1/2,1/4]. The vtk files "spacetime_vtk___a___X.vtk" correspond to the plots of the space-time sub-domains shown in Figure 4.


The data files contains the following columns: 

* deltat: mesh width for time discretization 
* L2-err-B: L2-error in the set B 
* L2-err-Bcompl: L2-error in the complement of B so in Q \ B
* L2-err-omega: L2-error in the data set 
* Qall: L2-error in the entire space time cylinder Q 

To generate Figure 2, switch to the folder `plots` and run 

    latexmk -pdf Cylinder-Hoelder-conv.tex

To generate Figure 3, switch to the folder `plots` and run 
 
    latexmk -pdf Cylinder-Hoelder-conv-log-nop3.tex

To generate Figure 4, switch to the folder `plots` and run 

    latexmk -pdf Cylinder-Hoelder-stretch.tex


## Figure 5
The reproduction of this experiment proceeds in the following steps. 
First we have to solve the generalized eigenvalue problem to obtain the mode which is 
later to be used as noise. This step can be skipped when preferred as the data for 
the mode is already available in the file "mode0-ref_lvl4.out". 
If you wish to recompute the mode, change to directory `scripts` and run 

    python3 Cylinder_eig.py 

The second step is to reproduce the data for the visualization of the mass of the mode shown in the upper right
panel of Figure 5. To this end, change to directory `scripts` and run 

    python3 Cylinder-eig-plot-mode.py

Afterwars, the data will be available in the file "Cylinder-bad-mode-mass.dat" in the data folder.
The fist column "t" is the time, the other columns contain the mass in the subdomains corresponding to their names, 
e.g. "mass-B" gives the mass in B at time t and so on. The created vtk files "2D-cylinder-noise-reflvl2-q1___j__.vtk" 
for __j__ in [0,..,15] can be used to reproduce the plot of the mode shown in the upper left panel of the figure 
using paraview. Additionally, the file '2D-cylinder-noise-time.vtu' allows to create a video of the mode.

Now we can proceed to using convergence test with noisy data. Let us first compute the results with noise 
taking from a smooth function (\delta u^s in the paper). To this end, change to directory `scripts` and run 

    python3 Cylinder-eig-normalized-finestep-smooth.py

The created files "Cylinder-q1-qstar1-k1-kstar1-noise-bad-mode-theta__X__.dat" contain the L^2-errors for 
the variable theta=__X__ in [1,2]. That means that the noise is scaled proportial to h^theta.

Let us finally compute taking the computed mode as noise. To this end, change to directory `scripts` and run  

    python3 Cylinder-eig-normalized-finestep.py

The created files "Cylinder--q1-qstar1-k1-kstar1-noise-bad-mode-theta__X__.dat" contain the L^2-errors for 
the variable theta=__X__ in [1,2].  

Finally, to generate Figure 5, switch to the folder `plots` and run 

    latexmk -pdf Cylinder-noise.tex


## Figure 6
Change to directory `scripts`. Run

    python3 Solve-cylinder-finite-trace.py

Data files of the form "Cylinder--q2-qstar0-k2-kstar1msol2-Mmodes__M__.dat.dat". Here __M__ in [1,2,3] 
describes the dimension of the space V_M as in the paper. Note that the data for the case where V_M 
is L^2(\Sigma) is produced during the creation of the data for Figures 2, 3 & 4. 

To generate Figure 6, switch to the folder `plots` and run 

    latexmk -pdf Cylinder-Trace-msol2-q2k2.tex


## Figure 7

Change to directory `scripts`. Run

    python3 Solve-cylinder-finite-trace-approx.py 

Data files of the form "Cylinder--q2-qstar0-k2-kstar1msol3-Mmodes2-eta-tenthousand.dat__X__.dat.dat". 
Here __X__ in [ten,hundred,thousand,tenthousand] describes the value of 1/eta.  
To generate Fig 7, switch to the folder `plots` and run 

    latexmk -pdf Cylinder-Trace-msol3-approx.tex


## Figure 8

Change to directory `scripts`. Run

    python3 Solve-cylinder-trace-constant-M.py 

The data for the constant C_M^opt obtained with the optimal (one-dimensional) space is contained in the 
generated file "Cylinder-Const--q2-qstar0-k2-kstar1msol-var-Mmodes-1.dat". The data for the constant 
C_M obtained with the space of dimension M is contained in the file Cylinder-Const--q2-qstar0-k2-kstar1msol-eq-Mmodes.dat. 
To generate Fig 7, switch to the folder `plots` and run 

    latexmk -pdf constant-M_pres.tex


