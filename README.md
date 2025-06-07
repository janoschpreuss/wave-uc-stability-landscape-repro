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


## Fig. 2 & 3
Change to directory `scripts`. Run

    python3 


## Fig. 4
Change to directory `scripts`. Run

    python3 


## Fig. 5

    python3 

## Fig. 6
Change to directory `scripts`. Run

    python3 Solve-cylinder-finite-trace.py

Data files of the form "Cylinder--q2-qstar0-k2-kstar1msol2-Mmodes__M__.dat.dat". Here __M__ in [1,2,3] 
describes the dimension of the space V_M as in the paper. The data files contains the following columns: 

* deltat: mesh width for time discretization 
* L2-err-B: L2-error in the set B 
* L2-err-Bcompl: L2-error in the complement of B so in Q \ B
* L2-err-omega: L2-error in the data set 
* Qall: L2-error in the entire space time cylinder Q 

!!!! NEED TO GENERATE V_M = empty data as well !!!!!

To generate Fig 6, switch to the folder `plots` and run 

    latexmk -pdf Cylinder-Trace-msol2-q2k2.tex


## Fig. 7

Change to directory `scripts`. Run

    python3 Solve-cylinder-finite-trace-approx.py 

Data files of the form "Cylinder--q2-qstar0-k2-kstar1msol3-Mmodes2-eta-tenthousand.dat__X__.dat.dat". 
Here __X__ in [ten,hundred,thousand,tenthousand] describes the value of 1/eta.  
To generate Fig 7, switch to the folder `plots` and run 

    latexmk -pdf Cylinder-Trace-msol3-approx.tex


## Fig. 8

Change to directory `scripts`. Run

    python3 Solve-cylinder-trace-constant-M.py 

The data for the constant C_M^opt obtained with the optimal (one-dimensional) space is contained in the 
generated file "Cylinder-Const--q2-qstar0-k2-kstar1msol-var-Mmodes-1.dat". The data for the constant 
C_M obtained with the space of dimension M is contained in the file Cylinder-Const--q2-qstar0-k2-kstar1msol-eq-Mmodes.dat. 
To generate Fig 7, switch to the folder `plots` and run 

    latexmk -pdf constant-M_pres.tex


