# Mixture of Dirichlet Process Mixtures (MDPM)

This repository contains the source code for the simulations and real-data applications in the paper

> **A Bayesian Approach to Learning Mixtures of Nonparametric Components** 

This repository implements the Bayesian MDPM framework proposed in the paper. It includes three simulation studies, and applications
to astronomical and animal movement data.


## Repository Structure

```bash
NPMixture/
├── LocationMixture/   # Simulation Example 1 (Figures 2a–2b)
├── SpikeSlab/         # Simulation Example 2 (Figures 2c–2d)
├── Multivariate/      # Multivariate simulation (Figure 3)
├── XMM/               # Astronomical data application (Figures 4–5)
├── Shark/             # Shark acceleration data application (Figure 6)
├── results/           # Generated figures and outputs
├── Project.toml       # Julia project environment
├── LICENSE
└── README.md
```


## Setup

1. Install Julia (tested with Julia version 1.12.1).

2. Clone the repository:

```bash
git clone https://github.com/yilei-zhang/NPMixture.git
cd NPMixture
```

3. Instantiate the project environment. Run the following command in your terminal:
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```
This command installs all package dependencies specified in `Project.toml` and `Manifest.toml`. 
To view all installed dependencies, run:
```bash
julia --project=. -e 'using Pkg; Pkg.status()'
```

## Simulations

### Figures 2a and 2b
1. Navigate to the `LocationMixture` directory

```bash
cd LocationMixture
```

2. Run the MCMC script to generate the simulation results

This simulation uses multi-threading by default. To reproduce the results exactly, run the following command with **4 threads**:

```bash
julia --threads=4 --project=.. mcmc.jl
```
This generates the file `loc_mix_mcmc.jld2` in the `results/` directory. Using a different number of threads may lead to slight numerical differences.

To disable multi-threading, run:

```bash
julia --project=.. mcmc.jl false
```
Note that disabling multi-threading may also produce slightly different numerical results.

3. Generate the figures
```bash
julia --project=.. fig2a.jl
julia --project=.. fig2b.jl
```
These commands generate `fig2a.pdf` and `fig2b.pdf` in the `results/` directory.

### Figures 2c and 2d

As in Figures 2a and 2b, to reproduce the results exactly, run the MCMC script with **4 threads**:

```bash
cd SpikeSlab
julia --threads=4 --project=.. mcmc.jl   ## generates spike_slab_mcmc.jld2 in results/
julia --project=.. fig2c.jl              ## generates fig2c.pdf in results/
julia --project=.. fig2d.jl              ## generates fig2d.pdf in results/
```

To disable multi-threading, replace the MCMC command in the second line above with:

```bash
julia --project=.. mcmc.jl false
```
Disabling multi-threading may lead to slight numerical differences.


### Figure 3
The multivariate MDPM implementation relies on the R package `TruncatedNormal` via the `RCall` interface. To enable this functionality, R must be installed on your system. Before running the Julia scripts, install the required R package by executing the following command in an R session:
```r
install.packages("TruncatedNormal")
```
By default, this simulation does not use multi-threading. Run the following commands to reproduce the results exactly:
```bash
cd Multivariate
julia --project=.. mcmc.jl   ## generates mv_mcmc.jld2 in results/
julia --project=.. fig3.jl   ## generates fig3a.png, fig3b.png, fig3c.png and fig3.pdf in results/
```

## Real Data Applications

### XMM Astronomical Data
To reproduce Figures 4 and 5, the dataset `XMM.csv` must be placed in the `XMM/` directory.  The dataset is available from the authors upon request.
This section also requires the R package `TruncatedNormal` to be installed, as described in the Figure 3 section. 
Run the following commands to reproduce the results exactly:

```bash
cd XMM
julia --threads=4 --project=.. mcmc.jl      ## generates xmm_full_mcmc.jld2 in results/
julia --project=.. fig4and5.jl              ## generates fig4.pdf and fig5.pdf in results/
```

### Whitetip Shark Acceleration Data
To reproduce Figure 6, the datasets `sharkdata.csv`, `adSP.csv`, and `fpSP.csv` must be placed in the `Shark/` directory.  These datasets are available from the authors upon request.

Run the following commands to reproduce the results exactly:

```bash
cd Shark
julia --project=.. mcmc.jl      ## generates shark_mcmc.jld2 in results/
julia --project=.. fig6.jl      ## generates fig6.pdf in results/
```