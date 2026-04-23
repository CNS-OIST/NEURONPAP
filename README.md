# Welcome to the NEURON model for depolarization of Perisynaptic Astrocyitc Processes(PAPs)

## Global Nernstian Dynamics Breakdown for Local Synaptic Stimuli in Astrocytic Depolarization

- Ryo J. Nakatani(<ryo.nakatani@oist.jp>)

Last Modified: Thu, 23 Apr 15:01:54 JST 2026

# introduction

Welcome to the NEURON model for depolarization of Perisynaptic Astrocyitc Processes(PAPs)
This model tries to explore the electrophysiolgical properties at PAPs.
It is the model used in the paper,

"Global Nernstian Dynamics Breakdown for Local Synaptic Stimuli in Astrocytic Depolarization"
by RJ Nakatani and E De Schutter

# installation

Clone this git repository to your computer. The software runs using NEURON and python3.
The model has been tested on NEURON ver 8.2 and python 3.10 using MacOSX Ventura 13.0 on a Apple M1 Ultra cpu and Ubuntu Jammy Jellyfish (22.04.4) on a AMD Ryzen Threadripper PRO 7955WX.

# quick start usage

To run the simulations and acquire the main figures of the paper run the command

```bash
 % cd src
 % bash analysis.sh
```

It is recommended to run in parallel with a machine that has over 8 processes.
The ```analysis.sh``` will try to access the number of available processes before running suitable experimental protocols in parallel.
If you have ```nproc``` command installed use it to check the number of processes available.

# program capabilities

The model called by ```NEURONPAP.py``` runs various experiments used in the paper.
For explanation of experiments use,

```bash
% python NEURONPAP.py -h
```

Which should explain the various command arguments you can utilize to trigger specific experiments.

To run in parallelized mode (only specific experimental protocols will utilize parallelization),
run the code with ```mpiexec -n [numProc] python NEURONPAP.py [args]```

```
usage: NEURONPAP.py [-h] [-r] [-o] [-s] [--intraDiff] [-c] [--GABAComp] [--freqComp] [--spillOver] [--testPhys] [--kComp] [--ekComp] [--koComp] [--gluSpill]
                    [-v] [-b] [-p] [--somaClamp] [--ko KO] [--PAPCount PAPCOUNT] [--stimCount STIMCOUNT] [--freq FREQ] [--optVm] [--ExpVm] [--ExpRMP]
                    [--ek EK] [--stim] [--delay DELAY] [--stimGlu] [--stimGaba] [--stimK] [--GluT GLUT] [--NMDAR NMDAR] [--GABAR GABAR] [--GAP GAP]
                    [--NKA NKA] [--OE OVEREXPRESS] [--bathExperiment] [--shellExperiment] [--shiftExperiment]
                    [seed]

Somatosensory cortical astrocyte model implemented in NEURONHOC with python and CL interface

positional arguments:
  seed                  The seed for the simulation. Will use a randomly generated seed from 0-100 if none are provided.

options:
  -h, --help            show this help message and exit
  -r, --mRi             measure input Resistance
  -o, --optimize        flag for optimization (works with input Resistance Flag
  -s, --singleRun       Used to run a single simulation
  --intraDiff           Used to specifiy if run intracellular diffusion, works only for single run
  -c, --channels        Used to run simulations for various channel counts
  --GABAComp            Used to run simulations for various GABA/NMDAR counts
  --freqComp            Used to run simulations for various frequency counts
  --spillOver           Used to run simulations with spill over, only works for KV Phase plot
  --testPhys            Used to run simulations with physiological stimuli
  --kComp               Used to run simulations for various potassium conc. Use --stimGlu to turn on Glutamate Stim.
  --ekComp              Used to run simulations for comparison of ek clamp
  --koComp              Used to run simulations for comparing various KO conditions
  --gluSpill            Used to run simulations for glutamate spillover.
  -v, --video           Used to run simulations for video
  -b, --branch          Used to run simulations for branch attenuation. Use --stimGlu to turn on Glutamate stimulus.
  -p, --phase           Used to run simulations for phase plot
  --somaClamp           Used to run v clamp simulations
  --ko KO               Used to set extracellular potassium of simulation. Default: 0.5 mM.
  --PAPCount PAPCOUNT   Used to set number of active PAPs in simulation. Default: 1
  --stimCount STIMCOUNT
                        Used to set the number of potassium stimulations in the simulation. Default: 0.5 mM at 100 Hz.
  --freq FREQ           Used to set the frequency in Hz
  --optVm               Used when optimizing Vm
  --ExpVm               Used when optimizing Vm to experimental values
  --ExpRMP              Used when optimizing RMP
  --ek EK               Used to set the ek; only set when called. Works only during channel comparison experiment and single run
  --stim                Called for running experiments with specific stimulus setting. Run along with option --stimGlu --stimK to turn on the specific
                        stimulus. Glu, K are both initially set to False when --stim is called. With --delay you can delay the glutamate stimulus
  --delay DELAY         Used to set the delay of glutamate stimulus after K stimulus
  --stimGlu             Works with --kComp, --branch to turn on/off glutamate stim. (With option --stimGlu it is turned on.)
  --stimGaba            Works with --kComp
  --stimK
  --GluT GLUT           Used to set Glutamate Transporters; 1 True; 0 False
  --NMDAR NMDAR         Used to set NMDAR; 1 True; 0 False
  --GABAR GABAR         Used to set GABAR; 1 True; 0 False
  --GAP GAP             Used to set GAP; 1 True; 0 False (KComp runs opposite)
  --NKA NKA             Used to set NKA; 1 True; 0 False
  --OE OVEREXPRESS      Used to set overexpression; 1 True; 0 False
  --bathExperiment      Used to run all bath experiments
  --shellExperiment     Used to run shell experiments
  --shiftExperiment     Used to run shift PAP experiments
```

# directory explanation

    - src
      - NEURONPAP.py  
        The main program that takes arguments to run the experimental protocols  
        - Geometry  
          Morphology of model  
          - GeometryAstrocyteCA1.hoc  
            savtchenko et al. Morphology  
          - threeCompartmentGeom.hoc  
            simple morphology  
        - neuronHoc  
          NEURON core of model  
          - astrocyte.hoc  
            core model  
        - neuronMOD  
          MODL mechanisms  (contains unused)
          - GluTrans.mod
          - Kir2.mod
          - SynExp5NMDA.mod
          - TWIK.mod
          - [cl,na,k]leak.mod
          - [potassium,sodium]Accum.mod
          - inputRes.mod
        - analysis.sh  
          shell script to run all experimental protocols  
        - astrocyte.py  
          python wrapper of NEURON HOC model  
        - experiments.py  
          simulation/experimental protocols  
        - geneManip.py
          Code to access and robustly alter conductances  
        - classResults.py  
          Code for workaround for NEURON Parallization  
        - CLIargParser.py  
          Code for command line interpretaion for NEURONPAP.py
        - utils.py  
          parallization code and etc. 
        - textSDIO.py
          print message formatter  
    
# Requirements

The model uses NEURON ver 8.2 and python 3.10.
Please install,

- NEURON 8.2
- python 3.10 >=

Any other version has not been tested.

## Python Requirements

Contains a ```./src/pyproject.toml``` listing python dependencies. Using venv uv is recommended.
