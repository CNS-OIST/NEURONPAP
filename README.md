# Welcome to the NEURON model for depolarization of Perisynaptic Astrocyitc Processes(PAPs) 
## Active enhancement of synapse driven depolarization of perisynaptic astrocytic processes
- Ryo J. Nakatani(ryo.nakatani@oist.jp)

Last Modified: Wed, 29 May 14:21:54 JST 2024

# introduction
Welcome to the NEURON model for depolarization of Perisynaptic Astrocyitc Processes(PAPs) 
This model tries to explore the electrophysiolgical properties at PAPs.
It is the model used in the paper,

"Active enhancement of synapse driven depolarization of perisynaptic astrocytic processes"
by RJ Nakatani and E De Schutter

# installation
Clone this git repository to your computer. The software runs using NEURON and python3.
The model has been tested on NEURON ver 8.2 and python 3.10 using MacOSX Ventura 13.0 on a Apple M1 Ultra cpu.


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

To run in parallized mode (only specific experimental protocols will utilize paralization),
run the code with ``` mpiexec -n [numProc] python NEURONPAP.py [args]```

```
usage: NEURONPAP.py [-h] [-r] [-o] [-s] [-n] [-d] [-c] [--spillOver] [--kComp] [--ekComp] [--koComp] [--gluSpill] [-v] [-b] [-p] [--vClamp] [--debug] [--ko KO] [--PAPCount PAPCOUNT] [--stimCount STIMCOUNT] [--ExpVm] [--ek EK]
                    [--stim] [--delay DELAY] [--stimGlu] [--stimK] [--GluT GLUT] [--NMDAR NMDAR]
                    [seed]

Hippocampal astrocyte model implemented in NEURON

positional arguments:
  seed                  The seed for the simulation. Will use a randomly generated seed from 0-100 if none are provided.

options:
  -h, --help            show this help message and exit
  -r, --mRi             measure input Resistance
  -o, --optimize        flag for optimization (works with input Resistance Flag
  -s, --singleRun       Used to run a single simulation
  -n, --nonhoc          Used to run a single simulation in nonhoc
  -d, --distance        Used to run simulations for various distances
  -c, --channels        Used to run simulations for various channel counts
  --spillOver           Used to run simulations with spill over, only works for KV Phase plot
  --kComp               Used to run simulations for various potassium conc. Use --stimGlu to turn on Glutamate Stim.
  --ekComp              Used to run simulations for comparison of ek clamp
  --koComp              Used to run simulations for comparing various KO conditions
  --gluSpill            Used to run simulations for glutamate spillover.
  -v, --video           Used to run simulations for video
  -b, --branch          Used to run simulations for branch attenuation. Use --stimGlu to turn on Glutamate stimulus.
  -p, --phase           Used to run simulations for phase plot
  --vClamp              Used to run v clamp simulations
  --debug               Used to employ debug mode. You can use to run with verbose comments. Mainly used to test if the simulation will construct -> run -> save properly
  --ko KO               Used to set extracellular potassium of simulation. Default: 0.5 mM.
  --PAPCount PAPCOUNT   Used to set number of active PAPs in simulation. Default: 1
  --stimCount STIMCOUNT
                        Used to set the number of potassium stimulations in the simulation. Default: 0.5 mM at 100 Hz.
  --ExpVm               Used when optimizing Vm
  --ek EK               Used to set the ek; only set when called. Works only during channel comparison experiment and single run
  --stim                Called for running experiments with specific stimulus setting. Run along with option --stimGlu --stimK to turn on the specific stimulus. Glu, K are both initially set to False when --stim is called. With
                        --delay you can delay the glutamate stimulus
  --delay DELAY         Used to set the delay of glutamate stimulus after K stimulus
  --stimGlu             Works with --kComp, --branch to turn on/off glutamate stim. (With option --stimGlu it is turned on.)
  --stimK
  --GluT GLUT           Used to set Glutamate Transporters; 1 True; 0 False
  --NMDAR NMDAR         Used to set NMDAR; 1 True; 0 False
```

# directory explantion
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
          MODL mechanisms  
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
          parallization code  
        - textSDIO.py
          print message formatter  
    
    - results  
      Results for figures  
    - morphResults  
      Results for Morphology  

# Requirements
The model uses NEURON ver 8.2 and python 3.10.
Please install,
- NEURON 8.2 >=
- python 3.10 >=

Any other version has not been tested.

## Python Requirements
Below is a list of python dependencies
- neuron 8.2 >=
- matplotlib 3.8.4 >= 
- json 2.0.9 >= 
- mpi4py 3.1.4 >=
- scipy 1.11.4 >=
- numpy 1.26.4 >= 
- pandas 2.2.2 >=
- tqdm 4.66.4 >=
