# M1 Cortex
#### Code by Ziao Chen, Tyler Banks and Matthew Stroud. In partnership with Dr. Headley.
Modelling Beta-Gamma Oscillations in the primary motor cortex of a Wistar rat.


## Running the Model

### 1. Build network configuration files

Building of the network can be customized by altering `build_network.py` configuration at the beginning of the script and enable the code block at the end.

To generate necessary network specification files in the `./network` directory, execute
```
python build_network.py
```

The configuration file and the network folder need to be built only once. Configurations can be changed by directly editing the configuration file. Some network parameters can be changed directly in the .csv files in the network directory.

### 2. Build network input files :

To generate thalamic and M2 input using
```
python build_input.py
```

To see parameters that can be specified for building the input, use
```
python build_input.py -h
```

Once this step has been completed, input files **WILL NOT** need to be regenerated again.

### 3. Execute run script

Run network using any one of the simulation configuration files (`python run_network.py [configuration file] [use coreneuron (bool)]`)

| Configuration file | Input Details |
|--------------------|---------------|
| [config_baseline.json](./config_baseline.json) | 20 Hz baseline input |
| [config_short.json](./config_short.json) | 50 Hz short pulse |
| [config_long.json](./config_long.json) | 20 Hz long pulse |


#### Run in Parallel
Simulation of 10000 Cell models for 16 seconds run for **20-25 minutes** on ~50 cores.
```
mpirun -n 50 nrniv -mpi -python run_network.py config_baseline.json
```

Or... Use Slurm to run:
```
sbatch batchfile_newserver.sh
```

Change the `TRIALNAME` parameter in the batch file will move the simulation output files to a folder named by `TRIALNAME` under the directory [Analysis/simulation_results/](../Analysis/simulation_results).
```
TRIALNAME="baseline_0"
```

### Analysis of the model

Analysis of the model is primarily comprised of a spike raster, mean firing rates, raw LFP, and LFP PSD. Launch using:
```
python analysis.py
```

To simply get **firing rates** for quick analysis run
```
python analysis.py --no-plots
```

Plots can also be generated in MATLAB. A spike raster, mean firing rates, LFP and LFP PSD on 4 separate graphs.
```
analysis('../outputECP/ecp.h5','../outputECP/spikes.h5');
```


#### [ipsc_analysis.m](./matlab/ipsc_analysis.m)

Used in conjuction with [simulation_configECP_base_vclamp.json](./simulation_configECP_base_vclamp.json) and [simulation_configECP_vpsi_vclamp.json](./simulation_configECP_vpsi_vclamp.json) to sum igaba currents and perform a PSD to produce the raw signal and PSD plots.
```
ipsc_analysis('../outputECP/syn_report.h5';
```

#### [connection_info.py](./connection_info.py)

Used to print the connectivity between cell types.

### Other important files (files specific to Amygdala theta model. need to change)

| File/Directory | Description |
|------|-------------|
|[connectors.py](./connectors.py)|This is where you should define all connection rules - keeps build_network.py clean |
|[synapses.py](./synapses.py)| When adding new synapse types to be used by `build_network.py` they must be defined here |
|[components/templates/feng.hoc](./components/templates/feng.hoc)| Contains the PN A, PN C and INT hoc cell templates|
|[components/templates/SOM.hoc](./components/templates/SOM.hoc)| Contains the SOM and CR hoc cell templates|
|[components/synaptic_models](./components/synaptic_models)| Contains parameter configuration files for synapses used |
|[components/mechanisms/modfiles](./components/mechanisms/modfiles)| `.mod` definition files for cells and synapses|
|[tuning/current_clamp](./tuning/current_clamp)| Directory of easy to use hoc-based testers/tuners to simulate current injection into cells in the model




## Single Cell Profiling

Cell templates can be found in:
```
./components/templates/
```
Templates used are:
```
CP.hoc
CS.hoc
FSI.hoc
LTS.hoc
```


NOTE from Amygdala Theta source:
To get fi curves for individual cells, run
```
bmtool util cell --template Cell_Af fi
bmtool util cell --template Cell_Cf fi
bmtool util cell --template InterneuronCellf fi
bmtool util cell --template SOM_Cell fi
bmtool util cell --template CR_Cell fi
```

## Network Profiling

### Plot Connection Totals
```
bmtool plot connection total
```

### Plot the connections for a single cell

This is useful for verifying that your connection rules are functioning propertly, especially if they're space dependent

```
python plot_conns.py
```


### BMTOOLS

*Version 0.2.1+*

Install BMTools by running

```
pip install bmtool
```

Help Examples:
```
bmtool --help
bmtool util --help
bmtool util cell --help
bmtool util cell fi --help
```

### BMTK

Built using [BMTK](https://github.com/AllenInstitute/bmtk) checkout 52fee and later. Parallel netcon recording was added 6/29/21.

```
git clone https://github.com/AllenInstitute/bmtk
cd bmtk
python setup.py develop
````


## Appendix

### Installing NEURON and BMTK

See [https://github.com/tjbanks/easy-nrn-install](https://github.com/tjbanks/easy-nrn-install)

## Tuning tutorial

To tune the model, several things must be in done in order.
1. [Tune individual cells](#single-cell-profiling)
2. [Build the network](#2-build-network-configuration-files)
3. [Ensure connectivity is correct](#connection_infopy)
4. [Run the network](#parallel-mode)
5. [Analyze the network](#analysis-of-the-model)

We want connectivity and base firing rates to match literature.
