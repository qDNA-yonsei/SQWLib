<div align="center">
 
# SQWLib: Simulator for quantum walks.
Sergio A. Ortega and Daniel K. Park

Department of Statistics and Data Science, Yonsei University, Seoul 03722, Republic of Korea

[![arXiv](http://img.shields.io/badge/arXiv-2606.14226-B31B1B.svg)](https://arxiv.org/abs/2606.14226)
<!--  
[![Journal](http://img.shields.io/badge/...)](...)
-->  

</div>
 
## Description
This package is a simulator of the Szegedy Quantum Walk allowing the efficient simulation on sparse graphs, and also expanding the library SQUWALS for dense graphs with further operators. Moreover, it also provides a module for the simulation of quantum phase estimation algorithms based on Szegedy quantum walk.

## Installation  
Open a system's console or an Anaconda Prompt depending on your python installation. Install the package using pip and git.

```bash
pip install git+https://github.com/qDNA-yonsei/SQWLib
```

<!--  
First, clone the repository.
```bash
git clone https://github.com/qDNA-yonsei/SQWLib
```
This creates a folder called SQWLib. Change the directory to it.
```bash
cd SQWLib
```
Install the package using pip.
```bash
pip install .
```
-->  

Alternativelly, you can download the folder sqwlib and copy it in your python working directory, or in some directory included in PYTHONPATH.

## Optional

In order to use the operators for dense graphs, it is necessary to install the library SQUWALS, available at https://github.com/OrtegaSA/squwals-repo.

## Tutorials and Examples
There are tutorials for using the sparse simualtor and the quantum phase estimation modules in the folder Tutorials. Example notebooks for QPE-based algorithms are in the folder Examples. 

### Citation 

```
@article{ortega2026,
  title={SQWLib},
  author={Ortega, S. A. and Park, D. K.},
  journal={arXiv:2606.14226},
  year={2026},
}
```
<!---
```
@article{ortega2026,
	title={SQWLib},
	author={Ortega, S. A. and Park, D. K.},
	journal={...},
  volume = {...},
	pages = {...},
	year={...}
}
```
-->
