# AutoFolio

AutoFolio is an algorithm selection tool,
i.e., selecting a well-performing algorithm for a given instance [Rice 1976].
In contrast to other algorithm selection tools,
users of AutoFolio are bothered with the decision which algorithm selection approach to use
and how to set its hyper-parameters.
AutoFolio uses one of the state-of-the-art algorithm configuration tools, namely SMAC [Hutter et al LION'16]
to automatically determine a well-performing algorithm selection approach
and its hyper-parameters for a given algorithm selection data.
Therefore, AutoFolio has a robust performance across different algorithm selection tasks.

## Version

This package is a re-implementation of the original AutoFolio.
It follows the same approach as the original AutoFolio
but it has some crucial differences:

* instead of SMAC v2, we use the pure Python implementation of SMAC (v3)
* less implemented algorithm selection approaches -- focus on promising approaches to waste not unnecessary time during configuration
* support of solution quality scenarios

## License

This program is free software: you can redistribute it and/or modify it under the terms of the 2-clause BSD license (please see the LICENSE file).
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
You should have received a copy of the 2-clause BSD license along with this program (see LICENSE file). If not, see https://opensource.org/licenses/BSD-2-Clause.

## Installation

### Requirements

AutoFolio runs with '''Python 3.5'''.
Many of its dependencies can be fulfilled by using [Anaconda 3.4](https://www.continuum.io/).  

To install (nearly) all requirements, please run:

`pip install -r requirements.txt`

To use pre-solving schedules, [clingo](http://potassco.sourceforge.net/) is required. We provide binary compiled under Ubuntu 14.04 which may not work under another OS. Please put a working `clingo` binary with Python support into the folder `aspeed/`.
 
## Usage

We provide under `scripts` a command-line interface for AutoFolio.
To get an overview over all options of AutoFolio, simply run:

`python3 scripts/autofolio --help`

We provide some examples in `examples/`

### Input Formats 

AutoFolio reads two input formats: CSV and [ASlib](www.aslib.net).
The CSV format is easier for new users but has some limitations to express all kind of input data.
The ASlib format has a higher expressiveness -- please see [www.aslib.net](www.aslib.net) for all details on this input format.

For the CSV format, simply two files are required.
One file with the performance data of each algorithm on each instance (each row an instance, and each column an algorithm).
And another file with the instance features for each instance (each row an instance and each column an feature).
All other meta-data (such as runtime cutoff) has to be specified by command line options (see `python3 scripts/autofolio --help`).

### Cross-Validation Mode

The default mode of AutoFolio is running a 10-fold cross validation to estimate the performance of AutFolio.

### Prediction Mode

If you want to use AutoFolio to predict for instances not represented in the given data,
you need to train AutoFolio save its internal state to disk (use `python3 scripts/autofolio --save [filename]`).
To predict on a new instance,
please run

`python3 scripts/autofolio --load [filename] --feature_vec [space-separated feature vector]`

### Self-Tuning Mode

To use algorithm configuration to optimize the performance of AutoFolio please use the option `--tune`. 

## Reference

[JAIR Journal Article](http://aad.informatik.uni-freiburg.de/papers/15-JAIR-Autofolio.pdf)

@ARTICLE{lindauer-jair15a,
  author    = {M. Lindauer and H. Hoos and F. Hutter and T. Schaub},
  title     = {AutoFolio: An automatically configured Algorithm Selector},
  volume    = {53},
  journal   = {Journal of Artificial Intelligence Research},
  year      = {2015},
  pages     = {745-778}
}

## Contact

Marius Lindauer: lindauer@cs.uni-freiburg.de
