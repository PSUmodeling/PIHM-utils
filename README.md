# PIHM-utils
Library for reading [MM-PIHM](https://github.com/PSUmodeling/MM-PIHM) input and output files.

## Installation

To install:

```shell
pip install PIHM-utils
```

## Usage

The `read_grid` function reads domain setup from MM-PIHM `.mesh` and `.riv` input files:

```python
from pihm import read_grid

element_df, river_df, node_df = read_grid(pihm_dir, simulation)
```

`pihm_dir` is the path to the MM-PIHM directory, which should contain `input` and `output` directories,
and `simulation` is the name of the simulation.
`element_df`, `river_df`, and `node_df` are `pandas.DataFrame`s that contain grid elements, river segments, and grid nodes information.

The `read_output` function reads MM-PIHM simulation output files:

```python
from pihm import read_output

desc, df = read_output(pihm_dir, simulation, outputdir, var)
```

`desc` is strings containing description and unit of the specific output variable,
`df` is a `pandas.DataFrame` containing the simulation output.
`outputdir` is the name of the output directory,
and `var` is name of output variable.
For a complete list of available output variables, please refer to the MM-PIHM User's Guide.

## Examples

Please check out the [interactive Python notebook](https://colab.research.google.com/drive/1uD7ErWWUb5TFfOos6eQiX_5WZw-SV58h?usp=sharing) for a visualization example.
