# PIHM-utils
Library for reading [MM-PIHM](https://github.com/PSUmodeling/MM-PIHM) input and output files.

## Installation

To install:

```shell
pip install PIHM-utils
```

## Usage

The `read_mesh` function reads mesh information from MM-PIHM `.mesh` input files:

```python
from pihm import read_mesh

num_elem, num_nodes, tri, x, y, zmin, zmax = read_mesh(pihm_dir, simulation)
```

where `num_elem` is the number of triangular elements,
`num_nodes` is the number of nodes, `tri` is an array of triangles consisted of nodes,
`x`, `y`, `zmin`, and `zmax` are arrays of the x and y coordinates, and bottom and surface elevations of nodes, respectively.
`pihm_dir` is the path to the MM-PIHM directory, which should contain `input` and `output` directories,
and `simulation` is the name of the simulation.

The `read_river` function reads river information from MM-PIHM `.river` input files:

```python
from pihm import read_river

num_rivers, from_nodes, to_nodes, outlets = read_river(pihm_dir, simulation)
```

where `num_rivers` is the number of river segments,
`from_nodes` and `to_nodes` are arrays of from and to nodes of river segments,
and `outlets` is an array of river outlets.

The `read_output` function reads MM-PIHM simulation output files:

```python
from pihm import read_output

sim_time, sim_val, desc, unit = read_output(pihm_dir, simulation, outputdir, var)
```

where `sim_time` is an array of simulation time steps,
`sim_val` is an array of output values containing simulation results from all model grids at all model steps,
and `desc` and `unit` are strings containing description and unit of the specific output variable.
`outputdir` is the name of the output directory,
and `var` is name of output variable.
For a complete list of available output variables, please refer to the MM-PIHM User's Guide.

## Examples

Please check out the [interactive Python notebook](https://colab.research.google.com/drive/1uD7ErWWUb5TFfOos6eQiX_5WZw-SV58h?usp=sharing) for a visualization example.
