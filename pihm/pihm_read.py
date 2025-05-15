import numpy as np
import os
import pandas as pd
import struct
from io import StringIO
from pihm_output import OUTPUT

# Compatible with MM-PIHM v1.x
def _simulation_name(simulation):
    if simulation.rfind('.') != -1 and simulation[simulation.rfind('.') + 1:].isnumeric():
        return simulation[0:simulation.rfind('.')]
    else:
        return simulation


def _read_mesh(pihm_dir, simulation):
    # Read mesh file into an array of strings with leading white spaces removed
    # Line starting with "#" are not read in
    with open(f'{pihm_dir}/input/{simulation}/{simulation}.mesh') as f:
        strs = [line.strip() for line in f if line.strip() and line.strip()[0] != '#']

    # Read number of elements
    n_elements = int(strs[0].split()[1])

    element_df = pd.read_csv(
        StringIO('\n'.join(strs[2:2 + n_elements])),
        sep=r'\s+',
        header=None,
        names=['element', 'node1', 'node2', 'node3', 'neighbor1', 'neighbor2', 'neighbor3'],
        index_col='element'
    )

    node_df = pd.read_csv(
        StringIO('\n'.join(strs[4 + n_elements:])),
        sep=r'\s+',
        header=None,
        names=['node', 'x', 'y', 'zmin', 'zmax'],
        index_col='node'
    )

    # PIHM element indicies are 1-based. Add a dummy row to make it 0-based
    node_df.loc[0] = [node_df[col].mean() for col in node_df.columns]
    node_df.sort_index(inplace=True)

    return element_df, node_df


def _read_river(pihm_dir, simulation):
    # Read river file into an array of strings with leading white spaces removed
    # Line starting with "#" are not read in
    with open(f'{pihm_dir}/input/{simulation}/{simulation}.riv') as f:
        strs = [line.strip() for line in f if line.strip() and line.strip()[0] != '#']

    # Read number of river segments
    n_river_segments = int(strs[0].split()[1])

    # Read nodes and outlets information
    df = pd.read_csv(
        StringIO('\n'.join(strs[2:2 + n_river_segments:])),
        sep=r'\s+',
        header=None,
        usecols=[0, 1, 2, 3],
        names=['river', 'from', 'to', 'down'],
        index_col='river',
    )

    return df


def read_grid(pihm_dir: str, simulation: str) -> tuple[pd.DateFrame, pd.DataFrame, pd.DataFrame]:
    """ Read MM-PIHM grid setup
    """
    simulation = _simulation_name(simulation)

    element_df, node_df = _read_mesh(pihm_dir, simulation)
    river_df = _read_river(pihm_dir, simulation)

    element_df['triangle'] = element_df[['node1', 'node2', 'node3']].apply(
        lambda x: [x['node1'], x['node2'], x['node3']],
        axis=1,
    )
    element_df['neighbor'] = element_df[['neighbor1', 'neighbor2', 'neighbor3']].apply(
        lambda x: [x['neighbor1'], x['neighbor2'], x['neighbor3']],
        axis=1,
    )

    element_df.drop(columns=['node1', 'node2', 'node3', 'neighbor1', 'neighbor2', 'neighbor3'], inplace=True)

    return element_df, river_df, node_df


def read_output(pihm_dir: str, simulation: str, output_dir: str, extension: str) -> tuple[str, pd.DataFrame]:
    """Read MM-PIHM output file
    """

    # Read number of river segments and elements from input files
    element_df, river_df, _ = read_grid(pihm_dir, simulation)

    # Default values
    description = 'MM-PIHM output (N/A)'

    # Full file name (binary file)
    fn = f'{pihm_dir}/output/{output_dir}/{simulation}.{extension}.dat'

    # Check size of output file
    file_size = int(os.path.getsize(fn) / 8)

    # Determine output dimension, variable name and unit from extension
    dimension = len(river_df) if extension.startswith('river.') else len(element_df)

    with open(fn, 'rb') as binfile:
        # Read binary output file
        data_tuple = struct.unpack('%dd' %(file_size), binfile.read())

        # Rearrange read values to numpy array
        data_array = np.resize(data_tuple, (int(file_size / (dimension + 1)), dimension + 1))

    df = pd.DataFrame(
        data_array,
        columns=['time'] + [k + 1 for k in range(dimension)],
    )

    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    # Find output description and unit
    for key, func in OUTPUT.items():
        if extension.lower().startswith(key):
            description = func(extension)

            return description, df
    else:
        raise ValueError(f'Unknown output type: {extension}')
