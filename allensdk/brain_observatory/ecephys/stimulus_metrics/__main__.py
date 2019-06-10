
from argschema import ArgSchemaParser
import time

import pandas as pd
import numpy as np

from functools import reduce

from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession

import allensdk.brain_observatory.ecephys.stimulus_metrics.drifting_gratings as drifting_gratings
import allensdk.brain_observatory.ecephys.stimulus_metrics.static_gratings as static_gratings
import allensdk.brain_observatory.ecephys.stimulus_metrics.natural_scenes as natural_scenes
import allensdk.brain_observatory.ecephys.stimulus_metrics.natural_movies as natural_movies
import allensdk.brain_observatory.ecephys.stimulus_metrics.dot_motion as dot_motion
import allensdk.brain_observatory.ecephys.stimulus_metrics.contrast_tuning as contrast_tuning
import allensdk.brain_observatory.ecephys.stimulus_metrics.flashes as flashes
import allensdk.brain_observatory.ecephys.stimulus_metrics.receptive_field_mapping as receptive_field_mapping


def calculate_stimulus_metrics(args):

    print('ecephys: stimulus metrics module')

    start = time.time()

    functions = ( \
                  drifting_gratings.compute_metrics,
                  static_gratings.compute_metrics,
                  natural_scenes.compute_metrics,
                  natural_movies.compute_metrics,
                  dot_motion.compute_metrics,
                  contrast_tuning.compute_metrics,
                  flashes.compute_metrics,
                  receptive_field_mapping.compute_metrics
                )

    df = reduce(lambda nwb_path, output: \
                 pd.concat(output,
                           add_metrics_to_units_table(nwb_path, functions, args)),
                 args['nwb_paths'],
                 pd.DataFrame())

    df.to_feather(args['output_file_location'])

    execution_time = time.time() - start

    print('total time: ' + str(np.around(execution_time, 2)) + ' seconds\n')

    return {"execution_time" : execution_time}


def add_metrics_to_units_table(nwb_path, functions, args):

    """
    Adds columns to units table for one session, based on the metrics
    for each stimulus type.

    Parameters:
    -----------
    nwb_path : String
        Path to a spikes NWB file
    functions : tuple
        Functions that add unique columns to a units table
        Must be in the form:
            compute_metrics(DataFrame, EcephysSession, args)

    Returns:
    --------
    units_df : pandas.DataFrame
        Units table with new columns appended

    """

    session = EcephysSession.from_nwb_path(nwb_path)

    return reduce(lambda x, f: f(x, session, args), functions, session.units)


def main():

    from ._schemas import InputParameters, OutputParameters

    mod = ArgSchemaParser(schema_type=InputParameters,
                          output_schema_type=OutputParameters)

    output = calculate_stimulus_metrics(mod.args)

    output.update({"input_parameters": mod.args})

    if "output_json" in mod.args:
        mod.output(output, indent=2)
    else:
        print(mod.get_output_json(output))


if __name__ == "__main__":

    main()

