
from argschema import ArgSchemaParser
import time
import os

import pandas as pd
import numpy as np

import logging

from functools import reduce

from ..ecephys_session import EcephysSession

from .drifting_gratings import DriftingGratings
from .static_gratings import StaticGratings
from .natural_scenes import NaturalScenes
from .dot_motion import DotMotion
from .flashes import Flashes
from .receptive_field_mapping import ReceptiveFieldMapping

logger = logging.getLogger(__name__)


def calculate_stimulus_metrics(args):

    print('ecephys: stimulus metrics module')

    start = time.time()

    stimulus_classes = (
                 DriftingGratings,
                 StaticGratings,
                 NaturalScenes,
                 DotMotion,
                 Flashes,
                 ReceptiveFieldMapping,
                )

    metrics_df = calculate_metrics(args['nwb_path'], stimulus_classes, args)

    metrics_df.to_csv(args['output_file'])

    execution_time = time.time() - start

    print('total time: ' + str(np.around(execution_time, 2)) + ' seconds\n')

    return {"execution_time" : execution_time}


def calculate_metrics(nwb_path, stimulus_classes, args):

    """
    Adds columns to units table for one session, based on the metrics
    for each stimulus type.

    Parameters:
    -----------
    nwb_path : String
        Path to a spikes NWB file
    stimulus_classes : tuple
        Classes that add new columns to a units table
    args : module parameters

    Returns:
    --------
    units_df : pandas.DataFrame
        Units table with new columns appended

    """

    try:
        session = EcephysSession.from_nwb_path(nwb_path)
    except FileNotFoundError:
        return pd.DataFrame()

    metrics_list = []

    for stim in stimulus_classes:
        try:
            metrics_list.append(stim(session, params=args).metrics)
        except Exception:
            logger.info('Could not find data for ' + str(stim.name) + ' stimulus.')

    print(metrics_list)

    metrics_df = reduce(lambda left,right: pd.merge(left, right, on='unit_id'), metrics_list)

    return metrics_df


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

