import numpy as np

from allensdk.brain_observatory.ecephys.ecephys_session import removed_unused_stimulus_presentation_columns

def compute_metrics(units_df, session, args):

    # figure out which presentation applies
    # 1. extract presentationwise spike counts
    # 2. extract conditionwise mean spike counts
    # 3. extract running vs non-running for each trial
    # 4. extract spontaneous distribution

    # TODO:
    # - time to peak
    # - time to onset
    # - significant responsiveness (relative to blank and spontaneous)
    # - running modulation
    # - lifetime sparseness
    # - reliability

    presentation_ids = session.stimulus_presentations.loc[
            (session.stimulus_presentations['stimulus_name'] == args['flashes']['stimulus_key'])
            ].index.values

    presentationwise_response_matrix = session.presentationwise_spike_counts(
        bin_edges = bin_edges,
        stimulus_presentation_ids = presentation_ids,
        unit_ids = units_df.index.values,
        )

    conditionwise_response_table = \
        removed_unused_stimulus_presentation_columns( \
            session.conditionwise_mean_spike_counts( \
        unit_ids = units_df.index.values, \
        stimulus_presentation_ids = presentation_ids \
        )) # should include number of trials for each condition!

    return units_df.assign(fl_on_off_ratio = lambda df: compute_on_off_ratio(df.unit_id, conditionwise_response_table) \
                           )


def compute_on_off_ratio(unit_id, df):

    """
    Compute ON/OFF for one unit, based on mean spike count.

    Parameters:
    -----------
    unit_id : Int
        Unique index for unit
    df : pandas.DataFrame
        Table of conditionwise mean spike counts.

    Returns:
    --------
    on_off_ratio : float
        ON/OFF ratio

    """

    response_for_unit = df[df.unit_id == unit_id].groupby('Color').mean()

    on_off_ratio = response_for_unit['mean_spike_count'].loc['black'] / \
                   response_for_unit['mean_spike_count'].loc['white']

    return on_off_ratio


