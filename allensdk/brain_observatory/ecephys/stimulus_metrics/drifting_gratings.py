import numpy as np

from allensdk.brain_observatory.ecephys.ecephys_session import removed_unused_stimulus_presentation_columns

def compute_metrics(units_df, session, args):

    # figure out which presentation applies
    # 1. extract presentationwise spike counts
    # 2. extract conditionwise mean spike counts
    # 3. extract running vs non-running for each trial
    # 4. extract spontaneous distribution [pre-stim baseline]

    # TODO:
    # - time to peak
    # - time to onset
    # - significant responsiveness (relative to blank and spontaneous)
    # - running modulation
    # - lifetime sparseness
    # - reliability

    presentation_ids = session.stimulus_presentations.loc[
            (session.stimulus_presentations['stimulus_name'] == args['drifting_gratings']['stimulus_key'])
            ].index.values

    presentationwise_response_matrix = session.presentationwise_spike_counts(
        bin_edges = bin_edges,
        stimulus_presentation_ids = presentation_ids,
        unit_ids = units_df.index.values,
        )

    conditionwise_response_dataframe = \
        removed_unused_stimulus_presentation_columns( \
            session.conditionwise_mean_spike_counts( \
        unit_ids = units_df.index.values, \
        stimulus_presentation_ids = presentation_ids \
        )) # should include number of trials for each condition!

    return units_df.assign(dg_pref_ori = lambda df: compute_dg_pref_ori(df.unit_id, conditionwise_response_dataframe),
                           dg_pref_tf = lambda df: compute_dg_pref_tf(df.unit_id, conditionwise_response_dataframe),
                           dg_osi = lambda df: compute_dg_osi(df.unit_id, conditionwise_response_dataframe, df.dg_pref_tf),
                           dg_dsi = lambda df: compute_dg_dsi(df.unit_id, conditionwise_response_dataframe, df.dg_pref_tf),
                           dg_tfdi = lambda df: compute_df_tfdi(df.unit_id, conditionwise_response_dataframe, df.dg_pref_ori)
                           )



def compute_dg_pref_ori(unit_id, df):

    """
    Compute preferred orientation for one unit, based on mean spike count.

    Parameters:
    -----------
    unit_id : Int
        Unique index for unit
    df : pandas.DataFrame
        Table of conditionwise mean spike counts.

    Returns:
    --------
    pref_ori : float
        Preferred orientation

    """

    response_for_unit = df[df.unit_id == unit_id].groupby('Ori').mean()

    pref_ori = response_for_unit['mean_spike_count'].idxmax()

    return pref_ori



def compute_dg_pref_tf(unit_id, df):

    """
    Compute preferred temporal frequency for one unit, based on mean spike count.

    Parameters:
    -----------
    unit_id : Int
        Unique index for unit
    df : pandas.DataFrame
        Table of conditionwise mean spike counts.

    Returns:
    --------
    pref_tf : float
        Preferred temporal frequency

    """

    response_for_unit = df[df.unit_id == unit_id].groupby('TF').mean()

    pref_tf = response_for_unit['mean_spike_count'].idxmax()

    return pref_tf


def compute_dg_osi(unit_id, df, pref_tf):

    """
    Compute orientation selectivity index for one unit, based on mean spike count.

    Parameters:
    -----------
    unit_id : Int
        Unique index for unit
    df : pandas.DataFrame
        Table of conditionwise mean spike counts
    pref_tf : Float
        Preferred temporal frequency

    Returns:
    --------
    osi : float
        Orientation selectivity index

    """

    orivals = df.Ori.unique()[:-1]
    orivals_rad = np.deg2rad(orivals)

    tuning = df[(df.unit_id == unit_id) * (df.TF == pref_tf)].groupby('Ori').mean()

    CV_top_os = tuning['mean_spike_count'].values[:-1]*np.exp(1j*2*orivals_rad)

    osi = np.abs(CV_top_os.sum())/tuning.sum()

    return osi



def compute_dg_dsi(unit_id, df, pref_tf):

    """
    Compute direction selectivity index for one unit, based on mean spike count.

    Parameters:
    -----------
    unit_id : Int
        Unique index for unit
    df : pandas.DataFrame
        Table of conditionwise mean spike counts
    pref_tf : Float
        Preferred temporal frequency

    Returns:
    --------
    dsi : float
        Direction selectivity index

    """

    orivals = df.Ori.unique()[:-1]
    orivals_rad = np.deg2rad(orivals)

    tuning = df[(df.unit_id == unit_id) * (df.TF == pref_tf)].groupby('Ori').mean()

    CV_top_ds = tuning['mean_spike_count'].values[:-1]*np.exp(1j*orivals_rad)

    dsi = np.abs(CV_top_ds.sum())/tuning.sum()

    return dsi



def compute_df_tfdi(unit_id, df, pref_ori):

    """
    Compute temporal frequency discrimination index for one unit, based on mean spike count.

    Parameters:
    -----------
    unit_id : Int
        Unique index for unit
    df : pandas.DataFrame
        Table of conditionwise mean spike counts
    pref_ori : Float
        Preferred orientation

    Returns:
    --------
    tfdi : float
        Temporal frequency discrimination index

    """

    ### NEED TO FIX ###
    tf_tuning = df[(df.unit_id == unit_id) * (df.Ori == pref_ori)].groupby('TF').mean()[:-1]

    trials = self.mean_sweep_events[(self.stim_table.orientation==self.orivals[pref_ori])][v].values
    SSE_part = np.sqrt(np.sum((trials-trials.mean())**2)/(len(trials)-5))

    tfdi = (np.ptp(tf_tuning))/(np.ptp(tf_tuning) + 2*SSE_part)

    return tfdi

