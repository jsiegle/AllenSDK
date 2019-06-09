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
            (session.stimulus_presentations['stimulus_name'] == args['natural_scenes']['stimulus_key'])
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

    return units_df.assign(ns_image_selectivity = lambda df: compute_image_selectivity(df.unit_id, conditionwise_response_dataframe)
                            )


def compute_image_selectivity(unit_id, df):

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

    fmin = self.response_events[1:,nc,0].min()
    fmax = self.response_events[1:,nc,0].max()
    rtj = np.empty((1000,1))
    for j in range(1000):
        thresh = fmin + j*((fmax-fmin)/1000.)
        theta = np.empty((118,1))
        for im in range(118):
            if self.response_events[im+1,nc,0] > thresh:  #im+1 to only look at images, not blanksweep
                theta[im] = 1
            else:
                theta[im] = 0
        rtj[j] = theta.mean()
    biga = rtj.mean()

    image_selectivity = 1 - (2*biga)

    return image_selectivity



