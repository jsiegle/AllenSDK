import numpy as np

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

    rf_mapping_presentations = session.stimulus_presentations.loc[
            (session.stimulus_presentations['stimulus_name'] == args['receptive_field_mapping']['stimulus_key'])
            ]

    bin_edges = np.linspace(0, 0.250, 251)

    rf_mapping_presentations.loc[:, 'Pos_y'] = 8 - rf_mapping_presentations['Pos_y']

    presentationwise_response_matrix = session.presentationwise_spike_counts(
        bin_edges = bin_edges,
        stimulus_presentation_ids = rf_mapping_presentations.index.values,
        unit_ids = units_df.index.values,
        )

    rf_matrix = mean_response_by_position(presentationwise_response_matrix, rf_mapping_presentations)

    return units_df.assign(receptive_field = lambda df: get_receptive_field(df.unit_id, rf_matrix['mean_spike_count']),
                           receptive_field_gaussian_fit_params = lambda df : compute_receptive_field_gaussian_fit(df.receptive_field)
                           )


def mean_response_by_position(
        dataset, presentations,
        row_key='Pos_x', column_key='Pos_x',
        unit_key='unit_id', time_key='time_relative_to_stimulus_onset',
        mean_count_key='mean_spike_count'):

    dataset = dataset.copy()
    dataset['spike_counts'] = dataset['spike_counts'].sum(dim=time_key)
    dataset = dataset.drop(time_key)

    dataset[row_key] = presentations.loc[:, row_key]
    dataset[column_key] = presentations.loc[:, column_key]
    dataset = dataset.to_dataframe()

    dataset = dataset.reset_index(unit_key).groupby([row_key, column_key, unit_key]).mean()

    return dataset.rename(columns={'spike_counts': mean_count_key}).to_xarray()


def get_receptive_field(unit_id, rf_matrix):

    """
    Returns a receptive field for a particular unit from an xarray

    Parameters:
    -----------
    unit_id : int
        Unique identifier for one unit
    rf_matrix: xarray.DataArray
        3D matrix with axes of Pos_x, Pos_x, and unit_id

    Returns:
    --------
    receptive_field : numpy.ndarray
        2D matrix with axes of Pos_x and Pos_y

    """

    receptive_field = rf_matrix.loc[{'unit_id': unit_id}].values

    return receptive_field


def compute_receptive_field_gaussian_fit(rf):

    params, success = fitgaussian(rf)



    return gaussian_fit


def compute_receptive_field_center(rf):



    return receptive_field_center


