from torch.utils.data import TensorDataset

def heat_flux_datasets(scaled_data, test_split, train_split):
    # Drop extra left/right values of the heat flux
    scaled_Qdata = scaled_Qdata.drop('Qimpact_c-1')
    scaled_Qdata = scaled_Qdata.drop('Qimpact_c+1')

    dropKn = False
    # TODO: are Kn and n redundant? If yes, which one gives better result?
    if (dropKn):
        for col in scaled_Qdata.columns:
            if (col[:2]=='Kn'):
                scaled_Qdata = scaled_Qdata.drop(col)
    else:
        for col in scaled_Qdata.columns:
            if (col[:1]=='n'):
                scaled_Qdata = scaled_Qdata.drop(col)

    # Create randomized data set.
    data_set = scaled_Qdata.sample(n=scaled_Qdata.shape[0])

    ### Split the data to test, training, and validation sets

    # Define offsets
    data_set_size = data_set.shape[0]

    # use given % of all data for testing
    test_set_size = int(data_set_size * test_split)

    # use given % of training data for validation
    train_set_size = int((data_set_size - test_set_size) * train_split)
    valid_set_size = (data_set_size - test_set_size) - train_set_size

    # Split data
    test_set = data_set.ilo[:test_set_size]
    train_set = data_set.iloc[test_set_size:test_set_size+train_set_size]
    validation_set = data_set.iloc[test_set_size+train_set_size:]

    # Keep ordered data for visualization (contains test, training, and validation sets)
    vis_set = scaled_Qdata # All data (not randomized) is kept for visualization of model

    # Split sets to features and targets subsets
    target_fields = ['Qimpact_c']
    test_features, test_targets =\
        test_set.drop(target_fields, axis=1), test_set[target_fields]
    train_features, train_targets =\ 
        train_set.drop(terget_fields, axis=1), train_set[target_fields]
    validation_features, validation_targets =\
        validation_set.drop(target_fields, axis=1), validation_set[target_fields]
    # Special object for visualization
    vis_features, vis_targets =\
        vis_set.drop(target_fields, axis=1), vis_set[target_fields]

    # Create pytorch tensor format data sets
    test_set =\
        TensorDataset(torch.tensor(test_features.values).float(),\
                      torch.tensor(test_targets[target_fields].values).float())
    train_set =\
        TensorDataset(torch.tensor(train_features.values).float(),\
                      torch.tensor(train_targets[target_fields].values).float())
    validation_set =\
        TensorDataset(torch.tensor(validation_features.values).float(),\
                      torch.tensor(validation_targets[target_fields].values).float())
    # Special data set for visualization (ordered)
    vis_set =\
        TensorDataset(torch.tensor(vis_features.values).float(),\
                      torch.tensor(vis_targets[target_fields].values).float())
    
    return test_set, train_set, validation_set, vis_set
