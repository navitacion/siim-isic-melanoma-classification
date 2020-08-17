import random
import os
import numpy as np
import pandas as pd
import category_encoders as ce
import torch


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def preprocessing_meta(train, test):
    train = train[['image_name', 'patient_id', 'sex', 'age_approx', 'anatom_site_general_challenge', 'target', 'fold']]
    test = test[['image_name', 'patient_id', 'sex', 'age_approx', 'anatom_site_general_challenge']]
    test.loc[:, 'target'] = 0
    test.loc[:, 'fold'] = 0

    # Preprocessing
    train['age_approx'] /= train['age_approx'].max()
    test['age_approx'] /= test['age_approx'].max()
    train['age_approx'].fillna(0, inplace=True)
    test['age_approx'].fillna(0, inplace=True)
    for c in ['sex', 'anatom_site_general_challenge']:
        train[c].fillna('Nodata', inplace=True)
        test[c].fillna('Nodata', inplace=True)
    encoder = ce.OneHotEncoder(cols=['sex', 'anatom_site_general_challenge'], handle_unknown='impute')
    train = encoder.fit_transform(train)
    test = encoder.transform(test)

    test.drop(['target', 'fold'], axis=1, inplace=True)

    return train, test


def summarize_submit(sub_list, experiment, filename='submission.csv'):
    res = pd.DataFrame()
    for i, path in enumerate(sub_list):
        sub = pd.read_csv(path)

        if i == 0:
            res['image_name'] = sub['image_name']
            res['target'] = sub['target']
        else:
            res['target'] += sub['target']
        os.remove(path)

    # min-max norm
    res['target'] -= res['target'].min()
    res['target'] /= res['target'].max()
    res.to_csv(filename, index=False)
    experiment.log_asset(file_data=filename, file_name=filename)
    os.remove(filename)

    return res
