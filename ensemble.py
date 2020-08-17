import os, glob
import numpy as np
import pandas as pd
import itertools


def ensemble(subs):
    res = None
    for i, path in enumerate(subs):
        if i == 0:
            res = pd.read_csv(path)
        else:
            t = pd.read_csv(path)
            res['target'] += t['target']

    res['target'] = res['target'] / len(subs)

    return res


def ensemble_weight(subs, w):
    res = None
    for i, (path, weight) in enumerate(zip(subs, w)):
        if i == 0:
            res = pd.read_csv(path)
            res['target'] = res['target'] * weight
        else:
            t = pd.read_csv(path)
            res['target'] += t['target'] * weight

    return res


def ensemble_minmax(subs):
    res = None
    for i, path in enumerate(subs):
        if i == 0:
            res = pd.read_csv(path)
        else:
            t = pd.read_csv(path)
            res['target'] += t['target']

    # min-max norm
    res['target'] -= res['target'].min()
    res['target'] /= res['target'].max()

    return res


def ensemble_geomean(subs):
    res = None
    for i, path in enumerate(subs):
        if i == 0:
            res = pd.read_csv(path)
        else:
            t = pd.read_csv(path)
            res['target'] *= t['target']

    res['target'] = np.power(res['target'], 1/len(subs))

    return res


def ensemble_rank(subs):
    dfs = []
    for i, path in enumerate(subs):
        res = pd.read_csv(path)
        res = res.rename(columns={'target': f'target_{i}', 'image_name': f'image_name_{i}'})
        dfs.append(res)

    res = pd.concat(dfs, axis=1)

    for c in [c for c in res.columns if 'target' in c]:
        res[c + '_rank'] = res[c].rank()

    res['rank_sum'] = np.sum(res[col] for col in res.columns if '_rank' in col)
    res['target'] = res['rank_sum'] / (len(subs) * res.shape[0])

    res = res[['image_name_0', 'target']]
    res = res.rename(columns={'image_name_0': 'image_name'})

    return res


def ensemble_postprep(subs, best_path):
    dfs = []
    for i, path in enumerate(subs):
        res = pd.read_csv(path)
        res = res.rename(columns={'target': f'target_{i}', 'image_name': f'image_name_{i}'})
        dfs.append(res)

    best_res = pd.read_csv(best_path)
    best_res = best_res.rename(columns={'target': 'sub_best'})
    dfs.append(best_res)

    res = pd.concat(dfs, axis=1)

    tar_cols = [c for c in res.columns if 'target' in c]

    i = 0
    for pair in itertools.combinations(tar_cols, 2):
        res[f'diff_{i}'] = res[pair[0]] - res[pair[1]]
        i += 1

    col_diff = [c for c in res.columns if 'diff' in c]
    res['diff_avg'] = res[col_diff].mean(axis=1)
    WEIGHT = 1

    res["sub_new"] = res.apply(lambda x: (1 + WEIGHT * x["diff_avg"]) * x["sub_best"] if x["diff_avg"] < 0 else (1 - WEIGHT * x["diff_avg"]) * x["sub_best"] + WEIGHT * x["diff_avg"], axis=1)

    res = res[['image_name_0', 'sub_new']]
    res = res.rename(columns={'image_name_0': 'image_name', 'sub_new': 'target'})

    return res


if __name__ == '__main__':

    # Weight Stacking
    subs = [
        './submission/ensemble_submission_weights_enet_b2_384_1.csv',
        './submission/ensemble_submission_weights_enet_b2_384_2.csv',
        './submission/ensemble_submission_weights_enet_b2_256.csv',
        './submission/ensemble_submission_weights_enet_b4_192.csv',
        './submission/ensemble_submission_weights_enet_b0_192.csv',
        # './submission/submission_0.9648.csv',
        # './submission/ensembled_Analysis of Melanoma Metadata and EffNet Ensemble_0.9513.csv',
    ]

    weights = [0.3, 0.3, 0.2, 0.1, 0.1]

    res = ensemble_weight(subs, weights)
    res.to_csv('./submission/ensemble_submission_weights_mean.csv', index=False)

    # minmax norm
    subs = [
        './submission/ensemble_submission_weights_enet_b2_384_1.csv',
        './submission/ensemble_submission_weights_enet_b2_384_2.csv',
        './submission/ensemble_submission_weights_enet_b2_256.csv',
        './submission/ensemble_submission_weights_enet_b4_192.csv',
        './submission/ensemble_submission_weights_enet_b0_192.csv',
        # './submission/ensembled_Analysis of Melanoma Metadata and EffNet Ensemble_0.9513.csv',
        # './submission/submission_minmax_highest_public_0.9619.csv',
    ]

    res = ensemble_minmax(subs)
    res.to_csv('./submission/ensemble_submission_minmax.csv', index=False)

    # Geometric Mean
    subs = [
        './submission/ensemble_submission_weights_enet_b2_384_1.csv',
        './submission/ensemble_submission_weights_enet_b2_384_2.csv',
        './submission/ensemble_submission_weights_enet_b2_256.csv',
        './submission/ensemble_submission_weights_enet_b4_192.csv',
        './submission/ensemble_submission_weights_enet_b0_192.csv',
        # './submission/ensembled_Analysis of Melanoma Metadata and EffNet Ensemble_0.9513.csv',
        # './submission/submission_minmax_highest_public_0.9619.csv',
    ]

    res = ensemble_geomean(subs)
    res.to_csv('./submission/ensemble_submission_geomean.csv', index=False)

    # RankAve
    subs = [
        './submission/ensemble_submission_weights_enet_b2_384_1.csv',
        './submission/ensemble_submission_weights_enet_b2_384_2.csv',
        './submission/ensemble_submission_weights_enet_b2_256.csv',
        './submission/ensemble_submission_weights_enet_b4_192.csv',
        './submission/ensemble_submission_weights_enet_b0_192.csv',
        # './submission/ensembled_Analysis of Melanoma Metadata and EffNet Ensemble_0.9513.csv',
        # './submission/submission_minmax_highest_public_0.9619.csv',
    ]

    res = ensemble_rank(subs)
    res.to_csv('./submission/ensemble_submission_rank.csv', index=False)

    # Mean Ensemble
    subs = [
        './submission/ensemble_submission_weights_mean.csv',
        './submission/ensemble_submission_minmax.csv',
        './submission/ensemble_submission_geomean.csv',
        './submission/ensemble_submission_rank.csv',
    ]

    res = ensemble(subs)

    res.to_csv('./submission/total_res.csv', index=False)

    # Kernel Ensemble
    # https://www.kaggle.com/mekhdigakhramanian/top-7-lb-0-9648-post-processing
    # subs = [
    #     './submission/ensemble_submission_weights_enet_b2_384.csv',
    #     './submission/ensemble_submission_weights_enet_b2_256.csv',
    #     './submission/ensemble_submission_weights_enet_b4_192.csv',
    #     './submission/ensemble_submission_weights_enet_b0_192.csv',
    # ]
    # subs = glob.glob('./submission/submission_all_enet_*.csv')
    # best_path = './submission/submission_0.9648.csv'
    #
    # res = ensemble_postprep(subs, best_path)
    #
    # res.to_csv('./submission/ensemble_submission_postprep.csv', index=False)
