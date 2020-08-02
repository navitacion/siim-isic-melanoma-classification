import os, glob
import pandas as pd


subs = glob.glob('./submission/*.csv')

res = pd.DataFrame()

for i, path in enumerate(subs):

    if i == 0:
        res = pd.read_csv(path)
        res['target'] = res['target'] / len(subs)
    else:
        t = pd.read_csv(path)
        res['target'] += t['target'] / len(subs)

res.to_csv('./submission/ensemble_submission.csv', index=False)
