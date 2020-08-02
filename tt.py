import pandas as pd
import matplotlib.pyplot as plt


a = pd.read_csv('./submission/ensemble_submission.csv')
b = pd.read_csv('./submission/ensembled.csv')


df = pd.merge(a, b, on='image_name')
print(df.head())

plt.scatter(df['target_x'], df['target_y'])
plt.show()
