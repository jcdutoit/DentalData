import data_utils
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

NUM_CLUSTERS = 3
DATA_PATH = "data/Interview_data.csv"

raw = pd.read_csv(DATA_PATH)
df = data_utils.clean_data(raw)
train, test = data_utils.split_data(df)

kmeans = KMeans(NUM_CLUSTERS)
kmeans.fit(train['PayAmt'].to_numpy())

preds = []
for i in test:
    preds.append(kmeans.predict(i))
test['Preds'] = preds

c1 = test.loc(test['Preds'] == 0)
c2 = test.loc(test['Preds'] == 1)
c3 = test.loc(test['Preds'] == 2)

plt.scatter(c1['ElapsedDays'], c1['PayAmt'], c='blue')
plt.scatter(c2['ElapsedDays'], c2['PayAmt'], c='green')
plt.scatter(c3['ElapsedDays'], c3['PayAmt'], c='red')

plt.show()
