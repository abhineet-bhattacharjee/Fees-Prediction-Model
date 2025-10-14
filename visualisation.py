import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


df = pd.read_csv('Harvard Fees Dataset.csv')

sns.histplot(df['cost'], bins=20)
plt.title('Distribution of Tuition Fees')
plt.xlabel('Tuition Fees')
plt.ylabel('Frequency')
plt.show()
