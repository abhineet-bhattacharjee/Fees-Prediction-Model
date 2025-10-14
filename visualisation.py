import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


df = pd.read_csv('Harvard Fees Dataset.csv')

sns.histplot(df['cost'], bins=20)
plt.title('Cost vs Frequency Graph')
plt.xlabel('Tuition Fees')
plt.ylabel('Frequency')
plt.show()

sns.scatterplot(data=df, x='academic.year', y='cost', hue='school')
plt.title('Year vs Cost Graph')
plt.xlabel('Academic Year')
plt.ylabel('Tuition Fees')
plt.show()

sns.boxplot(data=df, x='school', y='cost')
plt.title('School vs Cost Graph')
plt.xlabel('School')
plt.ylabel('Tuition Fees')
plt.xticks(rotation=45)
plt.show()