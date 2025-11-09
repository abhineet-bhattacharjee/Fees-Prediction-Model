import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


df = pd.read_csv('dataset.csv')
tuition_cols = [c for c in df.columns if c not in ['academic.year', 'inflation_rate', 'endowment_billions']]
df_long = df.melt(id_vars='academic.year',
                  value_vars=tuition_cols,
                  var_name='school',
                  value_name='cost')


def cost_vs_frequency():
    sns.histplot(df_long['cost'], bins=20)
    plt.title('Cost vs Frequency Graph')
    plt.xlabel('Tuition Fees')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def year_vs_cost():
    sns.scatterplot(data=df_long, x='academic.year', y='cost', hue='school')
    plt.title('Year vs Cost Graph')
    plt.xlabel('Academic Year')
    plt.ylabel('Tuition Fees')
    plt.tight_layout()
    plt.show()

def school_vs_cost():
    sns.boxplot(data=df_long, x='school', y='cost')
    plt.title('School vs Cost Graph')
    plt.xlabel('School')
    plt.ylabel('Tuition Fees')
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    cost_vs_frequency()
    year_vs_cost()
    school_vs_cost()