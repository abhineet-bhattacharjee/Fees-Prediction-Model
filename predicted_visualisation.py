import matplotlib.pyplot as plt
import pandas as pd

df_actual = pd.read_csv('dataset.csv')
df_pred = pd.read_csv('predicted.csv')

school_cols = [c for c in df_actual.columns if c not in ['academic.year', 'inflation_rate', 'endowment_billions']]

def plot_actual_vs_predicted_lines():
    any_school = False
    for school in school_cols:
        plt.figure()
        plt.plot(df_actual['academic.year'], df_actual[school], color='blue', label='Actual', linewidth=2)

        if school in df_pred.columns:
            plt.plot(df_actual['academic.year'], df_pred[school], color='red', label='Predicted', linewidth=2)
            any_school = True
        else:
            print(f"No predicted values found for school: {school}")

        plt.title(f'Actual vs Predicted Tuition Fees: {school}')
        plt.xlabel('Academic Year')
        plt.ylabel('Tuition Fees')
        plt.legend()
        plt.tight_layout()
        plt.show()
    if not any_school:
        print("No predicted values present for any school.")


if __name__ == '__main__':
    plot_actual_vs_predicted_lines()
