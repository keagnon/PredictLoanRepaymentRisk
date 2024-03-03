import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from exploration import DataExploration

df = pd.read_csv('data/credit_customers_upload.csv')


if __name__ == "__main__":

    print("--------------------------------------------")
    DataExploration.display_columns_by_type(df)
    print("--------------------------------------------")
    DataExploration.display_unique_values(df)
    print("----------------------------------------------------------------")
    DataExploration.display_missing_values(df)
    print("----------------------------------------------------------------")
    DataExploration.explore_data(df)

