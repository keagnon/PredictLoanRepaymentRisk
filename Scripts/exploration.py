import pandas as pd

class DataExploration:
    """
    Class to perform exploration of any dataset.
    """

    def __init__(self):
        pass

    def display_columns_by_type(data):
        """
        Display all columns and the first 5 lines of the dataset
        Display columns and their types
        -------------------------------------------------------------
        params:
            data (DataFrame): DataFrame.
        """
        print(data.head())

        # Nombre de lignes
        nombre_lignes = len(data)

        # Nombre de colonnes
        nombre_colonnes = len(data.columns)

        print("Nombre de lignes :", nombre_lignes)
        print("Nombre de colonnes :", nombre_colonnes)
        print("--------------------------------------------------")

        for dtype in data.dtypes.unique():
            print(f"Columns of type {dtype}:")
            print(list(data.select_dtypes(include=[dtype]).columns))
            print()

    def explore_data(data, columns_to_drop=None):
        """
        Perform data exploration.
        -------------------------------------------------------------
        params:
            data: DataFrame containing the data.
            columns_to_drop: List of columns to drop from the DataFrame (default is None).
        """
        print("Data information:")
        print(data.info())
        print("\nDescriptive statistics:")
        print(data.describe())
        if columns_to_drop:
            data.drop(columns_to_drop, axis=1)

    def display_unique_values(data):
        """
        Display unique values of each column in a DataFrame.
        -------------------------------------------------------------
        params:
            data : DataFrame containing the data.
        """
        for column in data.columns:
            print("Column:", column)
            print(data[column].unique())
            print()

    def display_missing_values(data):
        """
        Display the number of missing values per column in the DataFrame.
        -------------------------------------------------------------
        params:
            data: DataFrame containing the data.
        """
        missing_values = data.isnull().sum()
        if missing_values.sum() == 0:
            print("No missing values found.")
        else:
            print("Missing values per column:")
            print(missing_values[missing_values > 0])

    def data_correlation(data):
        """
        Calculate and display the correlation matrix of the DataFrame.
        Only numeric columns are considered for correlation calculation.
        -------------------------------------------------------------
        params:
            data : DataFrame containing the data.
        """

        numeric_data = data.select_dtypes(include=['float64', 'int64'])

        if not numeric_data.empty:
            correlation_matrix = numeric_data.corr()
            print("Correlation Matrix:")
            print(correlation_matrix)
        else:
            print("No numeric columns found in the DataFrame for correlation calculation.")


