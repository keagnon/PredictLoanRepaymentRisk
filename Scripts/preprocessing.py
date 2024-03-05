import pandas as pd
from sklearn.ensemble import IsolationForest
from visualization import VisualizationData
from exploration import DataExploration
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class PreprocessingData:
    """
    Class for preprocessing any data.
    """

    def __init__(self):
        pass

    def remove_outliers(self, dataframe, column):
        """
        Remove outliers from a specific column of a DataFrame using Isolation Forest.
        -------------------------------------------------------------
        params:
            dataframe: Pandas DataFrame containing the data.
            column (str): The name of the column to process.

        Returns:
            DataFrame: DataFrame with outliers removed.
        """
        isolation_forest = IsolationForest(contamination='auto', random_state=42)
        isolation_forest.fit(dataframe[[column]])
        anomalies = isolation_forest.predict(dataframe[[column]])
        dataframe_filtered = dataframe[anomalies == 1]

        return dataframe_filtered


    def preprocess_special_values(df):
        """
        Replace special values with NaN in the DataFrame, then replace NaN values
        with the most frequent value in each specified column.
        -------------------------------------------------------------
        params:
            df: Pandas DataFrame containing the data.

        Returns:
            DataFrame: DataFrame with special values replaced and NaN values replaced by the most frequent values.
        """
        # Replace special values with NaN
        df_processed = df.replace('-1', pd.NA)

        # Replace NaN values with the most frequent value in each specified column
        special_columns = ['Objet_credit', 'Statut_domicile', 'Telephone']
        for col in special_columns:
            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)

        return df_processed


    def encode_onehot(df):
        """
        Apply one-hot encoding to categorical columns of a DataFrame.

        Args:
            df (DataFrame): Pandas DataFrame containing the data.

        Returns:
            DataFrame: DataFrame with categorical columns encoded using one-hot encoding.
        """
        # Select only the categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns

        # Apply one-hot encoding to the categorical columns
        df_encoded = pd.get_dummies(df, columns=categorical_columns)

        return df_encoded



    def convert_bool_to_float(df):
        """
        Convert boolean columns of a DataFrame to integers.
        -------------------------------------------------------------
        params:
            df: Pandas DataFrame containing the data

        Returns:
            DataFrame: DataFrame with boolean columns converted to integers
        """
        for column in df.columns:
            if df[column].dtype == bool:
                df[column] = df[column].astype(float)

        return df


    def scale_data(df):
        """
        Normalize the numeric data of a DataFrame
        -------------------------------------------------------------
        params:
            df: Pandas DataFrame containing the data

        Returns:
            DataFrame: DataFrame with normalized numeric data
        """
        scaler = StandardScaler()
        numeric_columns = df.select_dtypes(include=['int']).columns
        df_scaled = df.copy()
        df_scaled[numeric_columns] = scaler.fit_transform(df_scaled[numeric_columns])

        return df_scaled
