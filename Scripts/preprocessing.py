import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class preprocessing_data:
    """
        Do a preprocessing of any data
    """
    def __init__(self):
        pass

    def afficher_colonnes_par_type(data):
        """
        Display all columns and the 5 first lines of the dataset
        Display couluns and theirs types
        Args:
            data (DataFrame): DataFrame.
        """
        print(data.head())
        print(data.columns )

        for dtype in data.dtypes.unique():
            print(f"Colonnes de type {dtype}:")
            print(list(data.select_dtypes(include=[dtype]).columns))
            print()

