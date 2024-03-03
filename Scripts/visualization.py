import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class VisualizationData:
    """
    Visualization to explore data in depth.
    """

    def __init__(self):
        pass

    def numeric_histograms(data, columns=None):
        """
        Create histograms to explore numeric variables.
        -------------------------------------------------------------
        params:
            data: DataFrame containing the data.
            columns: List of numeric columns to visualize (default is None).
        """
        if columns is None:
            columns = ['Duree_credit', 'Montant_credit', 'Age']
        plt.figure(figsize=(15, 5))
        for i, column in enumerate(columns, 1):
            plt.subplot(1, len(columns), i)
            sns.histplot(data[column], kde=True)
            plt.title(f'Distribution of {column}')
        plt.tight_layout()
        plt.show()

    def scatter_plots(data, columns=None):
        """
        Create scatter plots to explore relationships between variables.
        -------------------------------------------------------------
        params:
            data: DataFrame containing the data.
            columns: List of tuples specifying pairs of variables to plot (default is None).
        """
        if columns is None:
            columns = [('Duree_credit', 'Montant_credit'), ('Age', 'Taux_effort')]
        plt.figure(figsize=(12, 5))
        for i, (x_var, y_var) in enumerate(columns, 1):
            plt.subplot(1, len(columns), i)
            sns.scatterplot(x=x_var, y=y_var, data=data)
            plt.title(f'Relation between {x_var} and {y_var}')
        plt.tight_layout()
        plt.show()

    def categorical_bar_charts(data, columns=None):
        """
        Create bar charts to explore categorical variables.
        -------------------------------------------------------------
        params:
            data: DataFrame containing the data.
            columns: List of categorical columns to visualize (default is None).
        """
        if columns is None:
            columns = ['Historique_credit', 'Objet_credit', 'Epargne']
        plt.figure(figsize=(15, 8))
        for i, column in enumerate(columns, 1):
            plt.subplot(1, len(columns), i)
            sns.countplot(x=column, data=data, hue='Cible')
            plt.title(f'Distribution of {column}')
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def boxplot(data, columns=None):
        """
        Create boxplots to explore credit data.
        -------------------------------------------------------------
        params:
            data: DataFrame containing the credit data.
            columns: List of numeric columns to visualize (default is None).
        """
        if columns is None:
            columns = ['Duree_credit', 'Montant_credit', 'Taux_effort', 'Anciennete_domicile', 'Age', 'Nb_credits', 'Nb_pers_charge', 'Cible']
        plt.figure(figsize=(10, 6))
        for column in columns:
            sns.boxplot(x='Cible', y=column, data=data)
            plt.title(f'Boxplot of {column} by Target')
            plt.show()

    def violin_plots(data, columns=None):
        """
        Create violin plots to explore the distribution of numeric variables.
        -------------------------------------------------------------
        params:
            data: DataFrame containing the data.
            columns: List of numeric columns to visualize (default is None).
        """
        if columns is None:
            columns = ['Duree_credit', 'Montant_credit', 'Age']
        plt.figure(figsize=(15, 5))
        for i, column in enumerate(columns, 1):
            plt.subplot(1, len(columns), i)
            sns.violinplot(x='Cible', y=column, data=data)
            plt.title(f'Distribution of {column}')
        plt.tight_layout()
        plt.show()

    def pairplot(data, columns=None):
        """
        Create pairplot to explore pairwise relationships between numeric variables.
        -------------------------------------------------------------
        params:
            data: DataFrame containing the data.
            columns: List of numeric columns to visualize (default is None).
        """
        if columns is None:
            columns = ['Duree_credit', 'Montant_credit', 'Age', 'Taux_effort']
        sns.pairplot(data[columns])
        plt.title("Pairplot of Numeric Variables")
        plt.show()

    def heatmap_correlation(data, columns=None):
        """
        Create a heatmap to visualize the correlation matrix of numeric variables.
        -------------------------------------------------------------
        params:
            data: DataFrame containing the data.
            columns: List of numeric columns to include in the correlation matrix (default is None).
        """
        if columns is None:
            columns = ['Duree_credit', 'Montant_credit', 'Taux_effort', 'Anciennete_domicile', 'Age', 'Nb_credits', 'Nb_pers_charge', 'Cible']
        plt.figure(figsize=(10, 8))
        sns.heatmap(data[columns].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title("Correlation Matrix of Numeric Variables")
        plt.show()
