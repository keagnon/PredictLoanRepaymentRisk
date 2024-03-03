import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from exploration import DataExploration
from visualization import VisualizationData
from preprocessing import PreprocessingData
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv('data/credit_customers_upload.csv')


if __name__ == "__main__":

    # Data exploration
    print("--------------------------------------------")
    DataExploration.display_columns_by_type(df)
    print("--------------------------------------------")
    DataExploration.display_unique_values(df)
    print("----------------------------------------------------------------")
    DataExploration.display_missing_values(df)
    print("----------------------------------------------------------------")
    DataExploration.explore_data(df)
    print("----------------------------------------------------------------")
    DataExploration.data_correlation(df)
    print("----------------------------------------------------------------")

    # Data visualization
    VisualizationData.numeric_histograms(df)
    print("----------------------------------------------------------------")
    VisualizationData.scatter_plots(df)
    print("----------------------------------------------------------------")
    VisualizationData.categorical_bar_charts(df)
    print("----------------------------------------------------------------")
    VisualizationData.boxplot(df)
    print("----------------------------------------------------------------")
    VisualizationData.violin_plots(df)
    print("----------------------------------------------------------------")
    VisualizationData.pairplot(df)
    print("----------------------------------------------------------------")
    VisualizationData.heatmap_correlation(df)
    print("----------------------------------------------------------------")

    # Preprocessing
    preprocessor = PreprocessingData()

    columns_to_work_on = ['Age','Nb_credits','Duree_credit', 'Montant_credit']

    for colonne in columns_to_work_on:
        df_clean = preprocessor.remove_outliers(df, colonne)
    print(df_clean)
    print("----------------------------------------------------------------")

    VisualizationData.boxplot(df_clean, columns=columns_to_work_on)

    df_processed = PreprocessingData.preprocess_special_values(df_clean)
    print(df_processed)
    print("----------------------------------------------------------------")

    columns_to_encode = [
        'Comptes','Historique_credit','Objet_credit','Epargne',
        'Anciennete_emploi','Situation_familiale','Garanties',
        'Biens','Autres_credits', 'Statut_domicile',
        'Type_emploi', 'Telephone', 'Etranger']

    df_encoded = PreprocessingData.encode_onehot(df_processed, columns_to_encode)

    df_encoded = PreprocessingData.convert_bool_to_float(df_encoded)
    print(df_encoded)
    print("----------------------------------------------------------------")
    DataExploration.display_columns_by_type(df_encoded)
    print("----------------------------------------------------------------")


    # Scaling data
    df_scaled = PreprocessingData.scale_data(df_encoded)
    print(df_scaled)
    print("----------------------------------------------------------------")



