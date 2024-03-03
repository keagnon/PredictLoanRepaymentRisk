import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from exploration import DataExploration
from visualization import VisualizationData
from preprocessing import PreprocessingData

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
    print("----------------------------------------------------------------")
    DataExploration.data_correlation(df)
    print("----------------------------------------------------------------")

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

    columns_to_work_on = ['Age','Nb_credits','Duree_credit', 'Montant_credit']

    for colonne in columns_to_work_on:
        df_clean= PreprocessingData.remove_outliers(df, colonne)
    print(df_clean)
    print("----------------------------------------------------------------")

    df_processed = PreprocessingData.preprocess_special_values(df_storage)
    df_processed

colonnes_catégorielles = ['Comptes', 'Historique_credit', 'Objet_credit', 'Epargne', 'Anciennete_emploi', 'Situation_familiale', 'Garanties', 'Biens', 'Autres_credits', 'Statut_domicile', 'Type_emploi', 'Telephone', 'Etranger']
# Appliquer le codage one-hot
df_encoded = encode_onehot(df_processed, colonnes_catégorielles)
df_encoded.columns
afficher_colonnes_par_type(df_encoded)

df_encoded = convert_bool_to_float(df_encoded)
df_encoded
afficher_colonnes_par_type(df_encoded)


# Appliquer la normalisation
df_preprocessed = scale_data(df_encoded)



