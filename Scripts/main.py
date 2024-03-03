import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from exploration import DataExploration
from visualization import VisualizationData
from preprocessing import PreprocessingData
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from train_and_evaluate import TrainEvaluateData

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold

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
    #DataExploration.display_columns_by_type(df_encoded)
    print("----------------------------------------------------------------")

    df_scaled = PreprocessingData.scale_data(df_encoded)
    print(df_scaled)
    print("----------------------------------------------------------------")

    # Train and Evaluation
    X = df_scaled.drop(columns=['Cible'])  # Caractéristiques
    y = df_scaled['Cible']  # Cible

    train_evaluator = TrainEvaluateData(X, y)
    X_train, X_test, y_train, y_test = train_evaluator.split_data()
    models = train_evaluator.initialize_models()

    print("Training and evaluating models:")
    train_evaluator.train_evaluate_models(models, X_train, X_test, y_train, y_test)

"""
    print("\nCross-validation results:")
    train_evaluator.cross_validation(models, X_train, y_train)

    print("\nEvaluation metrics:")
    train_evaluator.evaluate_metrics(models, X_train, X_test, y_train, y_test)
    """


