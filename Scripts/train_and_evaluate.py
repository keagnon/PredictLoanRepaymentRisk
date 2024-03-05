from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import joblib
import os

class TrainEvaluateData:
    """
    Train and evaluate various machine learning models.
    """
    def __init__(self, X, y):
        """
        Initialize the class with features and target variables.

        Args:
            X (DataFrame): Features.
            y (Series): Target variable.
        """
        self.X = X
        self.y = y

    def split_data(self):
        """
        Split the data into training and testing sets.

        Returns:
            tuple: X_train, X_test, y_train, y_test.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def initialize_models(self):
        """
        Initialize machine learning models.

        Returns:
            dict: Dictionary of initialized models.
        """
        models = {
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Logistic Regression": LogisticRegression(max_iter=2500),
            "Support Vector Machine": SVC(),
            "Artificial Neural Network": MLPClassifier(),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB()
        }
        return models

    def train_evaluate_models(self, models, X_train, X_test, y_train, y_test):
            """
            Train and evaluate machine learning models.

            params:
                models (dict): Dictionary of initialized models.
                X_train (DataFrame): Training features.
                X_test (DataFrame): Testing features.
                y_train (Series): Training target.
                y_test (Series): Testing target.
            """
            accuracies = {}
            for name, model in models.items():
                if name == "Logistic Regression":
                    # Mettre à l'échelle des données pour la régression logistique
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    # Initialiser le modèle avec un nombre maximal d'itérations plus élevé
                    model = LogisticRegression(max_iter=2100)

                    # Entraîner le modèle
                    model.fit(X_train_scaled, y_train)

                    # Faire des prédictions
                    y_pred = model.predict(X_test_scaled)
                else:
                    # Pas besoin de mise à l'échelle pour les autres modèles
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                accuracies[name] = accuracy

            sorted_accuracies = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
            for name, accuracy in sorted_accuracies:
                print(f"{name}: {accuracy}")

    def cross_validation(self, models, X_train, y_train):
        """
        Apply cross-validation for each model.

        Params:
            models (dict): Dictionary of initialized models.
            X_train (DataFrame): Training features.
            y_train (Series): Training target.
        """
        for name, model in models.items():
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
            print(f"{name}: {scores.mean()}")

    def evaluate_metrics(self, models, X_train, X_test, y_train, y_test):
        """
        Calculate evaluation metrics for each model.

        Params:
            models (dict): Dictionary of initialized models.
            X_train (DataFrame): Training features.
            X_test (DataFrame): Testing features.
            y_train (Series): Training target.
            y_test (Series): Testing target.
        """
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred)
            confusion = confusion_matrix(y_test, y_pred)

            print(f"{name}:")
            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1-score: {f1}")
            print(f"ROC-AUC: {roc_auc}")
            print(f"Confusion matrix:\n{confusion}\n")

    def tune_random_forest(X_train, y_train):
        param_grid_rf = {
            'n_estimators': [150, 290, 450],
            'max_depth': [None, 15, 25],
            'min_samples_split': [3, 7, 15],
            'min_samples_leaf': [2, 4, 9]
        }
        grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=3, scoring='accuracy')
        grid_search_rf.fit(X_train, y_train)
        return grid_search_rf

    def tune_gradient_boosting(X_train, y_train):
        param_grid_gbt = {
            'n_estimators': [58, 185, 380],
            'learning_rate': [0.06, 0.2, 0.7],
            'max_depth': [2, 4, 9]
        }
        grid_search_gbt = GridSearchCV(GradientBoostingClassifier(), param_grid_gbt, cv=3, scoring='accuracy')
        grid_search_gbt.fit(X_train, y_train)
        return grid_search_gbt


    def save_best_model(model, model_name, iteration):
        """
        Save the best trained model to a file using joblib.

        Params:
            model: Trained model object.
            model_name (str): Name of the model.
            iteration (int): Iteration number of the script.

        Returns:
            str: Path to the saved model file.
        """
        models_dir = f"models/iteration_{iteration}"
        os.makedirs(models_dir, exist_ok=True)  # Create directory if it doesn't exist
        model_path = f"{models_dir}/{model_name}_best_model.pkl"
        joblib.dump(model, model_path)
        print(f"Best {model_name} model saved successfully to:", model_path)
        return model_path

    def score_after_tune(X_train, y_train,iteration_number):
        rf_grid_search = TrainEvaluateData.tune_random_forest(X_train, y_train)
        gb_grid_search = TrainEvaluateData.tune_gradient_boosting(X_train, y_train)

        print("Random Forest - Best Parameters:", rf_grid_search.best_params_)
        print("Random Forest - Best Score:", rf_grid_search.best_score_)
        print("Gradient Boosting - Best Parameters:", gb_grid_search.best_params_)
        print("Gradient Boosting - Best Score:", gb_grid_search.best_score_)

        # Enregistrer le meilleur modèle Random Forest
        best_rf_model = rf_grid_search.best_estimator_
        TrainEvaluateData.save_best_model(best_rf_model, "random_forest", iteration_number)

        # Enregistrer le meilleur modèle Gradient Boosting
        best_gb_model = gb_grid_search.best_estimator_
        TrainEvaluateData.save_best_model(best_gb_model, "gradient_boosting", iteration_number)


    def tune_models_ensemble(X_train, y_train, X_test, y_test):
        rf_param_grid = {
            'n_estimators': [350, 450, 500],
            'max_depth': [11, 15, 21],
            'min_samples_split': [2, 5, 12],
            'min_samples_leaf': [1, 3, 6]
        }
        gb_param_grid = {
            'n_estimators': [55, 220, 250],
            'learning_rate': [0.01, 0.07, 0.3],
            'max_depth': [1, 6, 9]
        }
        rf = RandomForestClassifier(random_state=42)
        gb = GradientBoostingClassifier(random_state=42)
        cv_strat = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        rf_grid = GridSearchCV(rf, rf_param_grid, cv=cv_strat, scoring='accuracy', n_jobs=-1)
        rf_grid.fit(X_train, y_train)

        gb_grid = GridSearchCV(gb, gb_param_grid, cv=cv_strat, scoring='accuracy', n_jobs=-1)
        gb_grid.fit(X_train, y_train)

        best_rf = rf_grid.best_estimator_
        best_gb = gb_grid.best_estimator_

        ensemble_model = VotingClassifier(estimators=[('rf', best_rf), ('gb', best_gb)], voting='hard')
        ensemble_model.fit(X_train, y_train)

        ensemble_score = ensemble_model.score(X_test, y_test)
        print("Ensemble Model Score:", ensemble_score)


