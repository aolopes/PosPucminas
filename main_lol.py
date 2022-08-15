# Utils:
import sys
from typing import List, Tuple, Text

# Modelo:
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, SGDOneClassSVM
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Commons:
from common import data_science_algorithm
from common_lol import reader_lol, transform_columns_lol


def get_random_forest(
    tuned: bool = True,
    random_state: int = None
) -> Tuple[BaseEstimator, Text]:
    algorithm = "Random Forest Classifier"
    n_estimators_base = 1000
    min_samples_leaf_base = 20
    max_features_base = 0.75
    if tuned:
        return RandomForestClassifier(
            n_estimators=n_estimators_base,
            min_samples_leaf=min_samples_leaf_base,
            max_features=max_features_base,
            random_state=random_state,
            n_jobs=-1
        ), algorithm

    return RandomForestClassifier(
        n_estimators=n_estimators_base,
        min_samples_leaf=min_samples_leaf_base,
        max_features=max_features_base,
        random_state=random_state,
        n_jobs=-1
    ), algorithm


def get_naive_bayes(
    tuned: bool = True,
    random_state: int = None
) -> Tuple[BaseEstimator, Text]:
    algorithm = "Gaussian Naive Bayes"
    if tuned:
        return GaussianNB(), algorithm

    return GaussianNB(), algorithm


def get_logistic_regression(
    tuned: bool = True,
    random_state: int = None
) -> Tuple[BaseEstimator, Text]:
    algorithm = "Logistic Regression"
    if tuned:
        return LogisticRegression(
            random_state=random_state
        ), algorithm

    return LogisticRegression(
        random_state=random_state
    ), algorithm


def get_sgd_classifier(
    tuned: bool = True,
    random_state: int = None
) -> Tuple[BaseEstimator, Text]:
    algorithm = "SGD Classifier"
    max_iter = 10000
    tol = 1e-3
    if tuned:
        return make_pipeline(
            StandardScaler(),
            SGDClassifier(
                max_iter=max_iter,
                # tol=tol,
                random_state=random_state
            )
        ), algorithm

    return make_pipeline(
        StandardScaler(),
        SGDClassifier(
            max_iter=max_iter,
            # tol=tol,
            random_state=random_state
        )
    ), algorithm


def get_sgd_one_class_svm(
    tuned: bool = True,
    random_state: int = None
) -> Tuple[BaseEstimator, Text]:
    algorithm = "SGD One Class SVM"
    if tuned:
        return SGDOneClassSVM(
            random_state=random_state
        ), algorithm

    return SGDOneClassSVM(
        random_state=random_state
    ), algorithm


def main(argv: List[Text]):
    filename = "2022_LoL_esports_match_data_from_OraclesElixir.csv"
    version = 0
    fixed_first_feat_imp_columns = False
    if len(argv) > 1:
        version = int(argv[1])
        if len(argv) > 2:
            filename = str(argv[2])
    print(f"\nRodando para o arquivo {filename} e versão {version}:")
    dependent_column_name = "win_blue"
    models_functions = [
        tuple([get_random_forest, 0.15, True, 1, "binary"]),
        tuple([get_naive_bayes, 0.15, True, 1, "binary"]),
        tuple([get_logistic_regression, 0.15, True, 1, "binary"]),
        tuple([get_sgd_classifier, 0.15, True, 1, "binary"]),
        # tuple([get_sgd_one_class_svm, 0.15, True, "positive", "micro"]),
    ]

    for index, model_tuple in enumerate(models_functions):
        get_model, feature_importance_threshold, is_relative_feat_imp_threshold, f1_pos_label, f1_average = model_tuple
        current_to_keep = data_science_algorithm(
            filename=filename,
            reader=reader_lol,
            transform_columns=transform_columns_lol,
            dependent_column_name=dependent_column_name,
            get_model=get_model,
            feature_importance_threshold=feature_importance_threshold,
            is_relative_feat_imp_threshold=is_relative_feat_imp_threshold,
            feat_imp_columns_to_keep=None if (
                index == 0 or
                (not fixed_first_feat_imp_columns)
            ) else current_to_keep,
            show_test_restults=True,
            f1_pos_label=f1_pos_label,
            f1_average=f1_average,
            version=version,
            verbose_data=(index == 0)
        )

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
