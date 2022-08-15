# Uteis:
import os
from typing import Callable, Tuple, Dict, List, Text, Union

# Pandas:
import pandas as pd
from pandas.api.types import is_numeric_dtype

# Modelo:
from sklearn.base import BaseEstimator

# Pipeline e Composição:
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.compose import ColumnTransformer

# Preprocessamento e inputação de dados:
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import OrdinalEncoder

# Seleção de dados e métricas:
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve,\
    auc

# Arrays e matrizes multidimensionais:
import numpy as np

# Visualização de dados:
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('mode.chained_assignment', None)

INPUT_PATH = './dados'
PREPROCESSED_PATH = './preprocessados'
OUTPUT_PATH = './resultados'
rand_state = 1234


def preprocessing(
    filename: Text,
    intermediate_file: Text,
    reader: Callable[[pd.DataFrame, int, bool], pd.DataFrame],
    version: int = 0,
    verbose_data: bool = True
) -> pd.DataFrame:
    print("\n\nLoad dados originais: ")
    df = pd.read_csv(
        os.path.join(
            INPUT_PATH,
            filename
        )
    )
    df_total = reader(df, version, verbose_data)
    df_total.to_csv(
        os.path.join(
            PREPROCESSED_PATH,
            intermediate_file
        ),
        index=False
    )

    return df_total


def load_preprocessed(
    intermediate_file: Text,
    verbose_data: bool = True
) -> pd.DataFrame:
    print("\n\nLoad dados pré-salvos: ")
    df_total = pd.read_csv(
        os.path.join(
            PREPROCESSED_PATH,
            intermediate_file
        ),
        index_col=None
    )

    if verbose_data:
        print(df_total)

    return df_total


def print_columns_values(df: pd.DataFrame, column_names: List[Text] = []) -> None:
    if (len(column_names) == 0):
        column_names = df.columns.values.tolist()
    for column in column_names:
        valores = df[column].unique()
        sorted(valores, key=lambda x: str(x))
        print("Valores na coluna '" + column + "': \n[")
        virgula = ","
        for i, value in enumerate(valores):
            if (i == len(valores) - 1):
                virgula = ""
            print("\t", str(value) + virgula)
        print("]")


def get_dependent_variable(
    df: pd.DataFrame,
    dependent_column_name: Text
) -> Tuple[pd.DataFrame, np.ndarray]:
    y = df[dependent_column_name].values
    X = df.drop([dependent_column_name], axis=1)
    return X, y


def get_numeric_and_categoric(
    df: pd.DataFrame
) -> Tuple[List[Text], List[Text], List[List[Text]]]:
    features_numeric = []
    features_categoric = []
    categories = []
    for name, col in df.items():
        if is_numeric_dtype(col):
            features_numeric.append(name)
        else:
            features_categoric.append(name)
            col_categ = col.unique()
            col_categ = list(col_categ[~pd.isnull(col_categ)])
            col_categ.sort()
            categories.append(col_categ)
    return features_numeric, features_categoric, categories


def get_preprocessor(
    features_numeric: List[Text],
    features_categoric: List[Text],
    categories: List[List[Text]]
) -> FeatureUnion:
    imputer_numeric = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    imputer_categoric = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                  ('numerize', OrdinalEncoder(categories=categories))])

    features_preprocessor = ColumnTransformer([('imputer_numeric', imputer_numeric, features_numeric),
                                               ('imputer_categoric', imputer_categoric, features_categoric)])

    preprocessor = FeatureUnion([('features', features_preprocessor),
                                 ('indicators', MissingIndicator())])
    return preprocessor


def transform_dataset(
    preprocessor: FeatureUnion,
    X: pd.DataFrame,
    features_numeric: List[Text],
    features_categoric: List[Text]
) -> pd.DataFrame:
    X_transformed = preprocessor.transform(X)
    indicators = dict(preprocessor.transformer_list).get('indicators')

    new_columns = [X.columns[index] + '_na' for index in indicators.features_]
    all_columns = features_numeric + features_categoric + new_columns

    new_X = pd.DataFrame(X_transformed, columns=all_columns, index=X.index)

    return new_X


def fit_transform_dataset(
    preprocessor: FeatureUnion,
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    features_numeric: List[Text],
    features_categoric: List[Text]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    preprocessor.fit(X_train)

    new_X_train = transform_dataset(
        preprocessor, X_train, features_numeric, features_categoric)
    new_X_valid = transform_dataset(
        preprocessor, X_valid, features_numeric, features_categoric)
    new_X_test = None
    if (X_test is not None):
        new_X_test = transform_dataset(
            preprocessor, X_test, features_numeric, features_categoric)

    return new_X_train, new_X_valid, new_X_test


def print_score(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame = None,
    y_valid: np.ndarray = None,
    X_test: pd.DataFrame = None,
    y_test: np.ndarray = None,
    f1_pos_label: Union[int, Text] = 1,
    f1_average: Text = "binary"
) -> None:
    y_train_pred = model.predict(X_train)
    acc_train = accuracy_score(y_train, y_train_pred)
    f1_train = f1_score(y_train, y_train_pred,
                        pos_label=f1_pos_label, average=f1_average)
    auc_train = roc_auc_score(y_train, y_train_pred)

    acc_valid = None
    f1_valid = None
    auc_valid = None
    if (X_valid is not None) and (y_valid is not None):
        y_valid_pred = model.predict(X_valid)
        acc_valid = accuracy_score(y_valid, y_valid_pred)
        f1_valid = f1_score(y_valid, y_valid_pred,
                            pos_label=f1_pos_label, average=f1_average)
        auc_valid = roc_auc_score(y_valid, y_valid_pred)

    acc_test = None
    f1_test = None
    auc_test = None
    if (X_test is not None) and (y_test is not None):
        y_test_pred = model.predict(X_test)
        acc_test = accuracy_score(y_test, y_test_pred)
        f1_test = f1_score(y_test, y_test_pred,
                           pos_label=f1_pos_label, average=f1_average)
        auc_test = roc_auc_score(y_test, y_test_pred)

    print('[Train]:')
    print('\tAccuracy Score:', acc_train)
    print('\tF1 Score      :', f1_train)
    print('\tAUC Score     :', auc_train)
    if (acc_valid is not None) and (f1_valid is not None) and (auc_valid is not None):
        print('[Valid]:')
        print('\tAccuracy Score:', acc_valid)
        print('\tF1 Score      :', f1_valid)
        print('\tAUC Score     :', auc_valid)
    if (acc_test is not None) and (f1_test is not None) and (auc_test is not None):
        print('[Test]:')
        print('\tAccuracy Score:', acc_test)
        print('\tF1 Score      :', f1_test)
        print('\tAUC Score     :', auc_test)


def get_feature_importance(
    model: BaseEstimator,
    X_trn: pd.DataFrame,
    y_trn: np.ndarray,
    random_state: int = None
) -> pd.DataFrame:
    dict_importance = {'cols': [], 'imp': []}
    # trn_score = model.score(X_trn, y_trn)
    y_trn_pred = model.predict(X_trn)
    trn_score = accuracy_score(y_trn, y_trn_pred)

    for (columnName, columnData) in X_trn.iteritems():
        X_trn_copy = X_trn.copy()
        X_trn_copy[columnName] = np.random.RandomState(
            seed=random_state).permutation(columnData.values)

        # trn_column_score = model.score(X_trn_copy, y_trn)
        y_trn_column_pred = model.predict(X_trn_copy)
        trn_column_score = accuracy_score(y_trn, y_trn_column_pred)

        dict_importance['cols'].append(columnName)
        dict_importance['imp'].append(
            abs(trn_score - trn_column_score) / trn_score)

    return pd.DataFrame(data=dict_importance).sort_values(by=['imp'], ascending=False)


def replace_to_filename(filename: Text) -> Text:
    return '_'.join(filename.split())


def save_roc_curve_binary(
    model: BaseEstimator,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    algorithm: Text,
    dataset_type: Text,
    class_name_dict: Dict[int, Text]
) -> None:
    n_classes = 1
    y_valid_pred = model.predict(X_valid)

    y_valid_pred_class = np.reshape(y_valid_pred, (-1, 1))
    y_valid_class = np.reshape(y_valid, (-1, 1))

    save_roc_curve(y_valid_class, y_valid_pred_class, algorithm,
                   dataset_type, class_name_dict, n_classes)


def save_roc_curve(
    y_valid: np.ndarray,
    y_valid_pred: np.ndarray,
    algorithm: Text,
    dataset_type: Text,
    class_name_dict: Dict[int, Text],
    n_classes: int = None
) -> None:
    if n_classes is None:
        n_classes = len(y_valid[0])

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_valid[:, i], y_valid_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_valid.ravel(), y_valid_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    for i in range(n_classes):
        class_name = class_name_dict.get(i, str(i+1))
        save_roc_curve_class(
            fpr, tpr, roc_auc, i,
            f"Curva ROC - {algorithm} para {class_name} ({dataset_type})",
            replace_to_filename(
                f"ROC_{algorithm}_{dataset_type}_{class_name}.png")
        )


def save_roc_curve_class(
    fpr: Dict[int, np.ndarray],
    tpr: Dict[int, np.ndarray],
    roc_auc: Dict[int, float],
    class_index: int,
    title: Text,
    filename: Text
) -> None:
    plt.figure()
    lw = 2
    plt.plot(
        fpr[class_index],
        tpr[class_index],
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc[class_index],
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(OUTPUT_PATH, filename))


def get_extension(filename: Text) -> Tuple[Text, Text]:
    filename_parts = filename.split(".")
    return ".".join(filename_parts[:-1]), filename_parts[-1]


def get_data(
    filename: Text,
    reader: Callable[[pd.DataFrame, int, bool], pd.DataFrame],
    transform_columns: Callable[[pd.DataFrame, int, bool], pd.DataFrame],
    version: int = 0,
    verbose_data: bool = True
) -> pd.DataFrame:
    df_total = None
    filename_without_ext, filename_extension = get_extension(filename)
    intermediate_file = f"{filename_without_ext}_{version}.{filename_extension}"
    if os.path.exists(os.path.join(PREPROCESSED_PATH, intermediate_file)):
        df_total = load_preprocessed(intermediate_file, verbose_data)
    else:
        df_total = preprocessing(
            filename,
            intermediate_file,
            reader,
            version,
            verbose_data
        )

    df_total = transform_columns(df_total, version, verbose_data)

    return df_total


def split_data(
    df_total: pd.DataFrame,
    dependent_column_name: Text,
    verbose_data: bool = True
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray
]:
    if verbose_data:
        print("\n\n\nValores das colunas:")
        print_columns_values(df_total)

    print("\n\n\nDividindo entre variáveis dependentes e independentes:")
    X, y = get_dependent_variable(df_total, dependent_column_name)

    if verbose_data:
        print("X:", X.head(5))
        print("y:", y[:5])

    # Dividindo data entre treino e validação:
    X_train, X_valid_test, y_train, y_valid_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=rand_state,
        shuffle=True
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_valid_test, y_valid_test,
        test_size=0.50,
        random_state=rand_state,
        shuffle=True
    )

    if verbose_data:
        print(
            '\n\nTreinamento :', X_train.shape[0],
            'linhas e', X_train.shape[1], 'colunas'
        )
        print(
            'Validação   : ', X_valid.shape[0],
            'linhas e', X_valid.shape[1], 'colunas'
        )
        print(
            'Teste       : ', X_test.shape[0],
            'linhas e', X_test.shape[1], 'colunas'
        )

    return X, X_train, X_valid, X_test, y, y_train, y_valid, y_test


def preprocessing_data(
    X: pd.DataFrame,
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    verbose_data: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FeatureUnion]:
    features_numeric, features_categoric, categories = get_numeric_and_categoric(
        X)

    if verbose_data:
        for fc, cat in zip(features_categoric, categories):
            print('Categoric (5 valores): ', fc, ' -> ', cat[:5])

        print('\nQuantidade de Colunas :', len(X_train.columns))

        print("\n\nX_train antes de ser preprocessado:\n", X_train)

    preprocessor = get_preprocessor(
        features_numeric, features_categoric, categories)
    X_train, X_valid, X_test = fit_transform_dataset(
        preprocessor, X_train, X_valid, X_test, features_numeric, features_categoric)

    if verbose_data:
        print(
            '\n\nTreinamento :', X_train.shape[0],
            'linhas e', X_train.shape[1], 'colunas'
        )
        print(
            'Validação   : ', X_valid.shape[0],
            'linhas e', X_valid.shape[1], 'colunas'
        )
        print(
            'Teste       : ', X_test.shape[0],
            'linhas e', X_test.shape[1], 'colunas'
        )

        print("\n\nX_train após ser preprocessado:\n", X_train)

    return X_train, X_valid, X_test, preprocessor


def train_model(
    reg: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: np.ndarray
) -> BaseEstimator:
    reg.fit(X_train, y_train)
    return reg


def exec_feature_importance(
    reg: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    algorithm: Text
) -> pd.DataFrame:
    fi = get_feature_importance(reg, X_train, y_train, random_state=rand_state)

    print("\n\n\nFeature importance: ")
    for index_fi_row, fi_row in fi.iterrows():
        print(
            "Coluna '" + fi_row["cols"] + "' tem importancia: ", fi_row["imp"]
        )

    sns.set(style="whitegrid")

    f, ax = plt.subplots(figsize=(15, 8))

    sns.set_color_codes("muted")
    sns.barplot(
        x="imp",
        y="cols",
        data=fi.sort_values(
            by=['imp'],
            ascending=True
        ),
        label="Importance",
        color="b",
        ax=ax
    )

    ax.legend(ncol=1, frameon=True)
    ax.set(
        xlim=(0.0, 0.10),
        ylabel="",
        xlabel=f"Feature Importance - {algorithm}"
    )
    sns.despine(left=True, bottom=True)

    plt.savefig(
        os.path.join(OUTPUT_PATH, replace_to_filename(
            f"feature_importance_{algorithm}.png"
        ))
    )

    return fi


def filter_by_feature_importance(
    fi: pd.DataFrame,
    fi_threshold: float,
    feat_imp_columns_to_keep: List[Text],
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Text]]:
    if feat_imp_columns_to_keep is None:
        to_keep = list(fi[fi.imp > fi_threshold].cols)
        fi_cols = list(fi.cols)
    else:
        to_keep = feat_imp_columns_to_keep
        fi_cols = list(X_train.columns)

    to_keep.sort()
    fi_cols.sort()

    X_train_keep = X_train[to_keep].copy()
    X_valid_keep = X_valid[to_keep].copy()
    X_test_keep = X_test[to_keep].copy()

    print("Colunas (original): [")
    for col in fi_cols:
        print("\t", col)
    print(f"] ({len(fi_cols)} colunas)")
    print("Colunas (filtradas): [")
    for col in to_keep:
        print("\t", col)
    print(f"] ({len(to_keep)} colunas)")

    return X_train_keep, X_valid_keep, X_test_keep, to_keep


def save_roc(
    reg: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame = None,
    y_valid: np.ndarray = None,
    X_test: pd.DataFrame = None,
    y_test: np.ndarray = None,
    algorithm: Text = 'algorithm'
) -> None:
    class_name_dict = {
        0: "Vitória Azul"
    }

    dataset_type = "Train"
    save_roc_curve_binary(
        reg,
        X_train,
        y_train,
        algorithm,
        dataset_type,
        class_name_dict
    )

    if (X_valid is not None) and (y_valid is not None):
        dataset_type = "Valid"
        save_roc_curve_binary(
            reg,
            X_valid,
            y_valid,
            algorithm,
            dataset_type,
            class_name_dict
        )

    if (X_test is not None) and (y_test is not None):
        dataset_type = "Test"
        save_roc_curve_binary(
            reg,
            X_test,
            y_test,
            algorithm,
            dataset_type,
            class_name_dict
        )


def data_science_algorithm(
    filename: Text,
    reader: Callable[[pd.DataFrame, int, bool], pd.DataFrame],
    transform_columns: Callable[[pd.DataFrame, int, bool], pd.DataFrame],
    dependent_column_name: Text,
    get_model: Callable[[bool, int], Tuple[BaseEstimator, Text]],
    feature_importance_threshold: float = 0.005,
    is_relative_feat_imp_threshold: bool = False,
    feat_imp_columns_to_keep: List[Text] = None,
    show_test_restults: bool = True,
    f1_pos_label: Union[int, Text] = 1,
    f1_average: Text = "binary",
    version: int = 0,
    verbose_data: bool = True
) -> List[Text]:
    print("\n\n\nObtendo os dados:")
    df_total = get_data(
        filename,
        reader,
        transform_columns,
        version,
        verbose_data
    )

    print("\n\n\nSelecionando os dados:")
    X, X_train, X_valid, X_test, y, y_train, y_valid, y_test = split_data(
        df_total,
        dependent_column_name,
        verbose_data
    )

    print("\n\n\nInputando dados faltantes e encodificação (preprocessamento):")
    X_train, X_valid, X_test, _ = preprocessing_data(
        X,
        X_train,
        X_valid,
        X_test,
        verbose_data
    )
    reg, algorithm = get_model(False, rand_state)

    print(f"\n\n\nTreinando o modelo {algorithm} pela primeira vez:")
    reg = train_model(
        reg,
        X_train,
        y_train
    )
    print_score(
        model=reg,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        f1_pos_label=f1_pos_label,
        f1_average=f1_average
    )

    fi = None
    fi_threshold = None
    if feat_imp_columns_to_keep is None:
        print("\n\n\nCalculando a feature importance:")
        fi = exec_feature_importance(reg, X_train, y_train, algorithm)

        print("\n\n\nRemovendo colunas menos importantes:")
        fi_threshold = feature_importance_threshold
        if (is_relative_feat_imp_threshold):
            fi_threshold = feature_importance_threshold * fi.imp.max()
        print("Threshold: ", fi_threshold)
    X_train_keep, X_valid_keep, X_test_keep, current_to_keep = filter_by_feature_importance(
        fi,
        fi_threshold,
        feat_imp_columns_to_keep,
        X_train,
        X_valid,
        X_test
    )

    print(
        f"\n\n\nTreinando o modelo {algorithm} otimizado para as colunas mais importantes:"
    )
    reg, _ = get_model(True, rand_state)
    reg = train_model(
        reg,
        X_train_keep,
        y_train
    )
    print_score(
        model=reg,
        X_train=X_train_keep,
        y_train=y_train,
        X_valid=X_valid_keep,
        y_valid=y_valid,
        X_test=X_test_keep if show_test_restults else None,
        y_test=y_test if show_test_restults else None,
        f1_pos_label=f1_pos_label,
        f1_average=f1_average
    )
    save_roc(
        reg=reg,
        X_train=X_train_keep,
        y_train=y_train,
        X_valid=X_valid_keep,
        y_valid=y_valid,
        X_test=X_test_keep if show_test_restults else None,
        y_test=y_test if show_test_restults else None,
        algorithm=algorithm
    )
    return current_to_keep
