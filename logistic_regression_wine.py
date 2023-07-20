import os
import tarfile
import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
import scipy.stats as stats
from statsmodels.miscmodels.ordinal_model import OrderedModel
import joblib


WINEQUAL_PATH = os.path.join("data", "winequality")
INIT_WINEQUAL_CSV = "winequality-red.csv"
WINEQUAL_CSV = "winequality-formatted.csv"


def format_init_data(path=WINEQUAL_PATH, input_filename=INIT_WINEQUAL_CSV, output_filename=WINEQUAL_CSV):
    outfile = os.path.join(path, output_filename)
    if (not os.path.exists(outfile)):
        infile = open(os.path.join(path, input_filename), "r")
        lines = infile.readlines()

        content = []
        columns = []
        for index, line in enumerate(lines):
            if index == 0:
                columns.extend(
                    list(map(str, (line.strip("\"\n").split("\";\"")))))
            else:
                content.append(list(map(float, (line.strip("\n").split(";")))))

        df = pd.DataFrame(content, columns=columns)
        df.to_csv(outfile, encoding="utf-8", index=False)

    print("File is formatted. Open {}", output_filename)


def load_formatted_winequal_data(path=WINEQUAL_PATH, filename=WINEQUAL_CSV):
    csv_path = os.path.join(path, filename)
    return pd.read_csv(csv_path)


def get_multinomial_model_accuracy(regressor, X, y, X_test, y_test, display_full=False):
    regressor.fit(X[list(X.keys())], y)
    y_test_pred = regressor.predict(X=X_test[list(X_test.keys())])

    if display_full:
        for (pred, actual) in zip(y_test_pred, y_test):
            print("Prediction: {:<10.5f}Actual: {:<10.5f}".format(
                pred, actual))

    return (y_test_pred == y_test).mean()


def get_ordinal_model_accuracy(res_log, X_test, y_test, display_full=False):
    predicted = res_log.model.predict(
        res_log.params, exog=X_test[list(X_test.keys())])
    pred_choice = predicted.argmax(1) + 3
    # pred_choice [0, 1, 2, 3, 4, 5] while y_test [3, 4, 5, 6, 7, 8]

    if display_full:
        for (pred, actual) in zip(pred_choice, y_test):
            print("Prediction: {:<10.5f}Actual: {:<10.5f}".format(
                pred, actual))

    return (y_test == pred_choice).mean()


if __name__ == "__main__":
    # Part 1: Data Acquisition
    file_directory = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(file_directory, WINEQUAL_PATH)
    format_init_data(path=data_file_path)
    winequal = load_formatted_winequal_data(path=data_file_path)

    print("Wine Quality data has been loaded.")

    # Part 2: Data Scaling
    # Since there are many outliers in each criteria, propose the scaling method 'standardization'
    scaling_data = winequal.copy()
    scaling_data.drop("quality", axis=1, inplace=True)

    list_headers_scaling = list(scaling_data)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(scaling_data)
    final_data = pd.DataFrame(scaled_data, columns=list_headers_scaling)

    # Add "quality" data to final dataset used for training and testing
    final_data["quality"] = winequal["quality"]

    print("Wine Quality data has been re-scaled for model training.")
    print("FINAL Wine Quality data:")
    print(final_data.describe())

    # Part 3: Data Separation
    # 05 separated items for model comparison
    model_training_set, comparison_set = train_test_split(
        final_data, test_size=5, random_state=84)
    # Test Set: 20%; Train Set: 80%
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=84)
    for train_idx, test_idx in split.split(model_training_set, model_training_set["quality"]):
        train_set = winequal.loc[train_idx]
        test_set = winequal.loc[test_idx]

    # Part 4: Data Preparation
    # Scheme 1: Original data
    # Scheme 2: Trim 03 data: residual sugar, free sulfur dioxide, pH
    comparison_set_2 = comparison_set.copy()
    train_set_2 = train_set.copy()
    test_set_2 = test_set.copy()
    comparison_set_2.drop(
        ["residual sugar", "pH", "free sulfur dioxide"], axis=1, inplace=True)
    train_set_2.drop(
        ["residual sugar", "pH", "free sulfur dioxide"], axis=1, inplace=True)
    test_set_2.drop(["residual sugar", "pH", "free sulfur dioxide"],
                    axis=1, inplace=True)
    # Scheme 3: Trim 04 data: residual sugar, free sulfur dioxide, pH, chlorides
    comparison_set_3 = comparison_set.copy()
    train_set_3 = train_set.copy()
    test_set_3 = test_set.copy()
    comparison_set_3.drop(
        ["residual sugar", "pH", "free sulfur dioxide", "chlorides"], axis=1, inplace=True)
    train_set_3.drop(
        ["residual sugar", "pH", "free sulfur dioxide",
         "chlorides"], axis=1, inplace=True)
    test_set_3.drop(["residual sugar", "pH", "free sulfur dioxide",
                     "chlorides"],
                    axis=1, inplace=True)
    # Scheme 4: Trim 'density' data, since trained ordinal models witnessed high deviation from 'density'
    comparison_set_4 = comparison_set.copy()
    train_set_4 = train_set.copy()
    test_set_4 = test_set.copy()
    comparison_set_4.drop(["density"], axis=1, inplace=True)
    train_set_4.drop(["density"], axis=1, inplace=True)
    test_set_4.drop(["density"], axis=1, inplace=True)

    # Part 5: Model Training
    print("\n***NOTE: Because some models have extremely long training and tuning time, the author only displays best results achieved in each scheme.")

    # Scheme 1: Original Data
    print("\n\n----- TRAINING SCHEME 1: Full criteria, scaled data -----")
    X_test = test_set.copy()
    X_test = X_test.drop("quality", axis=1)
    y_test = test_set["quality"]

    X = train_set.copy()
    X = X.drop("quality", axis=1)
    y = train_set["quality"]
    print("Training criteria list: {}".format(list(X.keys())))

    print("\nMultinomial model:")
    sag_reg = LogisticRegression(
        multi_class="multinomial", solver="sag", max_iter=10000, C=10)
    acc = get_multinomial_model_accuracy(
        sag_reg, X=X, y=y, X_test=X_test, y_test=y_test)
    print("\tBest accuracy: {}\n\tsolver: 'sag'\n\tpenalty: 'l2'\n\tmax_iter: 10000\n\tC: 10".format(acc))

    print("\nOrdinal model:")
    ordinal_log = OrderedModel(y, X[list(X.keys())], distr="logit")
    res_bh_log = ordinal_log.fit(
        method="basinhopping", maxiter=1000, disp=False)
    acc = get_ordinal_model_accuracy(res_bh_log, X_test=X_test, y_test=y_test)
    print("\tBest accuracy: {}\n\tmethod: 'basinhopping'\n\tmaxiter: 1000".format(acc))

    # Scheme 2: Trimmed 03 data
    print("\n\n----- TRAINING SCHEME 2: Trimmed 03 criteria, scaled data -----")
    X_test_2 = test_set_2.copy()
    X_test_2 = X_test_2.drop("quality", axis=1)
    y_test_2 = test_set_2["quality"]

    X_2 = train_set_2.copy()
    X_2 = X_2.drop("quality", axis=1)
    y_2 = train_set_2["quality"]
    print("Training criteria list: {}".format(list(X_2.keys())))

    print("\nMultinomial model:")
    saga_enet_reg_2 = LogisticRegression(
        multi_class="multinomial", solver="saga", penalty="elasticnet", l1_ratio=0.5, max_iter=16458, C=6)
    acc = get_multinomial_model_accuracy(
        saga_enet_reg_2, X=X_2, y=y_2, X_test=X_test_2, y_test=y_test_2)
    print("\tBest accuracy: {}\n\tsolver: 'saga'\n\tpenalty: 'elasticnet'/'l1'/'l2'\n\tmax_iter: 16458\n\tC: 6".format(acc))

    print("\nOrdinal model:")
    ordinal_log_2 = OrderedModel(y_2, X_2[list(X_2.keys())], distr="logit")
    res_lbfgs_log_2 = ordinal_log_2.fit(
        method="lbfgs", maxiter=1000, disp=False)
    acc = get_ordinal_model_accuracy(res_lbfgs_log_2, X_test=X_test_2, y_test=y_test_2)
    print("\tBest accuracy: {}\n\tmethod: 'lbfgs'\n\tmaxiter: 1000".format(acc))

    # Scheme 3: Trimmed 04 data
    print("\n\n----- TRAINING SCHEME 3: Trimmed 04 criteria, scaled data -----")
    X_test_3 = test_set_3.copy()
    X_test_3 = X_test_3.drop("quality", axis=1)
    y_test_3 = test_set_3["quality"]

    X_3 = train_set_3.copy()
    X_3 = X_3.drop("quality", axis=1)
    y_3 = train_set_3["quality"]
    print("Training criteria list: {}".format(list(X_3.keys())))

    print("\nMultinomial model:")
    saga_enet_reg_3 = LogisticRegression(
        multi_class="multinomial", solver="saga", penalty="elasticnet", l1_ratio=0.5, max_iter=10000, C=10)
    acc = get_multinomial_model_accuracy(
        saga_enet_reg_3, X=X_3, y=y_3, X_test=X_test_3, y_test=y_test_3)
    print("\tBest accuracy: {}\n\tsolver: 'saga'\n\tpenalty: 'elasticnet'/'l1'/'l2'\n\tmax_iter: 10000\n\tC: 10".format(acc))

    print("\nOrdinal model:")
    ordinal_log_3 = OrderedModel(y_3, X_3[list(X_3.keys())], distr="logit")
    res_lbfgs_log_3 = ordinal_log_3.fit(
        method="lbfgs", maxiter=1000, disp=False)
    acc = get_ordinal_model_accuracy(res_lbfgs_log_3, X_test=X_test_3, y_test=y_test_3)
    print("\tBest accuracy: {}\n\tmethod: 'lbfgs'/'bfgs'/'ncg'\n\tmaxiter: 1000".format(acc))

    # Scheme 4: Trimmed 'density' data
    print("\n\n----- TRAINING SCHEME 4: Trimmed 'density' criteria, scaled data -----")
    X_test_4 = test_set_4.copy()
    X_test_4 = X_test_4.drop("quality", axis=1)
    y_test_4 = test_set_4["quality"]

    X_4 = train_set_4.copy()
    X_4 = X_4.drop("quality", axis=1)
    y_4 = train_set_4["quality"]
    print("Training criteria list: {}".format(list(X_4.keys())))

    print("\nMultinomial model:")
    sag_reg_4 = LogisticRegression(
        multi_class="multinomial", solver="sag", max_iter=10000, C=10)
    acc = get_multinomial_model_accuracy(
        sag_reg_4, X=X_4, y=y_4, X_test=X_test_4, y_test=y_test_4)
    print("\tBest accuracy: {}\n\tsolver: 'sag'\n\tpenalty: 'l2'\n\tmax_iter: 10000\n\tC: 10".format(acc))

    print("\nOrdinal model:")
    ordinal_log_4 = OrderedModel(y_4, X_4[list(X_4.keys())], distr="logit")
    res_lbfgs_log_4 = ordinal_log_4.fit(
        method="lbfgs", maxiter=1000, disp=False)
    acc = get_ordinal_model_accuracy(res_lbfgs_log_4, X_test=X_test_4, y_test=y_test_4)
    print("\tBest accuracy: {}\n\tmethod: 'lbfgs'/'bfgs'/'ncg'/'newton'/'basinhopping'\n\tmaxiter: 1000".format(acc))


    # Part 6: Comparison testing
    print("\n\n----- CONCLUSION & COMPARISON TESTING -----")
    print("Most accurate multinomial model:")
    acc = get_multinomial_model_accuracy(
        saga_enet_reg_2, X=X_2, y=y_2, X_test=X_test_2, y_test=y_test_2)
    print("\tBest accuracy: {}\n\tData Scheme 02 - trimmed 03 criteria, scaled data\n\tsolver: 'saga'\n\tpenalty: 'elasticnet'/'l1'/'l2'\n\tmax_iter: 16458\n\tC: 6\n\tfine-tuned: Yes".format(acc))
    print("Most accurate ordinal model:")
    acc = get_ordinal_model_accuracy(res_bh_log, X_test=X_test, y_test=y_test)
    print("\tBest accuracy: {}\n\tData Scheme 01 - Full criteria, scaled data\n\tmethod: 'basinhopping'\n\tmaxiter: 1000\n\tfine-tuned: No".format(acc))

    
    X_comparison_test = comparison_set.copy()
    X_comparison_test = X_comparison_test.drop("quality", axis=1)
    y_comparison_test = comparison_set["quality"]

    X_comparison_test_2 = comparison_set_2.copy()
    X_comparison_test_2 = X_comparison_test_2.drop("quality", axis=1)
    y_comparison_test_2 = comparison_set_2["quality"]
    
    print("\nComparison testing:")
    print("Multinomial model:")
    acc = get_multinomial_model_accuracy(
        saga_enet_reg_2, X=X_2, y=y_2, X_test=X_comparison_test_2, y_test=y_comparison_test_2, display_full=True)
    print("Accuracy: ", acc)
    print("Ordinal model:")
    acc = get_ordinal_model_accuracy(res_bh_log, X_test=X_comparison_test, y_test=y_comparison_test, display_full=True)
    print("Accuracy: ", acc)


    # Part 7: Save Models
    joblib.dump(saga_enet_reg_2, "regression_model_saga_elasticnet_multinomial_scheme2.pkl")
    joblib.dump(res_bh_log, "regression_model_basinhopping_ordinal_scheme1.pkl")
    print("Two best models achieved have been saved.")

