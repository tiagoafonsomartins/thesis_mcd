import io
import os
import sys
import pandas as pd
import auxiliary_functions as af


# Defining categorical and numerical columns for Credit Card
default_credit_cat_cols = ["X2", "X3", "X4"]
default_credit_num_cols = ["X0",
                           "X1",
                           "X5",
                           "X6",
                           "X7",
                           "X8",
                           "X9",
                           "X10",
                           "X11",
                           "X12",
                           "X13",
                           "X14",
                           "X15",
                           "X16",
                           "X17",
                           "X18",
                           "X19",
                           "X20",
                           "X21",
                           "X22",
                           "X23",
                           "Y"]
default_credit_cat_cols_num = [2, 3, 4]
default_credit_num_cols_num = [5, 6, 7, 8, 9 ,10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

# Defining categorical and numerical columns for German Credit
german_cat_cols = ["status",
                   "history",
                   "purpose",
                   "savings",
                   "employment_since",
                   "personal_status",
                   "debtors","property",
                   "installment_plans",
                   "housing",
                   "job",
                   "telephone",
                   "foreign_worker",
                   "risk"
                   ]
german_cat_cols_num = [1, 4, 7, 10, 12, 15, 17]
german_num_cols = ["duration",
                   "amount",
                   "installment_rate",
                   "residence_since",
                   "age",
                   "credits_bank",
                   "liable_to_maintenance"]
german_num_cols_num = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19, 20]



# Example usage
# Switching to "test" environment
#switch_env("test")

# Importing method from "methods" sub-folder
#method = af.import_method("methods")

# Reading "data.csv" file present in "data" sub-folder
default_credit = pd.read_csv("Datasets/default of credit card clients.csv", index_col = "ID", delimiter=';', header=0)
german_credit = pd.read_csv("Datasets/german_data.csv", delimiter=';', header=0)
german_credit_num = pd.read_csv("Datasets/german.data-numeric.csv", delimiter=';', header=0)
heloc = pd.read_csv("Datasets/heloc_dataset_v1.csv", delimiter=',', header=0)

# Default Credit - Data preparation and split into train/test
x_train, x_test, y_train, y_test = af.data_prep(default_credit, "default_credit.txt", "Y", replacer = [0, 1])


# German Credit - Data preparation and split into train/test
#x_train, x_test, y_train, y_test = af.data_prep(german_credit, "german_credit.txt", "risk", replacer = [1, 2])
# German Credit - Data preparation and split into train/test
x_train, x_test, y_train, y_test = af.data_prep(german_credit_num, "german_credit.txt", "risk", replacer = [1, 2])


# HELOC - Data preparation and split into train/test
x_train, x_test, y_train, y_test = af.data_prep(heloc, "heloc.txt", ["RiskPerformance"], ["Good", "Bad"], replacer = [0, 1])