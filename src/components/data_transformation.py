# Change the data format , feature engineering , data cleaning , convert it from categorical to numerical, Onehotencoding,Label encoding
# If confussed check Notebook

# python src/components/data_ingestion.py

'''import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# importing CustomException from src folder
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# DataTransformationConfig class to store configuration for data transformation
@dataclass
class DataTransformationConfig:
    # Path to save preprocessor object as 'preprocessor.pkl' inside 'artifacts' folder
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


# DataTransformation class handles creation of preprocessing pipelines
class DataTransformation:
    def __init__(self):
        # Initializing configuration for data transformation
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function creates preprocessing pipelines for both
        numerical and categorical columns and returns a combined preprocessor object.
        """
        try:
            # Defining numerical and categorical feature names
            numerical_columns = [
                "trek_rating",
                "Distance_km",
                "Duration_days",
                "Elevation_gain_m",
                "Max_altitude_m",
                "Avg_temp_C",
                "Group_size",
                "Permit_required",
                "Guide_cost_rupees",
                "Permit_fee_rupees",
                "Local_cost_index",
                "age",
                "zip_code",
                "emergency contact details",
                "weight",
                "height",
                "bmi",
                "accommodation_rating",
                "Hotel_Price_per_day",
                "wi-fi_facility",
                "breakfast",
                "lunch",
                "dinner",
                "pool",
                "bar",
            ]

            categorical_columns = [
                "trek_ID",
                "Company Name",
                "Trek_Location",
                "Region",
                "Country",
                "Industry",
                "Contact Email",
                "Difficulty",
                "Season",
                "Weather",
                "Accessibility",
                "Accommodation",
                "Currency",
                "Best_Season",
                "Best_Month",
                "surname",
                "address",
                "phone_no",
                "gender",
                "city",
                "state",
                "traveler_name",
                "health_check_up",
                "payment status",
                "booking_status",
                "traveller_profession",
                "payment information",
                "identification documents (for permits)",
                "Hotel_Name",
                "City",
                "trek_start_date",
                "trek_end_date",
                "trek_type",
                "Backpack",
                "Footwear",
                "waterproof/windproof jackets",
                "trekking_shoes",
                "Lighting",
                "Cash/online",
            ]

            # Pipeline for numerical data:
            # - Fill missing values with median
            # - Standardize numerical features
            # x = fit_transform(x) , x_train
            num_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median"),
                    ),  # handle missing values
                    ("scaler", StandardScaler()),
                ]
            )

            # Pipeline for categorical data:
            # - Fill missing values with most frequent category
            # - Convert categories into one-hot encoded variables
            # - Scale encoded values (without centering)
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            # Logging for debugging/tracking
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info("Categorical columns encoding pipeline created successfully")
            logging.info("Numerical columns scaling pipeline created successfully")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combining both pipelines using ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            # Returning the combined preprocessor object
            return preprocessor

        except Exception as e:
            # Raising custom exception for better error tracking
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Reads training and test datasets, applies preprocessing transformations,
        saves the preprocessor object, and returns transformed arrays.
        """
        try:
            # Reading the train and test CSV files into pandas DataFrames
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded successfully")
            logging.info(f"train_df shape: {train_df.shape}")
            logging.info(f"test_df shape: {test_df.shape}")
            logging.info("Obtaining preprocessing object")

            # Get the preprocessor object (created in get_data_transformer_object)
            preprocessing_obj = self.get_data_transformer_object()

            # Define target column and numerical columns
            target_column_name = "trek_total_cost(rupees)"
            numerical_columns = [
                "trek_rating",
                "Distance_km",
                "Duration_days",
                "Elevation_gain_m",
                "Max_altitude_m",
                "Avg_temp_C",
                "Group_size",
                "Permit_required",
                "Guide_cost_rupees",
                "Permit_fee_rupees",
                "Local_cost_index",
                "age",
                "zip_code",
                "emergency contact details",
                "weight",
                "height",
                "bmi",
                "accommodation_rating",
                "Hotel_Price_per_day",
                "wi-fi_facility",
                "breakfast",
                "lunch",
                "dinner",
                "pool",
                "bar",
            ]

            # Separate input features (X) and target variable (y) for training data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[[target_column_name]]

            # Separate input features (X) and target variable (y) for testing data
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[[target_column_name]]

            print("🧪 Type of input_feature_train_df:", type(input_feature_train_df))
            print("🧪 First few rows:\n", input_feature_train_df.head())
            print(
                "📋 Columns in input_feature_train_df:",
                input_feature_train_df.columns.tolist(),
            )

            logging.info(
                f"input_feature_train_df shape: {input_feature_train_df.shape}"
            )
            logging.info(
                f"target_feature_train_df shape: {target_feature_train_df.shape}"
            )
            logging.info(f"input_feature_test_df shape: {input_feature_test_df.shape}")
            logging.info(
                f"target_feature_test_df shape: {target_feature_test_df.shape}"
            )
            logging.info("Applying preprocessing on training and test data")

            # Fit preprocessor on training data and transform both train & test sets
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            print("🔍 input_feature_train_arr shape:", input_feature_train_arr.shape)
            print(
                "🔍 target_feature_train_df shape:",
                np.array(target_feature_train_df).shape,
            )
            print(f"input_feature_train_arr shape: {input_feature_train_arr.shape}")
            print(
                f"target_feature_train_df.values shape: {target_feature_train_df.values.shape}"
            )
            print(
                f"target_feature_train_df.values ndim: {target_feature_train_df.values.ndim}"
            )

            # Combine input features and target variable back into arrays
            """train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]"""

            train_arr = np.hstack(
                [input_feature_train_arr, target_feature_train_df.values]
            )

            test_arr = np.concatenate(
                [input_feature_test_arr, target_feature_test_df.values.reshape(-1, 1)],
                axis=1,
            )

            logging.info(
                f"Transformed input_feature_train_arr shape: {input_feature_train_arr.shape}"
            )
            logging.info(
                f"Transformed input_feature_test_arr shape: {input_feature_test_arr.shape}"
            )

            logging.info("Preprocessing complete. Saving preprocessor object.")

            # Save the fitted preprocessor object to the specified file path
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            # Return processed train & test arrays and preprocessor path
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            # Raise a custom exception for better error tracking
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataTransformation()
    train_arr, test_arr, _ = obj.initiate_data_transformation()

'''

# _______________________________________________________________comment

# python src/components/data_ingestion.py
# python src/components/data_transformation.py
# File: trek_cost_prediction.py

# ========== Imports ==========
"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Sklearn Modules
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Models
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


# ========== Load Data ==========
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df


# ========== Preprocessing ==========
def preprocess_data(df):
    X = df.drop(columns=["trek_total_cost(rupees)"])
    y = df["trek_total_cost(rupees)"]

    num_features = X.select_dtypes(exclude="object").columns
    cat_features = X.select_dtypes(include="object").columns

    numeric_transformer = StandardScaler()
    oh_transformer = OneHotEncoder()

    preprocessor = ColumnTransformer(
        [
            ("OneHotEncoder", oh_transformer, cat_features),
            ("StandardScaler", numeric_transformer, num_features),
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    return X_processed, y, preprocessor


# ========== Evaluation ==========
def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2 = r2_score(true, predicted)
    return mae, rmse, r2


# ========== Train and Evaluate All Models ==========
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "Lasso": Lasso(),
        "Ridge": Ridge(),
        "K-Neighbors Regressor": KNeighborsRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest Regressor": RandomForestRegressor(),
        "XGBRegressor": XGBRegressor(),
        "CatBoosting Regressor": CatBoostRegressor(verbose=False),
        "AdaBoost Regressor": AdaBoostRegressor(),
    }

    model_names = []
    r2_scores = []

    for name, model in models.items():
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Evaluate
        train_mae, train_rmse, train_r2 = evaluate_model(y_train, y_train_pred)
        test_mae, test_rmse, test_r2 = evaluate_model(y_test, y_test_pred)

        # Print results
        print(f"\n{name}")
        print("Model performance for Training set:")
        print(f"- RMSE: {train_rmse:.4f}")
        print(f"- MAE : {train_mae:.4f}")
        print(f"- R2  : {train_r2:.4f}")
        print("Model performance for Test set:")
        print(f"- RMSE: {test_rmse:.4f}")
        print(f"- MAE : {test_mae:.4f}")
        print(f"- R2  : {test_r2:.4f}")
        print("=" * 40)

        model_names.append(name)
        r2_scores.append(test_r2)

    results_df = pd.DataFrame(
        {"Model Name": model_names, "R2_Score": r2_scores}
    ).sort_values(by="R2_Score", ascending=False)

    return results_df


# ========== Single Model Accuracy ==========
def linear_regression_accuracy(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred) * 100
    print(f"\nLinear Regression Accuracy: {score:.2f}%")
    return score


# ========== Main Function ==========
def main():
    # Step 1: Load Data
    df = load_data(
        "/Users/apple/Downloads/Data_science_file/Agent8/Projects/Treking_cost_predictor/notebook/data/data.csv"
    )

    # Step 2: Preprocess
    X, y, preprocessor = preprocess_data(df)

    # Step 3: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 4: Train and Evaluate Models
    results_df = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Step 5: Print R2 Score Comparison
    print("\nModel R2 Score Comparison:")
    print(results_df)

    # Step 6: Accuracy of Linear Regression
    linear_regression_accuracy(X_train, X_test, y_train, y_test)


# ========== Entry Point ==========
if __name__ == "__main__":
    main()
"""


# __________________________GPT working___________________________

# src/components/data_transformation.py

'''import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    # Save path for the preprocessor object
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates a ColumnTransformer with pipelines for numeric and categorical columns.
        """
        try:
            numerical_columns = [
                "trek_rating",
                "Distance_km",
                "Duration_days",
                "Elevation_gain_m",
                "Max_altitude_m",
                "Avg_temp_C",
                "Group_size",
                "Permit_required",
                "Guide_cost_rupees",
                "Permit_fee_rupees",
                "Local_cost_index",
                "age",
                "zip_code",
                "emergency contact details",
                "weight",
                "height",
                "bmi",
                "accommodation_rating",
                "Hotel_Price_per_day",
                "wi-fi_facility",
                "breakfast",
                "lunch",
                "dinner",
                "pool",
                "bar",
            ]

            categorical_columns = [
                "trek_ID",
                "Company Name",
                "Trek_Location",
                "Region",
                "Country",
                "Industry",
                "Contact Email",
                "Difficulty",
                "Season",
                "Weather",
                "Accessibility",
                "Accommodation",
                "Currency",
                "Best_Season",
                "Best_Month",
                "surname",
                "address",
                "phone_no",
                "gender",
                "city",
                "state",
                "traveler_name",
                "health_check_up",
                "payment status",
                "booking_status",
                "traveller_profession",
                "payment information",
                "identification documents (for permits)",
                "Hotel_Name",
                "City",
                "trek_start_date",
                "trek_end_date",
                "trek_type",
                "Backpack",
                "Footwear",
                "waterproof/windproof jackets",
                "trekking_shoes",
                "Lighting",
                "Cash/online",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_columns),
                    ("cat", cat_pipeline, categorical_columns),
                ]
            )

            logging.info("Preprocessor pipeline constructed successfully.")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Transforms training and testing data using preprocessing pipelines,
        saves the preprocessor, and returns transformed arrays + labels.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column = "trek_total_cost(rupees)"

            # Split features and target
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            # Get and apply preprocessor
            preprocessor = self.get_data_transformer_object()

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # Save preprocessor to artifacts
            save_object(self.config.preprocessor_obj_file_path, preprocessor)
            logging.info(
                f"Saved preprocessor to: {self.config.preprocessor_obj_file_path}"
            )

            # Return transformed data + labels + path to preprocessor
            return (
                X_train_transformed,
                X_test_transformed,
                y_train,
                y_test,
                self.config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


"""if __name__ == "__main__":
    obj = DataTransformation()
    train_arr, test_arr, _ = obj.initiate_data_transformation()"""


if __name__ == "__main__":
    try:
        obj = DataTransformation()
        X_train, X_test, y_train, y_test, preprocessor_path = (
            obj.initiate_data_transformation(
                train_path="/Users/apple/Downloads/Data_science_file/Agent8/Projects/Treking_cost_predictor/src/components/artifacts/train.csv",
                test_path="/Users/apple/Downloads/Data_science_file/Agent8/Projects/Treking_cost_predictor/src/components/artifacts/test.csv",
            )
        )
        print("Data transformation successful!")
        print(f"Preprocessor saved at: {preprocessor_path}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
'''

# ___________________________________________________________old _____________OG____________________________

# src/components/data_transformation.py

# This file is to change the categorical features to numerical, Onehotencoding,Label encoding
# If confussed check Notebook


import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# importing CustomException from src folder
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# DataTransformationConfig class to store configuration for data transformation
@dataclass
class DataTransformationConfig:
    # Path to save preprocessor object as 'preprocessor.pkl' inside 'artifacts' folder
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


# DataTransformation class handles creation of preprocessing pipelines
class DataTransformation:
    def __init__(self):
        # Initializing configuration for data transformation
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function creates preprocessing pipelines for both
        numerical and categorical columns and returns a combined preprocessor object.
        """
        try:
            # Defining numerical and categorical feature names
            numerical_columns = [
                "Duration_days",
                "Elevation_gain_m",
                "Max_altitude_m",
                "Group_size",
                "Guide_cost_rupees",
                "Permit_fee_rupees",
                "age",
                "zip_code",
                "emergency contact details",
                "Hotel_Price_per_day",
            ]

            print("Numerical columns:", numerical_columns)
            categorical_columns = [
                "Trek_Location",
                "Region",
                "Country",
                "Industry",
                "Difficulty",
                "Season",
                "Weather",
                "Accessibility",
                "Accommodation",
                "Currency",
                "Best_Season",
                "Best_Month",
                "surname",
                "gender",
                "city",
                "state",
                "traveler_name",
                "health_check_up",
                "payment status",
                "booking_status",
                "traveller_profession",
                "payment information",
                "identification documents (for permits)",
                "City",
                "trek_type",
                "Backpack",
                "Footwear",
                "waterproof/windproof jackets",
                "trekking_shoes",
                "Lighting",
                "Cash/online",
            ]
            print("Categorical columns:", categorical_columns)

            # Pipeline for numerical data:
            # - Fill missing values with median
            # - Standardize numerical features
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            print(num_pipeline)
            # Pipeline for categorical data:
            # - Fill missing values with most frequent category
            # - Convert categories into one-hot encoded variables
            # - Scale encoded values (without centering)
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )
            print(cat_pipeline)
            # Logging for debugging/tracking
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info("Categorical columns encoding pipeline created successfully")
            logging.info("Numerical columns scaling pipeline created successfully")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combining both pipelines using ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            # Returning the combined preprocessor object
            return preprocessor

        except Exception as e:
            # Raising custom exception for better error tracking
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Reads training and test datasets, applies preprocessing transformations,
        saves the preprocessor object, and returns transformed arrays.
        """
        try:
            # Reading the train and test CSV files into pandas DataFrames
            train_df = pd.read_csv(train_path)
            print(train_df.head(2))
            test_df = pd.read_csv(test_path)
            print(test_df.head(2))

            logging.info("Train and test data loaded successfully")

            # Get the preprocessor object (created in get_data_transformer_object)
            preprocessing_obj = self.get_data_transformer_object()

            # Define target column and numerical columns
            target_column_name = "trek_total_cost(rupees)"
            numerical_columns = [
                "Duration_days",
                "Elevation_gain_m",
                "Max_altitude_m",
                "Group_size",
                "Guide_cost_rupees",
                "Permit_fee_rupees",
                "age",
                "zip_code",
                "emergency contact details",
                "Hotel_Price_per_day",
            ]
            print(target_column_name)
            # Separate input features (X) and target variable (y) for training data
            input_feature_train_df = train_df.drop(
                columns=[target_column_name], axis=1
            )  # Features train
            target_feature_train_df = train_df[target_column_name]  # Target train
            print(input_feature_train_df.head(2))
            print(target_feature_train_df.head(2))

            # Separate input features (X) and target variable (y) for testing data
            input_feature_test_df = test_df.drop(
                columns=[target_column_name], axis=1
            )  # Features test
            target_feature_test_df = test_df[target_column_name]  # Target test

            print(input_feature_test_df.head(2))
            print(target_feature_test_df.head(2))

            logging.info("Applying preprocessing on training and test data")

            # Fit preprocessor on training data and transform both train & test sets
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            print(input_feature_train_arr)
            print(input_feature_test_arr)
            print("Features shape:", input_feature_train_arr.shape)
            print("Features shape:", input_feature_train_arr.shape)
            print("Target shape:", np.array(target_feature_train_df).shape)

            # Combine input features and target variable back into arrays
            # train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            train_arr = np.c_[
                input_feature_train_arr.toarray(),  # convert sparse to dense
                np.array(target_feature_train_df),
            ]
            # test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            test_arr = np.c_[
                input_feature_test_arr.toarray(), np.array(target_feature_test_df)
            ]
            print(train_arr)
            print(test_arr)

            logging.info("Preprocessing complete. Saving preprocessor object.")

            # Save the fitted preprocessor object to the specified file path
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            # Return processed train & test arrays and preprocessor path
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            # Raise a custom exception for better error tracking
            raise CustomException(e, sys)
