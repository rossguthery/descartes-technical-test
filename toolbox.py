"""
This Python script contains all the classes and functions required to carry out
the Descartes Underwriting data scientist technical test, i.e. it:
- Gets the data,
- Cleans the data,
- Performs feature engineering,
- Executes select models,
- Reports the results, and
- Exports the results.
"""

# Import packages.
from typing import Callable, Tuple
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression


class Data:
    """Manages the reading in and the cleaning of the data stored at the file
    path the user passes as an argument.
    """

    def __init__(self, train: bool, file_path: str, index_col: int) -> None:
        """Initializes a Data object's attributes.

        Args:
            train (bool): Indicates if data is the train set.
            file_path (str): The path of the file where the data is stored.
            index_col (int): The column to index the final DataFrame by.
        """
        self.train: bool = train
        self.file_path: str = file_path
        self.index_col: int = index_col
        self.data: pd.DataFrame = pd.DataFrame()

    def __get_data(self) -> pd.DataFrame:
        """Returns the data stored at the file path the Data object was
        instantiated with. The output is indexed using the column specified by
        the user at instantiation.

        Returns:
            pd.DataFrame: The data stored at the file path.
        """
        return pd.read_csv(filepath_or_buffer=self.file_path, index_col=self.index_col)

    def __clean_data(self) -> None:
        """Cleans the data. This function was built for this exercise and is not
        intended for general use.
        """
        # Retitle columns so they are easier to understand.
        column_names: list = [
            "target_flag",
            "target_claim_amount",
            "num_kids_driving",
            "age",
            "num_kids_home",
            "yoj",
            "income",
            "is_single_parent",
            "home_value",
            "is_married",
            "is_female",
            "education",
            "job",
            "travel_time",
            "for_commercial_use",
            "bluebook_value",
            "tif",
            "car_type",
            "is_red_car",
            "last_claim_value",
            "claim_frequency",
            "was_revoked",
            "mvr_pts",
            "car_age",
            "is_urban",
        ]
        self.data = self.data.set_axis(labels=column_names, axis=1)

        # Drop columns that don't have a clear meaning or that we can't use.
        self.data = self.data.drop(
            columns=["target_claim_amount", "yoj", "tif", "mvr_pts"]
        )

        # Clean up string columns, convert monetary columns to the numeric type,
        # and convert columns intended to be booleans to true booleans.
        self.data = (
            self.data.assign(
                job=lambda df: df.job.replace(
                    {"z_": "", " ": "_"}, regex=True
                ).str.lower(),
                car_type=lambda df: df.car_type.replace(
                    {"z_": "", " ": "_"}, regex=True
                ).str.lower(),
                education=lambda df: df.education.replace(
                    {"z_": "", "<": "", " ": "_"}, regex=True
                ).str.lower(),
                income=lambda df: pd.to_numeric(
                    df.income.replace({"\$": "", ",": ""}, regex=True), errors="coerce",
                ),
                home_value=lambda df: pd.to_numeric(
                    df.home_value.replace({"\$": "", ",": ""}, regex=True),
                    errors="coerce",
                ),
                bluebook_value=lambda df: pd.to_numeric(
                    df.bluebook_value.replace({"\$": "", ",": ""}, regex=True),
                    errors="coerce",
                ),
                last_claim_value=lambda df: pd.to_numeric(
                    df.last_claim_value.replace({"\$": "", ",": ""}, regex=True),
                    errors="coerce",
                ),
                is_female=lambda df: pd.to_numeric(
                    df.is_female.replace({"z_F": "1", "M": "0"}, regex=True),
                    errors="coerce",
                ),
                is_red_car=lambda df: pd.to_numeric(
                    df.is_red_car.replace({"yes": "1", "no": "0"}, regex=True),
                    errors="coerce",
                ),
                was_revoked=lambda df: pd.to_numeric(
                    df.was_revoked.replace({"Yes": "1", "No": "0"}, regex=True),
                    errors="coerce",
                ),
                is_married=lambda df: pd.to_numeric(
                    df.is_married.replace({"Yes": "1", "z_No": "0"}, regex=True),
                    errors="coerce",
                ),
                is_urban=lambda df: pd.to_numeric(
                    df.is_urban.replace(
                        {"Highly Urban/ Urban": "1", "z_Highly Rural/ Rural": "0"},
                        regex=True,
                    ),
                    errors="coerce",
                ),
                is_single_parent=lambda df: pd.to_numeric(
                    df.is_single_parent.replace({"Yes": "1", "No": "0"}, regex=True),
                    errors="coerce",
                ),
                for_commercial_use=lambda df: pd.to_numeric(
                    df.for_commercial_use.replace(
                        {"Commercial": "1", "Private": "0"}, regex=True
                    ),
                    errors="coerce",
                ),
            )
        ).reset_index(drop=True)

        # If train set, drop duplicate entries and keep only positive car ages.
        if self.train:
            self.data = self.data.drop_duplicates()
            self.data = self.data[self.data.car_age >= 0]

    def deal_with_nulls(self) -> None:
        """Replaces null values in the age, job, income, car age, and home value
        columns according to the strategy explained in the Jupyter notebook.
        Also, drops the target flag column from the test set as it's only nulls.
        """
        # Replace nulls with column's mean value.
        self.data = self.data.assign(
            car_age=lambda df: df.car_age.fillna(value=np.mean(df.car_age)),
            age=lambda df: df.age.fillna(value=np.mean(df.age)),
        )

        # Drop nulls.
        self.data = self.data.dropna(subset=["job", "income", "home_value"], how="any")

        # Drop target flag column from test data.
        if not self.train:
            self.data = self.data.drop(columns=["target_flag"])

    def __call__(self) -> None:
        """Executes a Data object. It does not return the data. One can access
        the data via the data attribute of the object.
        """
        # Get data.
        self.data = self.__get_data()

        # Clean data.
        self.__clean_data()


class FeatureEngineerer:
    """Manages all feature engineering, which mainly consists of one-hot
    encoding the categorical variables, creating some interaction terms, and
    taking several log transformations.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """Initializes a FeatureEngineerer object's attributes.

        Args:
            data (pd.DataFrame): The data post cleaning.
        """
        self.data: pd.DataFrame = data

    def __one_hot_encoding(self) -> None:
        """Performs one-hot encoding on all the non-numerical categorical
        variables, which includes education, job, and car type.
        """
        # Do one-hot encoding. Columns we no longer need get automatically
        # dropped from the DataFrame.
        self.data = pd.get_dummies(
            data=self.data, columns=["education", "job", "car_type"],
        )

    def get_interaction_terms(self) -> None:
        """Adds the interaction terms mentioned in the Jupyter notebook to the
        data set the object was instantiated with.
        """
        self.data = self.data.assign(
            female_suv=lambda df: df.is_female * df.car_type_suv,
            female_red_car=lambda df: df.is_female * df.is_red_car,
            high_school_car_age=lambda df: (df.education_high_school * df.car_age),
        )

    def get_log_transforms(self) -> None:
        """Takes the log transformation of all the so-called monetary variables,
        which includes income, home value, bluebook value, and last claim value.
        """
        # Take log transform, but add one to avoid a divide-by-zero error.
        self.data = self.data.assign(
            log_income=lambda df: np.log(df.income + 1),
            log_home_value=lambda df: np.log(df.home_value + 1),
            log_bluebook_value=lambda df: np.log(df.bluebook_value + 1),
            log_last_claim_value=lambda df: np.log(df.last_claim_value + 1),
        )

        # Drop columns we no longer need.
        self.data = self.data.drop(
            columns=["income", "home_value", "bluebook_value", "last_claim_value",]
        )

    def __call__(self) -> None:
        """Executes a FeatureEngineerer object. It does not return the data. One
        can access the data via the data attribute of the object.
        """
        # Perform one-hot encoding.
        self.__one_hot_encoding()


class Modeller:
    """Manages everything to do with modelling the cleaned and feature
    engineered data. This includes, but is not limited to, computing model
    performance, plotting graphics that describe model performance, and
    executing the model itself.
    """

    def __init__(
        self,
        test: pd.DataFrame,
        train: pd.DataFrame,
        config_path: str,
        entry_point: str,
    ) -> None:
        """Initializes a Modeller object's attributes.

        Args:
            test (pd.DataFrame): The cleaned and feature engineered TEST set.
            train (pd.DataFrame): The cleaned and feature engineered TRAIN set.
            config_path (str): File path pointing to where model configs are.
            entry_point (str): String indicating which model config to use.
        """
        self.test: pd.DataFrame = test
        self.train: pd.DataFrame = train
        self.config_path: str = config_path
        self.entry_point: str = entry_point
        self.model_params: dict = {}
        self.model: Callable = Callable
        self.target: pd.Series = pd.Series(dtype=int)
        self.results: pd.DataFrame = pd.DataFrame()

    def __get_target(self) -> None:
        """Splits the target variable from the rest of the explanatory train
        variables.
        """
        # Separate the target variable.
        self.target = self.train.target_flag

        # Drop the target variable.
        self.train = self.train.drop(columns=["target_flag"])

    def __parse_model_params(self) -> dict:
        """Uses the config path and entry point defined at object creation to
        parse a YAML file. Doing so generates a dictionary of model params to
        use during modelling.

        Returns:
            dict: A dictionary of model params.
        """
        # Load the model configs YAML file.
        with open(self.config_path) as file:
            model_params: dict = yaml.load(file, Loader=yaml.FullLoader)

        # Set model params attribute to the set of parameters of interest.
        self.model_params = model_params[self.entry_point]

    def __scale_features(self) -> None:
        """Performs standard scaling on all the data in the test and train sets.
        """
        # Instantiate a StandardScaler object.
        scaler = StandardScaler()

        # Perform scaling on the test and train sets.
        self.test = pd.DataFrame(
            scaler.fit_transform(X=self.test), columns=self.test.columns
        )
        self.train = pd.DataFrame(
            scaler.fit_transform(X=self.train), columns=self.train.columns
        )

    def __logistic_regression(self) -> None:
        """Instantiates a regularized logistic regression model in the model
        attribute.
        """
        self.model = LogisticRegression(
            C=self.model_params["C"],
            dual=self.model_params["dual"],
            solver=self.model_params["solver"],
            penalty=self.model_params["penalty"],
            l1_ratio=self.model_params["l1_ratio"],
            max_iter=self.model_params["max_iter"],
            random_state=self.model_params["random_state"],
        )

    def __svc(self) -> None:
        """Instantiates a regularized support vector classifier model in the
        model attribute.
        """
        self.model = SVC(
            C=self.model_params["C"],
            kernel=self.model_params["kernel"],
            max_iter=self.model_params["max_iter"],
        )

    def __catboost(self) -> None:
        """Instantiates a regularized catboost classifier model in the model
        attribute.
        """
        self.model = CatBoostClassifier(
            depth=self.model_params["depth"],
            verbose=self.model_params["verbose"],
            iterations=self.model_params["iterations"],
            eval_metric=self.model_params["metric"],
            l2_leaf_reg=self.model_params["l2_leaf_reg"],
            learning_rate=self.model_params["learning_rate"],
            random_strength=self.model_params["random_strength"],
        )

    def __execute_cross_validation(self) -> Tuple[Callable, float]:
        """Carries out cross-validation according to the user's specifications.

        Returns:
            Tuple[Callable, float]: The best model according to the validation
                                    score as well as said score.
        """
        # Perform cross-validation according to the user's specifications.
        cross_val_output: dict = cross_validate(
            estimator=self.model,
            X=self.train,
            y=self.target,
            cv=self.model_params["cv_iter"],
            scoring=self.model_params["metric"].lower(),
            return_estimator=True,
        )

        # Locate the index of the best performing model.
        idx_best_model: int = list(cross_val_output["test_score"]).index(
            max(cross_val_output["test_score"])
        )

        # Return the best performing model.
        return (
            cross_val_output["estimator"][idx_best_model],
            max(cross_val_output["test_score"]),
        )

    def __plot_feature_importances(self, coefficients: np.ndarray) -> None:
        """Plots the importance of each feature used to fit and predict a model.
        Saves the plot in the plots folder under the run number.

        Args:
            coefficients (np.ndarray): The importance of each feature.
        """
        # Create a DataFrame of the features and their respective importances.
        df_features: pd.DataFrame = pd.DataFrame.from_dict(
            {"feature_names": list(self.train.columns), "coefficients": list(coefficients)}
        )

        # Sort the DataFrame in order of decreasing feature importance.
        df_features.sort_values(by=["coefficients"], ascending=False, inplace=True)

        # Plot Searborn bar chart.
        sns.barplot(x=df_features.coefficients, y=df_features.feature_names)

        # Add chart labels.
        plt.title(f"{self.entry_point.title()} Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Name")

        # Save plot in appropriate folder.
        plt.savefig(f"plots/{self.entry_point}.png")

    def __save_predictions(self, preds: np.ndarray) -> None:
        """Saves the predictions made on the test set in the format proposed on
        the original Kaggle competition's website, i.e.:

        index, target
        2,0
        5,0
        6,1
        ...

        Args:
            preds (np.ndarray): The predictions made using the test set.
        """
        # Make a DataFrame using the predictions.
        df_preds: pd.DataFrame = pd.DataFrame.from_records(
            {"index": list(self.test.index), "target": list(preds)}, index="index",
        )

        # Export CSV of predictions to the appropriate folder.
        df_preds.to_csv(f"predictions/{self.entry_point}.csv")

    def __execute_unique_iteration(self) -> None:
        """Executes a unique modelling iteration, which consists of:
        - Carrying out cross-validation,
        - Fitting the best model on the entire train set,
        - Making predictions on the train set,
        - Scoring these predictions with the F1 score,
        - Loggin the model's performance,
        - Plotting the model's feature importances, and
        - Saving the model's predictions on the test set.
        """
        # Carry out cross-validation.
        best_model: Tuple[Callable, float] = self.__execute_cross_validation()

        # Fit the best model on the entire train set.
        if self.model_params["model"] == "catboost":
            best_model[0].fit(Pool(self.train, self.target))
        else:
            best_model[0].fit(self.train, self.target)

        # Make predictions on train set.
        train_preds: pd.Series = best_model[0].predict(self.train)

        # Score train set performance.
        train_score: float = f1_score(y_true=self.target, y_pred=train_preds)

        # Log score and run information.
        self.results = pd.DataFrame.from_dict(
            {
                "run_number": [self.entry_point.split("_")[-1]],
                "model": [self.model_params["model"]],
                "metric": [self.model_params["metric"].lower()],
                "best_val_score": [best_model[1]],
                "train_score": [train_score],
            }
        )

        # Plot feature importances.
        if self.model_params["model"] == "catboost":
            self.__plot_feature_importances(
                coefficients=best_model[0].get_feature_importance()
            )
        elif self.model_params["model"] == "logistic_regression":
            self.__plot_feature_importances(coefficients=best_model[0].coef_[0])
        elif (
            self.model_params["model"] == "svc"
            and self.model_params["kernel"] == "linear"
        ):
            self.__plot_feature_importances(coefficients=best_model[0].coef_[0])

        # Save predictions on test set.
        self.__save_predictions(preds=best_model[0].predict(self.test))

    def __call__(self) -> None:
        """Executes a Modeller object. It does not return the results. One can
        access the results via the results attribute of the Modeller object.
        """
        # Get target variable.
        self.__get_target()

        # Parse model config pased by the user.
        self.__parse_model_params()

        # Execute model specified by the user. Inform the user if they haven't
        # specified one of the models below.
        if self.model_params["model"] == "logistic_regression":
            self.__scale_features()
            self.__logistic_regression()
            self.__execute_unique_iteration()
        elif self.model_params["model"] == "svc":
            self.__scale_features()
            self.__svc()
            self.__execute_unique_iteration()
        elif self.model_params["model"] == "catboost":
            self.__catboost()
            self.__execute_unique_iteration()
        else:
            print(
                f"Recieved {self.model_params['model']}. " +
                "Expected logistic_regression, svc, or catboost." +
                "Please retry."
            )
