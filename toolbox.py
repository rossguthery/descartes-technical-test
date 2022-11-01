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
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression


class Data():
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
        return pd.read_csv(
            filepath_or_buffer=self.file_path, index_col=self.index_col
        )

    def __clean_data(self) -> None:
        """Cleans the data. This function was built for this test and is not
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
                job = lambda df: df.job.replace(
                    {"z_": "", " ": "_"}, regex=True
                ).str.lower(),
                car_type = lambda df: df.car_type.replace(
                    {"z_": "", " ": "_"}, regex=True
                ).str.lower(),
                education = lambda df: df.education.replace(
                    {"z_": "", "<": "", " ": "_"}, regex=True
                ).str.lower(),
                income = lambda df: pd.to_numeric(
                    df.income.replace({"\$": "", ",": ""}, regex=True),
                    errors="coerce",
                ),
                home_value = lambda df: pd.to_numeric(
                    df.home_value.replace({"\$": "", ",": ""}, regex=True),
                    errors="coerce",
                ),
                bluebook_value = lambda df: pd.to_numeric(
                    df.bluebook_value.replace({"\$": "", ",": ""}, regex=True),
                    errors="coerce",
                ),
                last_claim_value = lambda df: pd.to_numeric(
                    df.last_claim_value.replace({"\$": "", ",": ""}, regex=True),
                    errors="coerce",
                ),
                is_female = lambda df: pd.to_numeric(
                    df.is_female.replace({"z_F": "1", "M": "0"}, regex=True),
                    errors="coerce",
                ),
                is_red_car = lambda df: pd.to_numeric(
                    df.is_red_car.replace({"yes": "1", "no": "0"}, regex=True),
                    errors="coerce",
                ),
                was_revoked = lambda df: pd.to_numeric(
                    df.was_revoked.replace({"Yes": "1", "No": "0"}, regex=True),
                    errors="coerce",
                ),
                is_married = lambda df: pd.to_numeric(
                    df.is_married.replace({"Yes": "1", "z_No": "0"}, regex=True),
                    errors="coerce",
                ),
                is_urban = lambda df: pd.to_numeric(
                    df.is_urban.replace(
                        {"Highly Urban/ Urban": "1", "z_Highly Rural/ Rural": "0"},
                        regex=True,
                    ),
                    errors="coerce",
                ),
                is_single_parent = lambda df: pd.to_numeric(
                    df.is_single_parent.replace({"Yes": "1", "No": "0"}, regex=True),
                    errors="coerce",
                ),
                for_commercial_use = lambda df: pd.to_numeric(
                    df.for_commercial_use.replace({"Commercial": "1", "Private": "0"},regex=True),
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
        self.data = (
            self.data.assign(
                car_age = lambda df: df.car_age.fillna(
                    value=np.mean(df.car_age)
                ),
                age = lambda df: df.age.fillna(value=np.mean(df.age))
            )
        )

        # Drop nulls.
        self.data = self.data.dropna(
            subset=["job", "income", "home_value"], how="any"
        )

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


class FeatureEngineerer():
    """Manages all feature engineering, which mainly consists of one-hot
    encoding the categorical variables, creating some interaction terms, and
    scaling. Other transformations are potentially part of this class as well.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """Initializes a FeatureEngineerer object's attributes.

        Args:
            data (pd.DataFrame): The data post cleaning.
        """
        self.data: pd.DataFrame = data

    def get_interaction_terms(self) -> None:
        """Adds the interaction terms mentioned in the Jupyter notebook to the
        data set the object was instantiated with.
        """
        # Add the interaction terms.
        self.data = (
            self.data.assign(
                female_suv = lambda df: df.is_female * df.car_type_suv,
                female_red_car = lambda df: df.is_female * df.is_red_car,
                high_school_car_age = lambda df: (
                    df.education_high_school * df.car_age
                ),
            )
        )

    def __one_hot_encoding(self) -> None:
        """Performs one-hot encoding on all the non-numerical categorical
        variables, which includes education, job, and car type.
        """
        # Do one-hot encoding. Columns we no longer need get automatically
        # dropped from the DataFrame.
        self.data = pd.get_dummies(
            data=self.data,
            columns=["education", "job", "car_type"],
        )

    def __scale_features(self) -> None:
        """Performs min-max scaling on all the data.
        """
        # Instantiate a MinMaxScaler object.
        scaler = MinMaxScaler()

        # Perform scaling.
        self.data = pd.DataFrame(
            scaler.fit_transform(X=self.data), columns=self.data.columns
        )

    def __call__(self) -> None:
        """Executes a FeatureEngineerer object. It does not return the data. One
        can access the data via the data attribute of the object.
        """
        # Perform one-hot encoding.
        self.__one_hot_encoding()

        # Perform min-max scaling.
        self.__scale_features()


class Modeller():
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
            entry_point (str): String indicating which model params to use.
        """
        self.test: pd.DataFrame = test
        self.train: pd.DataFrame = train
        self.config_path: str = config_path
        self.entry_point: str = entry_point
        self.model_params: dict = {}
        self.results: pd.DataFrame = pd.DataFrame()

    def __get_target(self) -> pd.Series:
        """Splits the target variable from the rest of the explanatory train
        variables.

        Returns:
            pd.Series: The target variable.
        """
        # Separate the target variable.
        target: pd.Series = self.train.target_flag

        # Drop the target variable.
        self.train = self.train.drop(columns=["target_flag"])

        # Return the target.
        return target

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
        df_features.sort_values(
            by=["coefficients"], ascending=False, inplace=True
        )

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
            {"index": list(self.test.index), "target": list(preds)},
            index="index",
        )

        # Export CSV of predictions to the appropriate folder.
        df_preds.to_csv(f"predictions/{self.entry_point}.csv")

    def __logistic_regression(self, target: pd.Series) -> None:
        """Executes a regularized logistic regression model. Model performance
        is recorded in the results attribute.

        Args:
            target (pd.Series): The train set target variable.
        """
        # Create logistic regression object.
        log_reg: LogisticRegression = LogisticRegression(
            C=self.model_params["C"],
            dual=self.model_params["dual"],
            solver=self.model_params["solver"],
            penalty=self.model_params["penalty"],
            max_iter=self.model_params["max_iter"],
            random_state=self.model_params["random_state"],
        )

        # Fit the model.
        log_reg.fit(X=self.train, y=target)

        # Make predictions on train set.
        train_preds: pd.Series = log_reg.predict(X=self.train)

        # Score train set performance.
        train_score: float = f1_score(y_true=target, y_pred=train_preds)

        # Log score and run information.
        self.results = pd.DataFrame.from_dict(
            {
                "run_number": [self.entry_point.split("_")[-1]],
                "model": [self.model_params["model"]],
                "metric": [self.model_params["metric"]],
                "train_score": [train_score],
            }
        )

        # Plot feature importance.
        self.__plot_feature_importances(coefficients=log_reg.coef_[0])

        # Save predictions on test set.
        self.__save_predictions(preds=log_reg.predict(X=self.test))

    def __linear_svc(self, target: pd.Series) -> None:
        """Executes a regularized support vector linear classifier model. Model
        performance is in the results attribute.

        Args:
            target (pd.Series): The train set target variable.
        """
        # Create a support vector linear classifier object.
        linear_svc: LinearSVC = LinearSVC(
            C=self.model_params["C"],
            loss=self.model_params["loss"],
            dual=self.model_params["dual"],
            penalty=self.model_params["penalty"],
            max_iter=self.model_params["max_iter"],
        )

        # Fit the model.
        linear_svc.fit(X=self.train, y=target)

        # Make predictions on train set.
        train_preds: pd.Series = linear_svc.predict(X=self.train)

        # Score train set performance.
        train_score: float = f1_score(y_true=target, y_pred=train_preds)

        # Log score and run information.
        self.results = pd.DataFrame.from_dict(
            {
                "run_number": [self.entry_point.split("_")[-1]],
                "model": [self.model_params["model"]],
                "metric": [self.model_params["metric"]],
                "train_score": [train_score],
            }
        )

        # Plot feature importance.
        self.__plot_feature_importances(coefficients=linear_svc.coef_[0])

        # Save predictions on test set.
        self.__save_predictions(preds=linear_svc.predict(X=self.test))

    def __call__(self) -> None:
        """Executes a Modeller object. It does not return the results. One can
        access the results via the results attribute of the Modeller object.
        """
        # Get target variable.
        target: pd.Series = self.__get_target()

        # Parse model config pased by the user.
        self.__parse_model_params()

        # Execute model specified by the user.
        if self.model_params["model"] == "logistic_regression":
            self.__logistic_regression(target=target)
        if self.model_params["model"] == "linear_svc":
            self.__linear_svc(target=target)
