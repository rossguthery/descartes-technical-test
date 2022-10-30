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
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegressionCV


class Data():
    """Manages the reading in and the cleaning of the data stored at the file
    path the user passes as an argument.
    """

    def __init__(self, train: bool, file_path: str, index_col: int) -> None:
        """Initializes a Data object's attributes.

        Args:
            train (bool): Indicates if data is training data.
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
        # and convert columns intended to be booleans into true booleans. Also,
        # enfore certain data types.
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
                        regex=True
                    ),
                    errors="coerce",
                ),
                is_single_parent = lambda df: pd.to_numeric(
                    df.is_single_parent.replace({"Yes": "1", "No": "0"}, regex=True),
                    errors="coerce",
                ),
                for_commercial_use = lambda df: pd.to_numeric(
                    df.for_commercial_use.replace({"Commercial": "1", "Private": "0"}, regex=True),
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
            self.data = self.data.drop("target_flag", axis=1)

    def __call__(self) -> None:
        """Executes a Data object. It does not return the data. One can access
        the data via the "data" attribute of the Data object.
        """
        # Get data.
        self.data = self.__get_data()

        # Clean data.
        self.__clean_data()


class FeatureEngineerer():
    """Manages all feature engineering, which mainly consists of one-hot
    encoding the categorical variables, taking the log transform of so-called
    monetary variables, and scaling.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """Initializes a FeatureEngineerer object's attributes.

        Args:
            data (pd.DataFrame): The data post-cleaning.
        """
        self.data: pd.DataFrame = data

    def __log_transform(self) -> None:
        """Takes the log transformation of all the so-called monetary variables,
        which includes income, home value, bluebook value, and last claim value.
        """
        # Take log transform, but add one to avoid divide-by-zero error.
        self.data = (
            self.data.assign(
                log_income = lambda df: np.log(df.income+1),
                log_home_value = lambda df: np.log(df.home_value+1),
                log_bluebook_value = lambda df: np.log(df.bluebook_value+1),
                log_last_claim_value = lambda df: np.log(df.last_claim_value+1),
            )
        )

    def __one_hot_encoding(self) -> None:
        """Performs one-hot encoding on all the categorical variables, which
        includes education, job, and car type.
        """
        # Do one-hot encoding and drop columns we no longer need.
        self.data = pd.get_dummies(
            data=self.data,
            columns=["education", "job", "car_type"],
        )

    def __scale_features(self) -> None:
        """Performs min-max scaling on all the data.
        """
        # Instantiate MinMaxScaler object.
        scaler = MinMaxScaler()

        # Perform scaling.
        scaler.fit_transform(self.data)

    def __call__(self) -> None:
        """Executes a FeatureEngineerer object. It does not return the data. One
        can access the data via the "data" attribute of the FeatureEngineerer
        object.
        """
        # Take select log transforms.
        self.__log_transform()

        # Perform one-hot encodings.
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
            test (pd.DataFrame): The cleaned and feature engineered test data.
            train (pd.DataFrame): The cleaned and feature engineered train data.
            config_path (str): File path pointing to where model configs are.
            entry_point (str): String indicating which model params to use.
        """
        self.test: pd.DataFrame = test
        self.train: pd.DataFrame = train
        self.config_path: str = config_path
        self.entry_point: str = entry_point
        self.model_params: dict = {}
        self.results: pd.DataFrame = pd.DataFrame()

    def __get_target(self, data: pd.DataFrame) -> pd.DataFrame:
        """Splits the target variable from the rest of the explanatory
        variables.

        Args:
            data (pd.DataFrame): Either the train or test data.

        Returns:
            pd.DataFrame: The target variable.
        """
        # Separate the target variable.
        target: pd.Series = data.target_flag

        # Drop the target variable.
        data = data.drop("target_flag", axis=1, inplace=True)

        # Return the target.
        return target

    def __parse_model_params(self) -> dict:
        """Uses the config path and entry point defined at object creation to
        parse a YAML file. Doing so generates a dictionary of model params to
        use during modelling.

        Returns:
            dict: A dictionary of model params.
        """
        # Load model configs YAML file.
        with open(self.config_path) as file:
            model_params: dict = yaml.load(file, Loader=yaml.FullLoader)
        
        # Set object attribute to the set of params of interest.
        self.model_params = model_params[self.entry_point]

    def __plot_feature_importance(self, coefficients: np.ndarray) -> None:
        """Plots the importance of each feature used to fit and predict a model.
        Saves the plot in the plots folder under the run number.

        Args:
            coefficients (np.ndarray): The importance of each feature.
        """
        # Create a DataFrame of the features and their respective importances.
        df_features: pd.DataFrame = pd.DataFrame.from_dict(
            {"feature_names": list(self.train.columns), "coefficients": list(coefficients)}
        )

        # Sort the DataFrame in order decreasing feature importance.
        df_features.sort_values(
            by=["coefficients"], ascending=False, inplace=True
        )

        # Define size of bar plot.
        plt.figure(figsize=(10, 8))

        # Plot Searborn bar chart.
        sns.barplot(x=df_features.coefficients, y=df_features.feature_names)

        # Add chart labels.
        plt.title(f"{self.entry_point} Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Name")

        # Save plot in appropriate folder without displaying it.
        plt.savefig(f"plots/{self.entry_point}.png")

    def __logistic_regression(self, train_target: pd.Series) -> None:
        """Executes a cross-validated and regularized logistic regression model.
        Model performance is recorded in the "results" attribute.

        Args:
            train_target (pd.Series): The train set target variable.
        """
        # Create logistic regression object.
        log_reg_clf: LogisticRegressionCV = LogisticRegressionCV(
            Cs=self.model_params["Cs"],
            cv=self.model_params["cv"],
            solver=self.model_params["solver"],
            penalty=self.model_params["penalty"],
            scoring=self.model_params["scoring"],
            max_iter=self.model_params["max_iter"],
        )

        # Fit the model.
        log_reg_clf.fit(X=self.train, y=train_target)

        # Score train set performance.
        train_score: float = log_reg_clf.score(X=self.train, y=train_target)

        # Log scores and run information.
        self.results = pd.DataFrame.from_dict(
            {
                "run_number": [self.entry_point.split("_")[-1]],
                "model": [self.model_params["model"]],
                "metric": [self.model_params["scoring"]],
                "train_score": [train_score],
            }
        )

        # Plot feature importance.
        self.__plot_feature_importance(coefficients=log_reg_clf.coef_[0])

    def __call__(self) -> None:
        """Executes a Modeller object. It does not return the results. One can
        access the results via the "results" attribute of the Modeller object.
        """
        # Get target variables.
        train_target: pd.Series = self.__get_target(data=self.train)

        # Parse model config pased by the user.
        self.__parse_model_params()

        # Execute model specified by the user.
        if self.model_params["model"] == "Logistic Regression":
            self.__logistic_regression(train_target=train_target)
