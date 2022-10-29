"""
This Python script contains all the classes and functions required to carry out
the Descartes Underwriting data scientist technical test, i.e. it:
- Gets the data,
- Cleans the data,
- Executes select models,
- Performs feature engineering,
- Reports the results, and
- Exports the results.
"""

# Import packages.
import pandas as pd


class Data():
    """Manages the reading in and the cleaning of the data stored at the file
    path the user passes as an argument.
    """

    def __init__(self, file_path: str, index_col: int) -> None:
        """Initializes a Data object's attributes.

        Args:
            file_path (str): The path of the file where the data is stored.
            index_col (int): The column to index the final DataFrame by.
        """
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

    def __clean_data(self) -> pd.DataFrame:
        """Takes in raw data and cleans it. This function was built for this
        test's data. It is not intended for general use.

        Returns:
            pd.DataFrame: The clean data.
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

        # Clean up string columns, convert monetary columns to numeric ones, and
        # convert columns meant to be dummy columns into true dummy columns.
        self.data = (
            self.data.assign(
                education = lambda df: df.education.replace(
                    {"z_": "", "<": ""}, regex=True
                ),
                job = lambda df: df.job.replace({"z_": ""}, regex=True),
                car_type = lambda df: df.car_type.replace({"z_": ""}, regex=True),
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
            )
        )

    def __call__(self):
        """Executes a Data object. It does not return clean data. One can access
        the clean data via the "data" attribute of the Data object.
        """
        # Get raw data.
        self.data: pd.DataFrame = self.__get_data()

        # Clean raw data.
        self.__clean_data()