import os
from collections import OrderedDict
from collections.abc import Iterable
from datetime import datetime, timedelta

try:
    from netCDF4 import Dataset, Variable, chartostring
except ImportError:
    raise ImportError("NetCDF4 is required when using NetCDF/NetCDFMixin")

try:
    from cftime import num2pydate as num2date
except ImportError:
    from cftime import num2date

import numpy as np


def _encode_character_array(value: str, width: int, encoding: str = "utf-8") -> np.ndarray:
    encoded_value = value.encode(encoding)
    if len(encoded_value) > width:
        raise ValueError(f"Value '{value}' does not fit in a character dimension of size {width}.")

    characters = np.zeros(width, dtype="S1")
    characters[: len(encoded_value)] = np.frombuffer(encoded_value, dtype="S1")
    return characters


class Stations:
    def __init__(self, dataset: Dataset, station_variable: Variable):
        self.__station_variable = station_variable

        station_dimension = station_variable.dimensions[0]

        self.__attribute_variables = {}
        for variable_name in dataset.variables:
            variable = dataset.variables[variable_name]
            is_scalar_attribute = variable.dimensions == (station_dimension,)
            is_character_attribute = (
                len(variable.dimensions) == 2
                and variable.dimensions[0] == station_dimension
                and getattr(variable.dtype, "kind", None) in ("S", "U")
            )
            if variable != station_variable and (is_scalar_attribute or is_character_attribute):
                self.__attribute_variables[variable_name] = variable

        self.__attributes = OrderedDict()
        for i in range(station_variable.shape[0]):
            id = str(chartostring(station_variable[i]))

            values = {}
            for variable_name, variable in self.__attribute_variables.items():
                value = variable[i]
                if len(variable.dimensions) == 2 and not isinstance(value, str):
                    if np.ndim(value) == 0 and getattr(value.dtype, "kind", None) in ("O", "U"):
                        value = str(value.item())
                    else:
                        value = str(chartostring(value))
                values[variable_name] = value

            self.__attributes[id] = values

    @property
    def station_ids(self) -> Iterable:
        """
        :return: An ordered iterable of the station ids (location ids) for which
            station data is available.

        """
        return self.__attributes.keys()

    @property
    def attributes(self) -> OrderedDict:
        """
        :return: An OrderedDict containing dicts containing the values for all
            station attributes of the input dataset. Attributes can be scalar
            variables indexed by station or character arrays whose first dimension
            is the station dimension.
        """
        return self.__attributes

    @property
    def attribute_variables(self) -> dict:
        """
        :return: A dict containing the station attribute variables of the input dataset.
        """
        return self.__attribute_variables


class ImportDataset:
    """
    A class used to open and import the data from a NetCDF file.
    Uses the NetCDF4 library. Contains various methods for reading the data in the file.
    """

    def __init__(self, folder: str, basename: str):
        """
        :param folder:    Folder the file is located in.
        :param basename:  Basename of the file, extension ".nc" will be appended to this
        """

        self.__ensemble_size = 1

        # Load the content of a NetCDF file into a Dataset.
        self.__filename = os.path.join(folder, basename + ".nc")
        self.__dataset = Dataset(self.__filename)

        # Find the number of ensemble members and the time and station id variables
        self.__time_variable = self.__find_time_variable()
        if self.__time_variable is None:
            raise Exception(
                "No time variable found in file " + self.__filename + ". "
                "Please ensure the file contains a time variable with standard_name "
                '"time" and axis "T".'
            )

        self.__ensemble_member_variable = self.__find_ensemble_member_variable()
        if self.__ensemble_member_variable:
            self.__ensemble_size = self.__dataset.dimensions["realization"].size

        self.__station_variable = self.__find_station_variable()
        if self.__station_variable is None:
            raise Exception(
                "No station variable found in file " + self.__filename + ". "
                'Please ensure the file contains a variable with cf_role "timeseries_id".'
            )

    def __str__(self):
        return self.__filename

    def __find_time_variable(self) -> Variable | None:
        """
        Find the variable containing the times in the given Dataset.

        :param dataset: The Dataset to be searched.
        :return: a netCDF4.Variable object of the time variable (or None if none found)
        """
        for variable in self.__dataset.variables.values():
            if (
                "standard_name" in variable.ncattrs()
                and "axis" in variable.ncattrs()
                and variable.standard_name == "time"
                and variable.axis == "T"
            ):
                return variable

        return None

    def __find_ensemble_member_variable(self) -> Variable | None:
        """
        Find the variable containing the ensemble member index in the given Dataset.

        :param dataset: The Dataset to be searched.
        :return: a netCDF4.Variable object of the ensemble member index variable (or None
                 if none found)
        """
        for variable in self.__dataset.variables.values():
            if "standard_name" in variable.ncattrs() and variable.standard_name == "realization":
                return variable

        return None

    def __find_station_variable(self) -> Variable | None:
        """
        Find the variable containing station id's  (location id's) in the given Dataset.

        :param dataset: The Dataset to be searched.
        :return: a netCDF4.Variable object of the station id variable (or None if none found)
        """
        for variable in self.__dataset.variables.values():
            if "cf_role" in variable.ncattrs() and variable.cf_role == "timeseries_id":
                return variable

        return None

    def read_import_times(self) -> np.ndarray:
        """
        Reads the import times in the time variable of the dataset.

        :param time_variable: The time variable containing input times
        :return: an array containing the input times as datetime objects
        """
        time_values = self.__time_variable[:]
        time_unit = self.__time_variable.units
        try:
            time_calendar = self.__time_variable.calendar
        except AttributeError:
            time_calendar = "gregorian"

        return num2date(time_values, units=time_unit, calendar=time_calendar)

    def read_station_data(self) -> Stations:
        return Stations(self.__dataset, self.__station_variable)

    def find_timeseries_variables(self) -> list[str]:
        """
        Find the keys of all 2-D or 3-D variables with dimensions {station, time} or {station, time,
        realization} where station is the dimension of the station_variable, time the dimension of
        the time_variable and realization the dimension for ensemble_member_index.

        :param dataset:           The Dataset to be searched.
        :param station_variable:  The station id variable.
        :param time_variable:     The time variable.
        :return: a list of strings containing all keys found.
        """
        station_dim = self.__station_variable.dimensions[0]
        time_dim = self.__time_variable.dimensions[0]
        if self.__ensemble_member_variable is not None:
            ensemble_dim = self.__ensemble_member_variable.dimensions[0]
            expected_dims = [
                (time_dim, station_dim, ensemble_dim),
                (time_dim, ensemble_dim, station_dim),
                (station_dim, time_dim, ensemble_dim),
                (station_dim, ensemble_dim, time_dim),
                (ensemble_dim, time_dim, station_dim),
                (ensemble_dim, station_dim, time_dim),
            ] + [(station_dim, time_dim), (time_dim, station_dim)]
        else:
            expected_dims = [(station_dim, time_dim), (time_dim, station_dim)]

        timeseries_variables = []
        for var_key, variable in self.__dataset.variables.items():
            if variable.dimensions in expected_dims:
                timeseries_variables.append(var_key)

        return timeseries_variables

    def read_timeseries_values(
        self, station_index: int, variable_name: str, ensemble_member: int = 0
    ) -> np.ndarray:
        """
        Reads the specified timeseries from the input file.

        :param station_index: The index of the station for which the values should be read
        :param variable_name: The name of the variable for which the values should be read
        :return: an array of values
        """

        station_dim = self.__station_variable.dimensions[0]
        timeseries_variable = self.__dataset.variables[variable_name]

        # possibly useful for in a debugger mode
        # assert set(timeseries_variable.dimensions)==set(('time', 'station')) \
        #        or set(timeseries_variable.dimensions)==set(('time', 'station', 'realization'))

        if (
            self.__ensemble_member_variable is not None
            and "realization" in timeseries_variable.dimensions
        ):
            ensemble_member_dim = self.__ensemble_member_variable.dimensions[0]
            for i in range(3):
                if timeseries_variable.dimensions[i] == station_dim:
                    station_arg_Index = i
                elif timeseries_variable.dimensions[i] == ensemble_member_dim:
                    ensemble_arg_Index = i
            time_arg_Index = set(range(3)) - {station_arg_Index, ensemble_arg_Index}
            time_arg_Index = time_arg_Index.pop()
            argument = [None] * 3
            argument[station_arg_Index] = station_index
            argument[ensemble_arg_Index] = ensemble_member
            argument[time_arg_Index] = slice(None)
            values = timeseries_variable[tuple(argument)]
        else:
            if timeseries_variable.dimensions[0] == station_dim:
                values = timeseries_variable[station_index, :]
            else:
                values = timeseries_variable[:, station_index]

        # NetCDF4 reads the values as a numpy masked array,
        # convert to a normal array with nan where mask == True
        return np.ma.filled(values, np.nan)

    def variable_dimensions(self, variable):
        return self.__dataset.variables[variable].dimensions

    @property
    def time_variable(self):
        return self.__time_variable

    @property
    def station_variable(self):
        return self.__station_variable

    @property
    def ensemble_member_variable(self):
        return self.__ensemble_member_variable

    @property
    def ensemble_size(self):
        """
        Ensemble size.
        """
        return self.__ensemble_size


class ExportDataset:
    """
    A class used to write data to a NetCDF file. Creates a new file or overwrites an old file. The
    file metadata will be written upon initialization. Data such as times, station data and
    timeseries data should be presented to the ExportDataset through the various methods. When all
    data has been written, the close method must be called to flush the changes from local memory
    to the actual file on disk.
    """

    def __init__(self, folder: str, basename: str):
        """
        :param folder:   Folder the file will be located in.
        :param basename: Basename of the file, extension ".nc" will be appended to this
        """
        # Create the file and open a Dataset to access it
        self.__filename = os.path.join(folder, basename + ".nc")
        # use same write format as FEWS
        self.__dataset = Dataset(self.__filename, mode="w", format="NETCDF3_CLASSIC")

        # write metadata to the file
        self.__dataset.title = "RTC-Tools Output Data"
        self.__dataset.institution = "Deltares"
        self.__dataset.source = "RTC-Tools"
        self.__dataset.history = f"Generated on {datetime.now()}"
        self.__dataset.Conventions = "CF-1.6"
        self.__dataset.featureType = "timeseries"

        # dimensions are created when writing times and station data, must be created before
        # writing variables
        self.__time_dim = None
        self.__station_dim = None
        self.__station_id_to_index_mapping = None
        self.__ensemble_member_index_dim = None

        self.__timeseries_variables = {}

    def __str__(self):
        return self.__filename

    def write_times(self, times: np.ndarray, forecast_time: float, forecast_date: datetime) -> None:
        """
        Writes a time variable to the given dataset.

        :param dataset:        The NetCDF4.Dataset object that the times will be written to
            (must have write permission)
        :param times:          The times that are to be written in seconds.
        :param forecast_time:  The forecast time in seconds corresponding to the forecast date
        :param forecast_date:  The datetime corresponding with time in seconds at the forecast
                               index.
        """

        # in a NetCDF file times are written with respect to a reference date
        # the written values for the times may never be negative, so use the earliest time as the
        # reference date
        reference_date = forecast_date
        minimum_time = np.min(times)
        if minimum_time < 0:
            times = times - minimum_time
            reference_date = reference_date - timedelta(seconds=forecast_time - minimum_time)

        self.__time_dim = self.__dataset.createDimension("time", None)

        time_var = self.__dataset.createVariable("time", "f8", ("time",))
        time_var.standard_name = "time"
        time_var.units = f"seconds since {reference_date}"
        time_var.axis = "T"
        time_var[:] = times

    def write_ensemble_data(self, ensemble_size):
        if ensemble_size > 1:
            self.__ensemble_member_dim = self.__dataset.createDimension(
                "realization", ensemble_size
            )
            ensemble_member_var = self.__dataset.createVariable(
                "realization", "i", ("realization",)
            )
            ensemble_member_var.standard_name = "realization"
            ensemble_member_var.long_name = "Index of an ensemble member within an ensemble"
            ensemble_member_var.units = 1

    def write_station_data(self, stations: Stations, output_station_ids: list[str]) -> None:
        """
        Writes the station ids and additional station information to the given dataset.

        :param stations:           The stations data read from the input file.
        :param output_station_ids: The set of station ids for which output will be written. Must be
            unique.
        """
        assert len(set(output_station_ids)) == len(output_station_ids)

        self.__station_dim = self.__dataset.createDimension("station", len(output_station_ids))

        # first write the ids
        attribute_dimension_sizes = {}
        for attr_var in stations.attribute_variables.values():
            for dimension_index, dimension_name in enumerate(attr_var.dimensions[1:], start=1):
                dimension_size = attr_var.shape[dimension_index]
                previous_size = attribute_dimension_sizes.setdefault(dimension_name, dimension_size)
                if previous_size != dimension_size:
                    raise ValueError(
                        f"Dimension '{dimension_name}' has inconsistent sizes in station metadata."
                    )

        max_id_length = max(len(id.encode("utf-8")) for id in output_station_ids)
        station_id_width = max(
            max_id_length, attribute_dimension_sizes.get("char_leng_id", max_id_length)
        )
        self.__dataset.createDimension("char_leng_id", station_id_width)
        station_id_var = self.__dataset.createVariable(
            "station_id", "c", ("station", "char_leng_id")
        )
        station_id_var.long_name = "station identification code"
        station_id_var.cf_role = "timeseries_id"

        # we must store the index we use for each station id, to be able to write the data at the
        # correct index later
        self.__station_id_to_index_mapping = {}
        for i, id in enumerate(output_station_ids):
            station_id_var[i, :] = _encode_character_array(id, station_id_width)
            self.__station_id_to_index_mapping[id] = i

        # now write the stored attributes
        for var_name, attr_var in stations.attribute_variables.items():
            dimensions = ["station"]
            for dimension_index, dimension_name in enumerate(attr_var.dimensions[1:], start=1):
                dimension_size = attr_var.shape[dimension_index]
                if dimension_name not in self.__dataset.dimensions:
                    self.__dataset.createDimension(dimension_name, dimension_size)
                elif len(self.__dataset.dimensions[dimension_name]) < dimension_size:
                    raise ValueError(
                        f"Dimension '{dimension_name}' is too small in the output dataset."
                    )
                dimensions.append(dimension_name)

            variable = self.__dataset.createVariable(var_name, attr_var.datatype, tuple(dimensions))
            # copy all attributes from the original input variable
            variable.setncatts(attr_var.__dict__)
            if len(dimensions) == 2:
                variable.set_auto_chartostring(False)

            for station_id in output_station_ids:
                if station_id in stations.attributes:
                    station_index = self.__station_id_to_index_mapping[station_id]
                    value = stations.attributes[station_id][var_name]
                    if len(dimensions) == 2:
                        encoding = getattr(attr_var, "_Encoding", "utf-8")
                        width = len(self.__dataset.dimensions[dimensions[1]])
                        variable[station_index, :] = _encode_character_array(value, width, encoding)
                    else:
                        variable[station_index] = value

    def create_variables(self, variable_names: list[str], ensemble_size: int) -> None:
        """
        Creates variables in the dataset for each of the provided parameter ids.
        The write_times and write_station_data methods must be called first, to ensure the necessary
        dimensions have already been created in the output NetCDF file.

        :param variable_names: The parameter ids for which variables must be created. Must be
            unique.
        :param ensemble_size: the number of members in the ensemble
        """
        assert len(set(variable_names)) == len(variable_names)

        assert self.__time_dim is not None, (
            "First call write_times to ensure the time dimension has been created."
        )
        assert self.__station_dim is not None, (
            "First call write_station_data to ensure the station dimension has been created"
        )
        assert (
            self.__station_id_to_index_mapping is not None
        )  # should also be created in write_station_data

        if ensemble_size > 1:
            assert self.__ensemble_member_dim is not None, (
                "First call write_ensemble_data to ensure "
                "the realization dimension has been created"
            )

            for variable_name in variable_names:
                self.__dataset.createVariable(
                    variable_name, "f8", ("time", "station", "realization"), fill_value=np.nan
                )
        else:
            for variable_name in variable_names:
                self.__dataset.createVariable(
                    variable_name, "f8", ("time", "station"), fill_value=np.nan
                )

    def write_output_values(
        self,
        station_id: str,
        variable_name: str,
        ensemble_member_index: int,
        values: np.ndarray,
        ensemble_size: int,
    ) -> None:
        """
        Writes the given data to the dataset. The variable must have already been created through
        the create_variables method. After all calls to write_output_values, the close method must
        be called to flush all changes.

        :param station_id: The id of the station the data is written for.
        :param variable_name: The name of the variable the data is written to (must have already
            been created).
        :param ensemble_member_index: The index associated to the ensemble member
        :param values:        The values that are to be written to the file
        :param ensemble_size: the number of members in the ensemble
        """
        assert self.__station_id_to_index_mapping is not None, (
            "First call write_station_data and create_variables."
        )

        station_index = self.__station_id_to_index_mapping[station_id]
        if ensemble_size > 1:
            self.__dataset.variables[variable_name][:, station_index, ensemble_member_index] = (
                values
            )
        else:
            self.__dataset.variables[variable_name][:, station_index] = values

    def close(self) -> None:
        """
        Closes the NetCDF4 Dataset to ensure all changes made are written to the file.
        This method must be called after writing all data through the various write method.
        """
        self.__dataset.close()
