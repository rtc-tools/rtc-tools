import logging
import os

import numpy as np

import rtctools.data.csv as csv
from rtctools._internal.caching import cached
from rtctools.simulation.io_mixin import IOMixin

logger = logging.getLogger("rtctools")


class CSVMixin(IOMixin):
    """
    Adds reading and writing of CSV timeseries and parameters to your simulation problem.

    During preprocessing, files named ``timeseries_import.csv``, ``initial_state.csv``,
    and ``parameters.csv`` are read from the ``input`` subfolder.

    During postprocessing, a file named ``timeseries_export.csv`` is written to the ``output``
    subfolder.

    :cvar csv_delimiter:           Column delimiter used in CSV files.  Default is ``,``.
    :cvar csv_validate_timeseries: Check consistency of timeseries.  Default is ``True``.
    """

    #: Column delimiter used in CSV files
    csv_delimiter = ","

    #: Check consistency of timeseries
    csv_validate_timeseries = True

    def __init__(self, **kwargs):
        # Call parent class first for default behaviour.
        super().__init__(**kwargs)

    def read(self):
        # Call parent class first for default behaviour.
        super().read()

        # Helper function to check if initial state array actually defines
        # only the initial state
        def check_initial_state_array(initial_state):
            """
            Check length of initial state array, throw exception when larger than 1.
            """
            if initial_state.shape:
                raise Exception(
                    "CSVMixin: Initial state file {} contains more than one row of data. "
                    "Please remove the data row(s) that do not describe the initial "
                    "state.".format(os.path.join(self._input_folder, "initial_state.csv"))
                )

        # Read CSV files
        _timeseries = csv.load(
            os.path.join(self._input_folder, self.timeseries_import_basename + ".csv"),
            delimiter=self.csv_delimiter,
            with_time=True,
        )
        self.__timeseries_times = _timeseries[_timeseries.dtype.names[0]]

        self.io.reference_datetime = self.__timeseries_times[0]

        for key in _timeseries.dtype.names[1:]:
            self.io.set_timeseries(
                key, self.__timeseries_times, np.asarray(_timeseries[key], dtype=np.float64)
            )

        logger.debug("CSVMixin: Read timeseries.")

        try:
            _parameters = csv.load(
                os.path.join(self._input_folder, "parameters.csv"), delimiter=self.csv_delimiter
            )
            for key in _parameters.dtype.names:
                self.io.set_parameter(key, float(_parameters[key]))
            logger.debug("CSVMixin: Read parameters.")
        except OSError:
            pass

        try:
            _initial_state = csv.load(
                os.path.join(self._input_folder, "initial_state.csv"), delimiter=self.csv_delimiter
            )
            logger.debug("CSVMixin: Read initial state.")
            check_initial_state_array(_initial_state)
            self.__initial_state = {
                key: float(_initial_state[key]) for key in _initial_state.dtype.names
            }
        except OSError:
            self.__initial_state = {}

        # Check for collisions in __initial_state and timeseries import (CSV)
        for collision in set(self.__initial_state) & set(_timeseries.dtype.names[1:]):
            if self.__initial_state[collision] == _timeseries[collision][0]:
                continue
            else:
                logger.warning(
                    f"CSVMixin: Entry {collision} in initial_state.csv conflicts with "
                    "timeseries_import.csv"
                )

        # Timestamp check
        if self.csv_validate_timeseries:
            times = self.__timeseries_times
            for i in range(len(times) - 1):
                if times[i] >= times[i + 1]:
                    raise Exception("CSVMixin: Time stamps must be strictly increasing.")

        times = self.__timeseries_times
        dt = times[1] - times[0]

        # Check if the timeseries are truly equidistant
        if self.csv_validate_timeseries:
            for i in range(len(times) - 1):
                if times[i + 1] - times[i] != dt:
                    raise Exception(
                        "CSVMixin: Expecting equidistant timeseries, the time step "
                        f"towards {times[i + 1]} is not the same as the time step(s) before. "
                        "Set equidistant=False if this is intended."
                    )

    def write(self):
        # Call parent class first for default behaviour.
        super().write()

        times = self._simulation_times

        # Write output
        names = ["time"] + sorted(set(self._io_output_variables))
        formats = ["O"] + (len(names) - 1) * ["f8"]
        dtype = {"names": names, "formats": formats}
        data = np.zeros(len(times), dtype=dtype)
        data["time"] = self.io.sec_to_datetime(times, self.io.reference_datetime)
        for variable in self._io_output_variables:
            data[variable] = np.array(self._io_output[variable])

        fname = os.path.join(self._output_folder, self.timeseries_export_basename + ".csv")
        csv.save(fname, data, delimiter=self.csv_delimiter, with_time=True)

    @cached
    def initial_state(self):
        """
        The initial state. Includes entries from parent classes and initial_state.csv

        :returns: A dictionary of variable names and initial state (t0) values.
        """
        # Call parent class first for default values.
        initial_state = super().initial_state()

        # Set of model vars that are allowed to have an initial state
        valid_model_vars = set(self.get_state_variables()) | set(self.get_input_variables())

        # Load initial states from __initial_state
        for variable, value in self.__initial_state.items():
            # Get the canonical vars and signs
            canonical_var, sign = self.alias_relation.canonical_signed(variable)

            # Only store variables that are allowed to have an initial state
            if canonical_var in valid_model_vars:
                initial_state[canonical_var] = value * sign

                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug(f"CSVMixin: Read initial state {variable} = {value}")
            else:
                logger.warning(
                    f"CSVMixin: In initial_state.csv, {variable} is not an input or state variable."
                )
        return initial_state
