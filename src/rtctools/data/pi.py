import bisect
import datetime
import logging
import os
import xml.etree.ElementTree as ET

import defusedxml.ElementTree as DefusedElementTree
import numpy as np

ns = {"fews": "http://www.wldelft.nl/fews", "pi": "http://www.wldelft.nl/fews/PI"}


class Diag:
    """
    diag wrapper.
    """

    ERROR_FATAL = 1 << 0
    ERROR = 1 << 1
    WARN = 1 << 2
    INFO = 1 << 3
    DEBUG = 1 << 4

    def __init__(self, folder, basename="diag"):
        """
        Parse diag file.

        :param folder:   Folder in which diag.xml is found or to be created.
        :param basename: Alternative basename for the diagnostics XML file.
        """
        self.__path_xml = os.path.join(folder, basename + ".xml")

        self.__tree = DefusedElementTree.parse(self.__path_xml)
        self.__xml_root = self.__tree.getroot()

    def get(self, level=ERROR_FATAL):
        """
        Return only the wanted levels (debug, info, etc.)

        :param level: Log level.
        """

        diag_lines = self.__xml_root.findall("*", ns)
        diag_lines_out = []
        USED_LEVELS = []

        if level & self.ERROR_FATAL:
            USED_LEVELS.append("0")
        if level & self.ERROR:
            USED_LEVELS.append("1")
        if level & self.WARN:
            USED_LEVELS.append("2")
        if level & self.INFO:
            USED_LEVELS.append("3")
        if level & self.DEBUG:
            USED_LEVELS.append("4")

        for child in diag_lines:
            for used_level in USED_LEVELS:
                if child.get("level") == used_level:
                    diag_lines_out.append(child)

        return diag_lines_out

    @property
    def has_errors(self):
        """
        True if the log contains errors.
        """
        error_levels = self.ERROR_FATAL | self.ERROR
        diag_lines = self.get(error_levels)

        if len(diag_lines) > 0:
            return True
        else:
            return False


class DiagHandler(logging.Handler):
    """
    PI diag file logging handler.
    """

    def __init__(self, folder, basename="diag", level=logging.NOTSET):
        super().__init__(level=level)

        self.__path_xml = os.path.join(folder, basename + ".xml")

        try:
            self.__tree = DefusedElementTree.parse(self.__path_xml)
            self.__xml_root = self.__tree.getroot()
        except Exception:
            self.__xml_root = ET.Element("{%s}Diag" % (ns["pi"],))  # noqa: UP031
            self.__tree = ET.ElementTree(element=self.__xml_root)

        self.__map_level = {50: 0, 40: 1, 30: 2, 20: 3, 10: 4, 0: 4}

    def emit(self, record):
        self.format(record)

        self.acquire()
        el = ET.SubElement(self.__xml_root, "{%s}line" % (ns["pi"],))  # noqa: UP031
        # Work around cElementTree issue 21403
        el.set("description", record.message)
        el.set("eventCode", record.module + "." + record.funcName)
        el.set("level", str(self.__map_level[record.levelno]))
        self.release()

    def append_element(self, el):
        self.acquire()
        self.__xml_root.append(el)
        self.release()

    def flush(self):
        self.__tree.write(self.__path_xml)

    def close(self):
        self.flush()
        super().close()


class ParameterConfig:
    """
    rtcParameterConfig wrapper.
    """

    def __init__(self, folder, basename):
        """
        Parses a rtcParameterConfig file.

        :param folder:    Folder in which the parameter configuration file is located.
        :param basename:  Basename of the parameter configuration file (e.g, 'rtcParameterConfig').
        """
        if os.path.splitext(basename)[1] != ".xml":
            basename = basename + ".xml"
        self.__path_xml = os.path.join(folder, basename)

        self.__tree = DefusedElementTree.parse(self.__path_xml)
        self.__xml_root = self.__tree.getroot()

    def get(self, group_id, parameter_id, location_id=None, model=None):
        """
        Returns the value of the parameter with ID parameter_id in the group with ID group_id.

        :param group_id:     The ID of the parameter group to look in.
        :param parameter_id: The ID of the parameter to look for.
        :param location_id:  The optional  ID of the parameter location to look in.
        :param model:        The optional ID of the parameter model to look in.

        :returns: The value of the specified parameter.
        """
        groups = self.__xml_root.findall(f"pi:group[@id='{group_id}']", ns)
        for group in groups:
            el = group.find("pi:locationId", ns)
            if location_id is not None and el is not None:
                if location_id != el.text:
                    continue

            el = group.find("pi:model", ns)
            if model is not None and el is not None:
                if model != el.text:
                    continue

            el = group.find(f"pi:parameter[@id='{parameter_id}']", ns)
            if el is None:
                raise KeyError
            return self.__parse_parameter(el)

        raise KeyError(f"No such parameter ({group_id}, {parameter_id})")

    def set(self, group_id, parameter_id, new_value, location_id=None, model=None):
        """
        Set the value of the parameter with ID parameter_id in the group with ID group_id.

        :param group_id:     The ID of the parameter group to look in.
        :param parameter_id: The ID of the parameter to look for.
        :param new_value:    The new value for the parameter.
        :param location_id:  The optional  ID of the parameter location to look in.
        :param model:        The optional ID of the parameter model to look in.
        """
        groups = self.__xml_root.findall(f"pi:group[@id='{group_id}']", ns)
        for group in groups:
            el = group.find("pi:locationId", ns)
            if location_id is not None and el is not None:
                if location_id != el.text:
                    continue

            el = group.find("pi:model", ns)
            if model is not None and el is not None:
                if model != el.text:
                    continue

            el = group.find(f"pi:parameter[@id='{parameter_id}']", ns)
            if el is None:
                raise KeyError
            for child in el:
                if child.tag.endswith("boolValue"):
                    if new_value is True:
                        child.text = "true"
                        return
                    elif new_value is False:
                        child.text = "false"
                        return
                    else:
                        raise Exception(f"Unsupported value for tag {child.tag}")
                elif child.tag.endswith("intValue"):
                    child.text = str(int(new_value))
                    return
                elif child.tag.endswith("dblValue"):
                    child.text = str(new_value)
                    return
                else:
                    raise Exception(f"Unsupported tag {child.tag}")

        raise KeyError(f"No such parameter ({group_id}, {parameter_id})")

    def write(self, folder=None, basename=None):
        """
        Writes the parameter configuration to a file.
        Default behaviour is to overwrite original file.

        :param path:     Optional alternative destination folder
        :param basename: Optional alternative basename of the file (e.g, 'rtcParameterConfig')
        """

        # No path changes- overwrite original file
        if folder is None and basename is None:
            path = self.path

        # We need to reconstruct the path
        else:
            # Determine folder
            if folder is not None:
                if not os.path.exists(folder):
                    # Make sure folder exists
                    raise FileNotFoundError(f"Folder not found: {folder}")
            else:
                # Reuse folder of original file
                folder = os.path.dirname(self.path)

            # Determine basename
            if basename is not None:
                # Make sure basename ends in '.xml'
                if os.path.splitext(basename)[1] != ".xml":
                    basename = basename + ".xml"
            else:
                # Reuse basename of original file
                basename = os.path.split(self.path)[1]

            # Construct path
            path = os.path.join(folder, basename)

        self.__tree.write(path)

    @property
    def path(self):
        return self.__path_xml

    def __parse_type(self, fews_type):
        # Parse a FEWS type to an np type
        if fews_type == "double":
            return np.dtype("float")
        else:
            return np.dtype("S128")

    def __parse_parameter(self, parameter):
        for child in parameter:
            if child.tag.endswith("boolValue"):
                if child.text.lower() == "true":
                    return True
                else:
                    return False
            elif child.tag.endswith("intValue"):
                return int(child.text)
            elif child.tag.endswith("dblValue"):
                return float(child.text)
            elif child.tag.endswith("stringValue"):
                return child.text
            # return dict of lisstart_datetime
            elif child.tag.endswith("table"):
                columnId = {}
                columnType = {}
                for key in child.find("pi:row", ns).attrib:
                    # default Id
                    columnId[key] = key
                    columnType[key] = np.dtype("S128")  # default Type

                # get Id's if present
                el_columnIds = child.find("pi:columnIds", ns)
                if el_columnIds is not None:
                    for key, value in el_columnIds.attrib.items():
                        columnId[key] = value

                # get Types if present
                el_columnTypes = child.find("pi:columnTypes", ns)
                if el_columnTypes is not None:
                    for key, value in el_columnTypes.attrib.items():
                        columnType[key] = self.__parse_type(value)

                # get table contenstart_datetime
                el_row = child.findall("pi:row", ns)
                table = {
                    columnId[key]: np.empty(len(el_row), columnType[key]) for key in columnId
                }  # initialize table

                i_row = 0
                for row in el_row:
                    for key, value in row.attrib.items():
                        table[columnId[key]][i_row] = value
                    i_row += 1
                return table
            elif child.tag.endswith("description"):
                pass
            else:
                raise Exception(f"Unsupported tag {child.tag}")

    def __iter__(self):
        # Iterate over all parameter key, value pairs.
        groups = self.__xml_root.findall("pi:group", ns)
        for group in groups:
            el = group.find("pi:locationId", ns)
            if el is not None:
                location_id = el.text
            else:
                location_id = None

            el = group.find("pi:model", ns)
            if el is not None:
                model_id = el.text
            else:
                model_id = None

            parameters = group.findall("pi:parameter", ns)
            for parameter in parameters:
                yield (
                    location_id,
                    model_id,
                    parameter.attrib["id"],
                    self.__parse_parameter(parameter),
                )


class Timeseries:
    """
    PI timeseries wrapper.
    """

    def __init__(
        self,
        data_config,
        folder,
        basename,
        binary=True,
        pi_validate_times=False,
        make_new_file=False,
    ):
        """
        Load the timeseries from disk.

        :param data_config:       A :class:`DataConfig` object.
        :param folder:            The folder in which the time series is located.
        :param basename:          The basename of the time series file.
        :param binary:            True if the time series data is stored in a separate binary file.
                                  Default is ``True``.
        :param pi_validate_times: Check consistency of times.  Default is ``False``.
        :param make_new_file:     Make new XML object which can be filled and written to a new file.
                                  Default is ``False``.
        """
        self.__data_config = data_config

        self.__folder = folder
        self.__basename = basename

        self.__internal_dtype = np.float64
        self.__pi_dtype = np.float32

        self.make_new_file = make_new_file
        if self.make_new_file:
            self.__reset_xml_tree()
        else:
            self.__tree = DefusedElementTree.parse(self.path)
            self.__xml_root = self.__tree.getroot()

        self.__values = [{}]
        self.__units = [{}]

        self.__binary = binary

        if not self.make_new_file:
            f = None
            if self.__binary:
                try:
                    f = open(self.binary_path, "rb")
                except OSError:
                    # Support placeholder XML files.
                    pass

            # Read timezone
            timezone = self.__xml_root.find("pi:timeZone", ns)
            if timezone is not None:
                self.__timezone = float(timezone.text)
            else:
                self.__timezone = None

            # Check data consistency
            self.__dt = None
            self.__start_datetime = None
            self.__end_datetime = None
            self.__forecast_datetime = None
            self.__forecast_index = None
            self.__contains_ensemble = False
            self.__ensemble_size = 1
            ensemble_indexes = set()
            for series in self.__xml_root.findall("pi:series", ns):
                header = series.find("pi:header", ns)

                variable = self.__data_config.variable(header)

                try:
                    dt = self.__parse_time_step(header.find("pi:timeStep", ns))
                except ValueError:
                    raise Exception(
                        f"PI: Multiplier of time step of variable {variable} "
                        "must be a positive integer per the PI schema."
                    )
                if self.__dt is None:
                    self.__dt = dt
                else:
                    if dt != self.__dt:
                        raise Exception("PI: Not all timeseries have the same time step size.")
                try:
                    start_datetime = self.__parse_date_time(header.find("pi:startDate", ns))
                    if self.__start_datetime is None:
                        self.__start_datetime = start_datetime
                    else:
                        if start_datetime < self.__start_datetime:
                            self.__start_datetime = start_datetime
                except (AttributeError, ValueError):
                    raise Exception(
                        "PI: Variable {} in {} has no startDate.".format(
                            variable, os.path.join(self.__folder, basename + ".xml")
                        )
                    )

                try:
                    end_datetime = self.__parse_date_time(header.find("pi:endDate", ns))
                    if self.__end_datetime is None:
                        self.__end_datetime = end_datetime
                    else:
                        if end_datetime > self.__end_datetime:
                            self.__end_datetime = end_datetime
                except (AttributeError, ValueError):
                    raise Exception(
                        "PI: Variable {} in {} has no endDate.".format(
                            variable, os.path.join(self.__folder, basename + ".xml")
                        )
                    )

                el = header.find("pi:forecastDate", ns)
                if el is not None:
                    forecast_datetime = self.__parse_date_time(el)
                else:
                    # the timeseries has no forecastDate, so the forecastDate
                    # is set to the startDate (per the PI-schema)
                    forecast_datetime = start_datetime
                if self.__forecast_datetime is None:
                    self.__forecast_datetime = forecast_datetime
                else:
                    if el is not None and forecast_datetime != self.__forecast_datetime:
                        raise Exception("PI: Not all timeseries share the same forecastDate.")

                el = header.find("pi:ensembleMemberIndex", ns)
                if el is not None:
                    ensemble_indexes.add(int(el.text))

            # We assume the ensemble ids are zero-based and increasing by 1
            if len(ensemble_indexes) > 1:
                # check if ids are zero-based and increasing by 1
                sorted_ensemble_indexes = sorted(ensemble_indexes)
                if sorted_ensemble_indexes != list(range(len(sorted_ensemble_indexes))):
                    raise ValueError(
                        "PI: Ensemble ids must be zero-based and increasing by 1 when more than one"
                        " ensemble member is present."
                    )
                self.__contains_ensemble = True
                self.__ensemble_size = len(ensemble_indexes)
            else:
                # There are no ensemble members, or a single ensemble member of arbitrary id
                self.__contains_ensemble = False
                self.__ensemble_size = 1

            # Define the times, and floor the global forecast_datetime to the
            # global time step to get its index
            if self.__dt:
                t_len = int(
                    round(
                        (self.__end_datetime - self.__start_datetime).total_seconds()
                        / self.__dt.total_seconds()
                        + 1
                    )
                )
                self.__times = [self.__start_datetime + i * self.__dt for i in range(t_len)]
            else:  # Timeseries are non-equidistant
                self.__times = []
                times_set = set()
                for series in self.__xml_root.findall("pi:series", ns):
                    events = series.findall("pi:event", ns)
                    # The union of all timeseries covers all the run-period (e.g.history)
                    times_of_timeseries = [self.__parse_date_time(e) for e in events]
                    times_set.update(times_of_timeseries)
                self.__times = sorted(times_set)
            # Check if the time steps of all series match the time steps of the global
            # time range.
            if pi_validate_times:
                ref_times_set = set(self.__times)
                for series in self.__xml_root.findall("pi:series", ns):
                    events = series.findall("pi:event", ns)
                    times = {self.__parse_date_time(e) for e in events}
                    if not ref_times_set.issuperset(times):
                        raise ValueError(
                            "PI: Not all timeseries share the same time step spacing. Make sure "
                            "the time steps of all series are a subset of the global time steps."
                        )

            if self.__forecast_datetime is not None:
                if self.__dt:
                    self.__forecast_datetime = self.__floor_date_time(
                        dt=self.__forecast_datetime, tdel=self.__dt
                    )
                try:
                    self.__forecast_index = self.__times.index(self.__forecast_datetime)
                except ValueError:
                    # This may occur if forecast_datetime is outside of
                    # the timeseries' range.  Can be a valid case for historical
                    # timeseries, for instance.
                    self.__forecast_index = -1

            # Parse data
            for series in self.__xml_root.findall("pi:series", ns):
                header = series.find("pi:header", ns)

                variable = self.__data_config.variable(header)

                dt = self.__parse_time_step(header.find("pi:timeStep", ns))
                start_datetime = self.__parse_date_time(header.find("pi:startDate", ns))
                end_datetime = self.__parse_date_time(header.find("pi:endDate", ns))

                make_virtual_ensemble = False
                if self.__contains_ensemble:
                    el = header.find("pi:ensembleMemberIndex", ns)
                    if el is not None:
                        ensemble_member = int(el.text)
                        while ensemble_member >= len(self.__values):
                            self.__values.append({})
                        while ensemble_member >= len(self.__units):
                            self.__units.append({})
                    else:
                        ensemble_member = 0
                    if el is None:
                        # Expand values dict to accommodate referencing of (virtual)
                        # ensemble series to the input values. This is e.g. needed
                        # for initial states that have a single historical values.
                        while self.ensemble_size > len(self.__values):
                            self.__values.append({})
                        while self.ensemble_size > len(self.__units):
                            self.__units.append({})
                        make_virtual_ensemble = True
                else:
                    ensemble_member = 0

                if self.__dt:
                    n_values = int(
                        round(
                            (end_datetime - start_datetime).total_seconds() / dt.total_seconds() + 1
                        )
                    )
                else:
                    n_values = (
                        bisect.bisect_left(self.__times, end_datetime)
                        - bisect.bisect_left(self.__times, start_datetime)
                        + 1
                    )

                if self.__binary:
                    if f is not None:
                        self.__values[ensemble_member][variable] = np.fromfile(
                            f, count=n_values, dtype=self.__pi_dtype
                        )
                    else:
                        self.__values[ensemble_member][variable] = np.empty(
                            n_values, dtype=self.__internal_dtype
                        )
                        self.__values[ensemble_member][variable].fill(np.nan)

                else:
                    events = series.findall("pi:event", ns)

                    self.__values[ensemble_member][variable] = np.empty(
                        n_values, dtype=self.__internal_dtype
                    )
                    self.__values[ensemble_member][variable].fill(np.nan)
                    # This assumes that start_datetime equals the datetime of the
                    # first value (which should be the case).
                    for i in range(min(n_values, len(events))):
                        self.__values[ensemble_member][variable][i] = float(events[i].get("value"))

                miss_val = float(header.find("pi:missVal", ns).text)
                self.__values[ensemble_member][variable][
                    self.__values[ensemble_member][variable] == miss_val
                ] = np.nan

                unit = header.find("pi:units", ns).text
                self.set_unit(variable, unit=unit, ensemble_member=ensemble_member)

                # Prepend empty space, if start_datetime > self.__start_datetime.
                if start_datetime > self.__start_datetime:
                    if self.__dt:
                        filler = np.empty(
                            int(
                                round(
                                    (start_datetime - self.__start_datetime).total_seconds()
                                    / dt.total_seconds()
                                )
                            ),
                            dtype=self.__internal_dtype,
                        )
                    else:
                        filler = np.empty(
                            int(
                                round(
                                    bisect.bisect_left(self.__times, start_datetime)
                                    - bisect.bisect_left(self.__times, self.__start_datetime)
                                )
                            ),
                            dtype=self.__internal_dtype,
                        )

                    filler.fill(np.nan)
                    self.__values[ensemble_member][variable] = np.hstack(
                        (filler, self.__values[ensemble_member][variable])
                    )

                # Append empty space, if end_datetime < self.__end_datetime
                if end_datetime < self.__end_datetime:
                    if self.__dt:
                        filler = np.empty(
                            int(
                                round(
                                    (self.__end_datetime - end_datetime).total_seconds()
                                    / dt.total_seconds()
                                )
                            ),
                            dtype=self.__internal_dtype,
                        )
                    else:
                        filler = np.empty(
                            int(
                                round(
                                    bisect.bisect_left(self.__times, self.__end_datetime)
                                    - bisect.bisect_left(self.__times, end_datetime)
                                )
                            ),
                            dtype=self.__internal_dtype,
                        )

                    filler.fill(np.nan)
                    self.__values[ensemble_member][variable] = np.hstack(
                        (self.__values[ensemble_member][variable], filler)
                    )

                if make_virtual_ensemble:
                    # Make references to the original input series for the virtual
                    # ensemble members.
                    for i in range(1, self.ensemble_size):
                        self.__values[i][variable] = self.__values[0][variable]
                        self.set_unit(variable, unit=unit, ensemble_member=i)

            if not self.__dt:
                # Remove time values outside the start/end datetimes.
                # Only needed for non-equidistant, because we can't build the
                # times automatically from global start/end datetime.
                self.__times = self.__times[
                    bisect.bisect_left(self.__times, self.__start_datetime) : bisect.bisect_left(
                        self.__times, self.__end_datetime
                    )
                    + 1
                ]

            if f is not None and self.__binary:
                f.close()

    def __reset_xml_tree(self):
        # Make a new empty XML tree
        self.__xml_root = ET.Element("{%s}" % (ns["pi"],) + "TimeSeries")  # noqa: UP031
        self.__tree = ET.ElementTree(self.__xml_root)

        self.__xml_root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        self.__xml_root.set("version", "1.2")
        self.__xml_root.set(
            "xsi:schemaLocation",
            "http://www.wldelft.nl/fews/PI "
            "http://fews.wldelft.nl/schemas/version1.0/pi-schemas/pi_timeseries.xsd",
        )

    def __parse_date_time(self, el):
        # Parse a PI date time element.
        return datetime.datetime.strptime(
            el.get("date") + " " + el.get("time"), "%Y-%m-%d %H:%M:%S"
        )

    def __parse_time_step(self, el):
        # Parse a PI time step element.
        if el.get("unit") == "second":
            return datetime.timedelta(seconds=int(el.get("multiplier")))
        elif el.get("unit") == "nonequidistant":
            return None
        else:
            raise Exception("Unsupported unit type: " + el.get("unit"))

    def __floor_date_time(self, dt, tdel):
        """
        Floor a PI date time to an integer number of PI time steps from the
        start date time.
        """
        roundTo = tdel.total_seconds()

        seconds = (dt - self.__start_datetime).seconds
        # // is a floor division:
        rounding = (seconds + roundTo / 2) // roundTo * roundTo
        return dt + datetime.timedelta(0, rounding - seconds, -dt.microsecond)

    def __add_header(
        self, variable, location_parameter_id, ensemble_member=0, miss_val=-999, unit="unit_unknown"
    ):
        """
        Add a timeseries header to the timeseries object.
        """
        # Save current datetime
        now = datetime.datetime.now()

        # Define the basic structure of the header
        header_elements = [
            "type",
            "locationId",
            "parameterId",
            "timeStep",
            "startDate",
            "endDate",
            "missVal",
            "stationName",
            "units",
            "creationDate",
            "creationTime",
        ]
        header_element_texts = [
            "instantaneous",
            location_parameter_id.location_id,
            location_parameter_id.parameter_id,
            "",
            "",
            "",
            str(miss_val),
            location_parameter_id.location_id,
            unit,
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
        ]

        # Add ensembleMemberIndex, forecastDate and qualifierId if necessary.
        if self.__forecast_datetime != self.__start_datetime:
            header_elements.insert(6, "forecastDate")
            header_element_texts.insert(6, "")
        if self.contains_ensemble:
            header_elements.insert(3, "ensembleMemberIndex")
            header_element_texts.insert(3, str(ensemble_member))
        if len(location_parameter_id.qualifier_id) > 0:
            # Track relative index to preserve original ordering of qualifier ID's
            i = 0
            for qualifier_id in location_parameter_id.qualifier_id:
                header_elements.insert(3, "qualifierId")
                header_element_texts.insert(3 + i, qualifier_id)
                i += 1

        # Fill the basics of the series
        series = ET.Element("{%s}" % (ns["pi"],) + "series")  # noqa: UP031
        header = ET.SubElement(series, "{%s}" % (ns["pi"],) + "header")  # noqa: UP031
        for i in range(len(header_elements)):
            el = ET.SubElement(header, "{%s}" % (ns["pi"],) + header_elements[i])  # noqa: UP031
            el.text = header_element_texts[i]

        el = header.find("pi:timeStep", ns)
        # Set time step
        if self.dt:
            el.set("unit", "second")
            el.set("multiplier", str(int(self.dt.total_seconds())))
        else:
            el.set("unit", "nonequidistant")

        # Set the time range.
        el = header.find("pi:startDate", ns)
        el.set("date", self.__start_datetime.strftime("%Y-%m-%d"))
        el.set("time", self.__start_datetime.strftime("%H:%M:%S"))
        el = header.find("pi:endDate", ns)
        el.set("date", self.__end_datetime.strftime("%Y-%m-%d"))
        el.set("time", self.__end_datetime.strftime("%H:%M:%S"))

        # Set the forecast date if applicable
        if self.__forecast_datetime != self.__start_datetime:
            el = header.find("pi:forecastDate", ns)
            el.set("date", self.__forecast_datetime.strftime("%Y-%m-%d"))
            el.set("time", self.__forecast_datetime.strftime("%H:%M:%S"))

        # Add series to xml
        self.__xml_root.append(series)

    def write(self, output_folder=None, output_filename=None) -> None:
        """
        Writes the time series data to disk.

        :param output_folder:   The folder in which the output file is located.
                                If None, the original folder is used.
        :param output_filename: The name of the output file without extension.
                                If None, the original filename is used.
        """
        xml_path = self.output_path(output_folder, output_filename)
        binary_path = self.output_binary_path(output_folder, output_filename)

        if self.__binary:
            f = open(binary_path, "wb")

        if self.make_new_file:
            # Force reinitialization in case write() is called more than once
            self.__reset_xml_tree()

            for ensemble_member in range(len(self.__values)):
                for variable in sorted(self.__values[ensemble_member].keys()):
                    location_parameter_id = self.__data_config.pi_variable_ids(variable)
                    unit = self.get_unit(variable, ensemble_member)
                    self.__add_header(
                        variable,
                        location_parameter_id,
                        ensemble_member=ensemble_member,
                        miss_val=-999,
                        unit=unit,
                    )

        for ensemble_member in range(len(self.__values)):
            if self.timezone is not None:
                timezone = self.__xml_root.find("pi:timeZone", ns)
                if timezone is None:
                    timezone = ET.Element("{%s}" % (ns["pi"],) + "timeZone")  # noqa: UP031
                    # timeZone has to be the first element according to the schema
                    self.__xml_root.insert(0, timezone)
                timezone.text = str(self.timezone)

            for series in self.__xml_root.findall("pi:series", ns):
                header = series.find("pi:header", ns)

                # First check ensembleMemberIndex, to see if it is the correct one.
                el = header.find("pi:ensembleMemberIndex", ns)
                if el is not None:
                    if ensemble_member != int(el.text):
                        # Skip over this series, wrong index.
                        continue

                # Update the time range, which may have changed.
                el = header.find("pi:startDate", ns)
                el.set("date", self.__start_datetime.strftime("%Y-%m-%d"))
                el.set("time", self.__start_datetime.strftime("%H:%M:%S"))

                el = header.find("pi:endDate", ns)
                el.set("date", self.__end_datetime.strftime("%Y-%m-%d"))
                el.set("time", self.__end_datetime.strftime("%H:%M:%S"))

                variable = self.__data_config.variable(header)

                miss_val = header.find("pi:missVal", ns).text
                values = self.__values[ensemble_member][variable]

                # Update the header, which may have changed
                el = header.find("pi:units", ns)
                el.text = self.get_unit(variable, ensemble_member)

                # No values to be written, so the entire element is removed from
                # the XML, and the loop restarts.
                if len(values) == 0:
                    self.__xml_root.remove(series)
                    continue

                # Write output
                nans = np.isnan(values)
                if self.__binary:
                    f.write(values.astype(self.__pi_dtype).tobytes())
                else:
                    events = series.findall("pi:event", ns)

                    t = self.__start_datetime
                    for i, value in enumerate(values):
                        if self.dt is None:
                            t = self.times[i]

                        if i < len(events):
                            event = events[i]
                        else:
                            event = ET.Element("pi:event")
                            series.append(event)

                        # Always set the date/time, so that any date/time steps
                        # that are wrong in the placeholder file are corrected.
                        event.set("date", t.strftime("%Y-%m-%d"))
                        event.set("time", t.strftime("%H:%M:%S"))

                        if nans[i]:
                            event.set("value", miss_val)
                        else:
                            event.set("value", str(value))

                        if self.dt:
                            t += self.dt

                    # Remove superfluous elements
                    if len(events) > len(values):
                        for i in range(len(values), len(events)):
                            series.remove(events[i])

        if self.__binary:
            f.close()

        self.format_xml_data()
        self.__tree.write(xml_path)

    def format_xml_data(self):
        """
        Format the XML data

        Make sure it has proper indentation and newlines.
        """
        self.format_xml_element(self.__xml_root)

    @staticmethod
    def format_xml_element(element: ET.Element, level=0):
        """
        Format an XML element

        Make sure it has proper indentation and newlines.
        This code is based on
        https://stackoverflow.com/questions/3095434/inserting-newlines-in-xml-file-generated-via-xml-etree-elementtree-in-python
        """
        indent = "\n" + level * "  "
        if len(element):
            if not element.text or not element.text.strip():
                element.text = indent + "  "
            if not element.tail or not element.tail.strip():
                element.tail = indent
            for subelement in element:
                Timeseries.format_xml_element(subelement, level + 1)
            if not element.tail or not element.tail.strip():
                element.tail = indent
        else:
            if level and (not element.tail or not element.tail.strip()):
                element.tail = indent

    @property
    def contains_ensemble(self):
        """
        Flag to indicate TimeSeries contains an ensemble.
        """
        return self.__contains_ensemble

    @contains_ensemble.setter
    def contains_ensemble(self, contains_ensemble):
        if not contains_ensemble:
            assert self.ensemble_size == 1
        self.__contains_ensemble = contains_ensemble

    @property
    def ensemble_size(self):
        """
        Ensemble size.
        """
        return self.__ensemble_size

    @ensemble_size.setter
    def ensemble_size(self, ensemble_size):
        if ensemble_size > 1:
            assert self.contains_ensemble
        self.__ensemble_size = ensemble_size
        while self.__ensemble_size > len(self.__values):
            self.__values.append({})
        while self.__ensemble_size > len(self.__units):
            self.__units.append({})

    @property
    def start_datetime(self):
        """
        Start time.
        """
        return self.__start_datetime

    @property
    def end_datetime(self):
        """
        End time.
        """
        return self.__end_datetime

    @property
    def forecast_datetime(self):
        """
        Forecast time (t0).
        """
        return self.__forecast_datetime

    @forecast_datetime.setter
    def forecast_datetime(self, forecast_datetime):
        self.__forecast_datetime = forecast_datetime
        self.__forecast_index = self.__times.index(forecast_datetime)

    @property
    def forecast_index(self):
        """
        Forecast time (t0) index.
        """
        return self.__forecast_index

    @property
    def dt(self):
        """
        Time step.
        """
        return self.__dt

    @dt.setter
    def dt(self, dt):
        self.__dt = dt

    @property
    def times(self):
        """
        Time stamps.
        """
        return self.__times

    @times.setter
    def times(self, times):
        self.__times = times
        self.__start_datetime = times[0]
        self.__end_datetime = times[-1]

    @property
    def timezone(self):
        """
        Time zone in decimal hours shift from GMT.
        """
        return self.__timezone

    @timezone.setter
    def timezone(self, timezone):
        self.__timezone = timezone

    def get(self, variable, ensemble_member=0):
        """
        Look up a time series.

        :param variable:        Time series ID.
        :param ensemble_member: Ensemble member index.

        :returns: A :class:`Timeseries` object.
        """
        return self.__values[ensemble_member][variable]

    def set(self, variable, new_values, unit=None, ensemble_member=0):
        """
        Fill a time series with new values, and set the unit.

        :param variable:        Time series ID.
        :param new_values:      List of new values.
        :param unit:            Unit.
        :param ensemble_member: Ensemble member index.
        """
        self.__values[ensemble_member][variable] = new_values
        if unit is None:
            unit = self.get_unit(variable, ensemble_member)
        self.set_unit(variable, unit, ensemble_member)

    def get_unit(self, variable, ensemble_member=0):
        """
        Look up the unit of a time series.

        :param variable:        Time series ID.
        :param ensemble_member: Ensemble member index.

        :returns: A :string: containing the unit. If not set for the variable,
            returns 'unit_unknown'.
        """
        try:
            return self.__units[ensemble_member][variable]
        except KeyError:
            return "unit_unknown"

    def set_unit(self, variable, unit, ensemble_member=0):
        """
        Set the unit of a time series.

        :param variable:        Time series ID.
        :param unit:            Unit.
        :param ensemble_member: Ensemble member index.
        """
        self.__units[ensemble_member][variable] = unit

    def resize(self, start_datetime, end_datetime):
        """
        Resize the timeseries to stretch from start_datetime to end_datetime.

        :param start_datetime: Start date and time.
        :param end_datetime:   End date and time.
        """

        if self.__dt:
            n_delta_s = int(
                round(
                    (start_datetime - self.__start_datetime).total_seconds()
                    / self.__dt.total_seconds()
                )
            )
        else:
            if start_datetime >= self.__start_datetime:
                n_delta_s = bisect.bisect_left(self.__times, start_datetime) - bisect.bisect_left(
                    self.__times, self.__start_datetime
                )
            else:
                raise ValueError(
                    "PI: Resizing a non-equidistant timeseries to stretch "
                    "outside of the global range of times is not allowed."
                )

        for ensemble_member in range(len(self.__values)):
            if n_delta_s > 0:
                # New start datetime lies after old start datetime (timeseries will be shortened).
                for key in self.__values[ensemble_member].keys():
                    self.__values[ensemble_member][key] = self.__values[ensemble_member][key][
                        n_delta_s:
                    ]
            elif n_delta_s < 0:
                # New start datetime lies before old start datetime (timeseries will be lengthened).
                filler = np.empty(abs(n_delta_s))
                filler.fill(np.nan)
                for key in self.__values[ensemble_member].keys():
                    self.__values[ensemble_member][key] = np.hstack(
                        (filler, self.__values[ensemble_member][key])
                    )
        self.__start_datetime = start_datetime

        if self.__dt:
            n_delta_e = int(
                round(
                    (end_datetime - self.__end_datetime).total_seconds() / self.__dt.total_seconds()
                )
            )
        else:
            if end_datetime <= self.__end_datetime:
                n_delta_e = bisect.bisect_left(self.__times, end_datetime) - bisect.bisect_left(
                    self.__times, self.__end_datetime
                )
            else:
                raise ValueError(
                    "PI: Resizing a non-equidistant timeseries to stretch "
                    "outside of the global range of times is not allowed."
                )

        for ensemble_member in range(len(self.__values)):
            if n_delta_e > 0:
                # New end datetime lies after old end datetime (timeseries will be lengthened).
                filler = np.empty(n_delta_e)
                filler.fill(np.nan)
                for key in self.__values[ensemble_member].keys():
                    self.__values[ensemble_member][key] = np.hstack(
                        (self.__values[ensemble_member][key], filler)
                    )
            elif n_delta_e < 0:
                # New end datetime lies before old end datetime (timeseries will be shortened).
                for key in self.__values[ensemble_member].keys():
                    self.__values[ensemble_member][key] = self.__values[ensemble_member][key][
                        :n_delta_e
                    ]
        self.__end_datetime = end_datetime

    @property
    def path(self) -> str:
        """
        The path to the original xml file.
        """
        return os.path.join(self.__folder, self.__basename + ".xml")

    @property
    def binary_path(self) -> str:
        """
        The path to the original binary data .bin file.
        """
        return os.path.join(self.__folder, self.__basename + ".bin")

    def _output_path_without_extension(self, output_folder=None, output_filename=None) -> str:
        """
        Get the output path without file extension.
        """
        if output_folder is None:
            output_folder = self.__folder
        if output_filename is None:
            output_filename = self.__basename
        return os.path.join(output_folder, output_filename)

    def output_path(self, output_folder=None, output_filename=None) -> str:
        """
        Get the path to the output xml file.

        The optional arguments are the same as in :py:method:`write`.
        """
        return self._output_path_without_extension(output_folder, output_filename) + ".xml"

    def output_binary_path(self, output_folder=None, output_filename=None) -> str:
        """
        Get the path to the output binary file.

        The optional arguments are the same as in :py:method:`write`.
        """
        return self._output_path_without_extension(output_folder, output_filename) + ".bin"

    def items(self, ensemble_member=0):
        """
        Returns an iterator over all timeseries IDs and value arrays for the given
        ensemble member.
        """
        for key in self.__values[ensemble_member].keys():
            yield key, self.get(key, ensemble_member)
