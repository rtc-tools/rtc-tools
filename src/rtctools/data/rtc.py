import logging
import os
from collections import namedtuple

import defusedxml.ElementTree as DefusedElementTree

ts_ids = namedtuple("ids", "location_id parameter_id qualifier_id")
p_ids = namedtuple("ids", "model_id location_id parameter_id")

ns = {"fews": "http://www.wldelft.nl/fews", "pi": "http://www.wldelft.nl/fews/PI"}

logger = logging.getLogger("rtctools")


class DataConfig:
    """
    rtcDataConfig wrapper.

    Used to map PI timeseries to RTC-Tools variable names.
    """

    def __init__(self, folder):
        """
        Parse rtcDataConfig file

        :param folder: Folder in which rtcDataConfig.xml is located.
        """
        self.__variable_map = {}
        self.__location_parameter_ids = {}
        self.__parameter_map = {}
        self.__model_parameter_ids = {}

        path = os.path.join(folder, "rtcDataConfig.xml")
        try:
            tree = DefusedElementTree.parse(path)
            root = tree.getroot()

            timeseriess1 = root.findall("./*/fews:timeSeries", ns)
            timeseriess2 = root.findall("./fews:timeSeries", ns)
            timeseriess1.extend(timeseriess2)

            for timeseries in timeseriess1:
                pi_timeseries = timeseries.find("fews:PITimeSeries", ns)
                if pi_timeseries is not None:
                    internal_id = timeseries.get("id")
                    external_id = self.__pi_timeseries_id(pi_timeseries, "fews")

                    if internal_id in self.__location_parameter_ids:
                        message = (
                            "Found more than one external timeseries "
                            f"mapped to internal id {internal_id} in {path}."
                        )
                        logger.error(message)
                        raise Exception(message)
                    elif external_id in self.__variable_map:
                        message = (
                            "Found more than one internal timeseries "
                            f"mapped to external id {external_id} in {path}."
                        )
                        logger.error(message)
                        raise Exception(message)
                    else:
                        self.__location_parameter_ids[internal_id] = (
                            self.__pi_location_parameter_id(pi_timeseries, "fews")
                        )
                        self.__variable_map[external_id] = internal_id

            for k in ["import", "export"]:
                res = root.find(f"./fews:{k}/fews:PITimeSeriesFile/fews:timeSeriesFile", ns)
                if res is not None:
                    setattr(self, f"basename_{k}", os.path.splitext(res.text)[0])

            parameters = root.findall("./fews:parameter", ns)
            if parameters is not None:
                for parameter in parameters:
                    pi_parameter = parameter.find("fews:PIParameter", ns)
                    if pi_parameter is not None:
                        internal_id = parameter.get("id")
                        external_id = self.__pi_parameter_id(pi_parameter, "fews")

                        if internal_id in self.__model_parameter_ids:
                            message = (
                                "Found more than one external parameter mapped "
                                f"to internal id {internal_id} in {path}."
                            )
                            logger.error(message)
                            raise Exception(message)
                        if external_id in self.__parameter_map:
                            message = (
                                "Found more than one internal parameter mapped to external "
                                f"modelId {external_id.model_id}, "
                                f"locationId {external_id.location_id}, "
                                f"parameterId {external_id.parameter_id} in {path}."
                            )
                            logger.error(message)
                            raise Exception(message)
                        else:
                            self.__model_parameter_ids[internal_id] = self.__pi_model_parameter_id(
                                pi_parameter, "fews"
                            )
                            self.__parameter_map[external_id] = internal_id

        except OSError:
            logger.error(f'No rtcDataConfig.xml file was found in "{folder}".')
            raise

    def __pi_timeseries_id(self, el, namespace):
        location_id = el.find(namespace + ":locationId", ns).text
        parameter_id = el.find(namespace + ":parameterId", ns).text

        timeseries_id = location_id + ":" + parameter_id

        qualifiers = el.findall(namespace + ":qualifierId", ns)
        qualifier_ids = []
        for qualifier in qualifiers:
            qualifier_ids.append(qualifier.text)

        if len(qualifier_ids) > 0:
            qualifier_ids.sort()

            return timeseries_id + ":" + ":".join(qualifier_ids)
        else:
            return timeseries_id

    def __pi_location_parameter_id(self, el, namespace):
        qualifier_ids = []
        qualifiers = el.findall(namespace + ":qualifierId", ns)
        for qualifier in qualifiers:
            qualifier_ids.append(qualifier.text)

        location_parameter_ids = ts_ids(
            location_id=el.find(namespace + ":locationId", ns).text,
            parameter_id=el.find(namespace + ":parameterId", ns).text,
            qualifier_id=qualifier_ids,
        )
        return location_parameter_ids

    def __pi_parameter_id(self, el, namespace):
        model_id = el.find(namespace + ":modelId", ns).text
        location_id = el.find(namespace + ":locationId", ns).text
        parameter_id = el.find(namespace + ":parameterId", ns).text

        return self.__long_parameter_id(parameter_id, location_id, model_id)

    def __pi_model_parameter_id(self, el, namespace):
        model_id = el.find(namespace + ":modelId", ns).text
        location_id = el.find(namespace + ":locationId", ns).text
        parameter_id = el.find(namespace + ":parameterId", ns).text

        model_parameter_ids = p_ids(
            model_id=(model_id if model_id is not None else ""),
            location_id=(location_id if location_id is not None else ""),
            parameter_id=(parameter_id if parameter_id is not None else ""),
        )

        return model_parameter_ids

    def __long_parameter_id(self, parameter_id, location_id=None, model_id=None):
        """
        Convert a model, location and parameter combination to a single parameter id
        of the form model:location:parameter.
        """
        if location_id is not None:
            parameter_id = location_id + ":" + parameter_id
        if model_id is not None:
            parameter_id = model_id + ":" + parameter_id
        return parameter_id

    def variable(self, pi_header):
        """
        Map a PI timeseries header to an RTC-Tools timeseries ID.

        :param pi_header: XML ElementTree node containing a PI timeseries header.

        :returns: A timeseries ID.
        :rtype: string
        """
        series_id = self.__pi_timeseries_id(pi_header, "pi")
        try:
            return self.__variable_map[series_id]
        except KeyError:
            return series_id

    def pi_variable_ids(self, variable):
        """
        Map an RTC-Tools timeseries ID to a named tuple of location, parameter
        and qualifier ID's.

        :param variable: A timeseries ID.

        :returns: A named tuple with fields location_id, parameter_id and qualifier_id.
        :rtype: namedtuple
        :raises KeyError: If the timeseries ID has no mapping in rtcDataConfig.
        """
        return self.__location_parameter_ids[variable]

    def parameter(self, parameter_id, location_id=None, model_id=None):
        """
        Map a combination of parameter ID, location ID, model ID to an
        RTC-Tools parameter ID.

        :param parameter_id: String with parameter ID
        :param location_id: String with location ID
        :param model_id: String with model ID

        :returns: A parameter ID.
        :rtype: string
        :raises KeyError: If the combination has no mapping in rtcDataConfig.
        """
        parameter_id_long = self.__long_parameter_id(parameter_id, location_id, model_id)

        return self.__parameter_map[parameter_id_long]

    def pi_parameter_ids(self, parameter):
        """
        Map an RTC-Tools model parameter ID to a named tuple of model, location
        and parameter ID's.

        :param parameter: A model parameter ID.

        :returns: A named tuple with fields model_id, location_id and parameter_id.
        :rtype: namedtuple
        :raises KeyError: If the parameter ID has no mapping in rtcDataConfig.
        """
        return self.__model_parameter_ids[parameter]
