import logging
import unittest
from unittest.mock import MagicMock, call, patch

from rtctools._internal.pymoca_helpers import load_pymoca_model

PYMOCA_TRANSFER_MODEL = (
    "rtctools._internal.pymoca_helpers.pymoca.backends.casadi.api.transfer_model"
)
MODEL_FOLDER = "/some/model"
MODEL_NAME = "MyModel"
LOGGER = logging.getLogger("test")


def _opts(cache=True):
    return {"cache": cache}


class TestLoadPymocaModel(unittest.TestCase):
    def test_success_returns_model(self):
        model = MagicMock()
        with patch(PYMOCA_TRANSFER_MODEL, return_value=model) as mock_transfer:
            result = load_pymoca_model(MODEL_FOLDER, MODEL_NAME, _opts(), LOGGER)
        self.assertIs(result, model)
        mock_transfer.assert_called_once_with(MODEL_FOLDER, MODEL_NAME, _opts())

    def test_non_cache_error_raises_immediately(self):
        with patch(PYMOCA_TRANSFER_MODEL, side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                load_pymoca_model(MODEL_FOLDER, MODEL_NAME, _opts(cache=False), LOGGER)

    def test_module_not_found_error_no_cache_raises_immediately(self):
        with patch(PYMOCA_TRANSFER_MODEL, side_effect=ModuleNotFoundError("missing module")):
            with self.assertRaises(ModuleNotFoundError):
                load_pymoca_model(MODEL_FOLDER, MODEL_NAME, _opts(cache=False), LOGGER)

    def test_module_not_found_error_cache_retries(self):
        model = MagicMock()
        with patch(
            PYMOCA_TRANSFER_MODEL, side_effect=[ModuleNotFoundError("stale cache"), model]
        ) as mock_transfer:
            result = load_pymoca_model(MODEL_FOLDER, MODEL_NAME, _opts(cache=True), LOGGER)
        self.assertIs(result, model)
        self.assertEqual(
            mock_transfer.call_args_list,
            [
                call(MODEL_FOLDER, MODEL_NAME, {"cache": True}),
                call(MODEL_FOLDER, MODEL_NAME, {"cache": False}),
            ],
        )

    def test_cache_failure_retries_without_cache(self):
        model = MagicMock()
        with patch(
            PYMOCA_TRANSFER_MODEL, side_effect=[RuntimeError("stale cache"), model]
        ) as mock_transfer:
            result = load_pymoca_model(MODEL_FOLDER, MODEL_NAME, _opts(cache=True), LOGGER)
        self.assertIs(result, model)
        self.assertEqual(
            mock_transfer.call_args_list,
            [
                call(MODEL_FOLDER, MODEL_NAME, {"cache": True}),
                call(MODEL_FOLDER, MODEL_NAME, {"cache": False}),
            ],
        )

    def test_cache_failure_does_not_mutate_compiler_options(self):
        model = MagicMock()
        original_opts = {"cache": True}
        with patch(PYMOCA_TRANSFER_MODEL, side_effect=[RuntimeError("stale cache"), model]):
            load_pymoca_model(MODEL_FOLDER, MODEL_NAME, original_opts, LOGGER)
        self.assertTrue(original_opts["cache"])

    def test_retry_failure_reraises(self):
        with patch(
            PYMOCA_TRANSFER_MODEL,
            side_effect=[RuntimeError("stale cache"), RuntimeError("compile fail")],
        ):
            with self.assertRaises(RuntimeError):
                load_pymoca_model(MODEL_FOLDER, MODEL_NAME, _opts(cache=True), LOGGER)

    def test_module_not_found_error_retry_failure_reraises(self):
        with patch(
            PYMOCA_TRANSFER_MODEL,
            side_effect=[ModuleNotFoundError("stale cache"), ModuleNotFoundError("still missing")],
        ):
            with self.assertRaises(ModuleNotFoundError):
                load_pymoca_model(MODEL_FOLDER, MODEL_NAME, _opts(cache=True), LOGGER)

    def test_msl_error_on_first_call_raises_runtime_error(self):
        with patch(
            PYMOCA_TRANSFER_MODEL,
            side_effect=RuntimeError("Could not find class 'Modelica.SIunits.Height'"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                load_pymoca_model(MODEL_FOLDER, MODEL_NAME, _opts(cache=False), LOGGER)
        self.assertIn(MODEL_FOLDER, str(ctx.exception))

    def test_msl_error_on_retry_raises_runtime_error(self):
        with patch(
            PYMOCA_TRANSFER_MODEL,
            side_effect=[
                RuntimeError("stale cache"),
                RuntimeError("Could not find class 'SI.VolumeFlowRate'"),
            ],
        ):
            with self.assertRaises(RuntimeError) as ctx:
                load_pymoca_model(MODEL_FOLDER, MODEL_NAME, _opts(cache=True), LOGGER)
        self.assertIn(MODEL_FOLDER, str(ctx.exception))

    def test_default_logger_used_when_not_provided(self):
        model = MagicMock()
        with patch(PYMOCA_TRANSFER_MODEL, side_effect=[RuntimeError("stale cache"), model]):
            result = load_pymoca_model(MODEL_FOLDER, MODEL_NAME, _opts(cache=True))
        self.assertIs(result, model)
