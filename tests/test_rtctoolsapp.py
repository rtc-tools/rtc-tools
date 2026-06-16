import tempfile
import unittest
import urllib.request
import zipfile
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import rtctools
from rtctools import rtctoolsapp


class TestDownloadExamples(unittest.TestCase):
    def _run_download_examples(self, version, zip_entries, capture_logs=False):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            download_dir = tmp_path / "downloads"
            download_dir.mkdir()

            zip_path = tmp_path / "repo.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                for archive_name, content in zip_entries:
                    zf.writestr(archive_name, content)

            requested_urls = []

            def fake_urlretrieve(url):
                requested_urls.append(url)
                return str(zip_path), None

            def fake_exit(message=0):
                raise SystemExit(message)

            with (
                patch.object(rtctools, "__version__", version),
                patch.object(urllib.request, "build_opener", return_value=object()),
                patch.object(urllib.request, "install_opener", return_value=None),
                patch.object(urllib.request, "urlretrieve", side_effect=fake_urlretrieve),
                patch.object(rtctoolsapp.sys, "exit", side_effect=fake_exit),
            ):
                if capture_logs:
                    log_context = self.assertLogs("rtctools", level="INFO")
                else:
                    log_context = nullcontext(None)

                with log_context as logs:
                    with self.assertRaises(SystemExit) as ctx:
                        rtctoolsapp.download_examples(str(download_dir))

            return {
                "exception": str(ctx.exception),
                "requested_urls": requested_urls,
                "zip_removed": not zip_path.exists(),
                "example_exists": (download_dir / "rtc-tools-examples" / "example.txt").exists(),
                "logs": "\n".join(logs.output) if capture_logs else None,
            }

    def test_download_examples_uses_release_version_and_logs(self):
        result = self._run_download_examples(
            version="2.7.2+9.g0f23240",
            zip_entries=[("rtc-tools-rtc-tools-abc123/examples/example.txt", "hello")],
            capture_logs=True,
        )

        self.assertIn("Successfully downloaded the RTC-Tools examples", result["exception"])
        self.assertEqual(
            result["requested_urls"], ["https://github.com/rtc-tools/rtc-tools/zipball/2.7.2"]
        )
        self.assertTrue(result["zip_removed"])
        self.assertTrue(result["example_exists"])
        self.assertIn(
            (
                "Using RTC-Tools version 2.7.2 "
                "(resolved from 2.7.2+9.g0f23240) to download examples."
            ),
            result["logs"],
        )

    def test_download_examples_uses_release_version_verbatim_when_already_released(self):
        result = self._run_download_examples(
            version="2.7.3",
            zip_entries=[("rtc-tools-rtc-tools-2.7.3/examples/example.txt", "hello")],
            capture_logs=True,
        )

        self.assertIn("Successfully downloaded the RTC-Tools examples", result["exception"])
        self.assertEqual(
            result["requested_urls"], ["https://github.com/rtc-tools/rtc-tools/zipball/2.7.3"]
        )
        self.assertTrue(result["zip_removed"])
        self.assertTrue(result["example_exists"])
        self.assertIn(
            "Using RTC-Tools version 2.7.3 to download examples.",
            result["logs"],
        )

    def test_download_examples_resolves_post_dev_local_version_to_prerelease_tag(self):
        result = self._run_download_examples(
            version="2.8.0a2.post1.dev63+g94b832a.d20260616",
            zip_entries=[("rtc-tools-rtc-tools-2.8.0a2/examples/example.txt", "hello")],
            capture_logs=True,
        )

        self.assertIn("Successfully downloaded the RTC-Tools examples", result["exception"])
        self.assertEqual(
            result["requested_urls"], ["https://github.com/rtc-tools/rtc-tools/zipball/2.8.0a2"]
        )
        self.assertTrue(result["zip_removed"])
        self.assertTrue(result["example_exists"])
        self.assertIn(
            (
                "Using RTC-Tools version 2.8.0a2 "
                "(resolved from 2.8.0a2.post1.dev63+g94b832a.d20260616) to download examples."
            ),
            result["logs"],
        )

    def test_download_examples_handles_empty_archive(self):
        result = self._run_download_examples(version="2.7.3", zip_entries=[])

        self.assertEqual(
            result["requested_urls"], ["https://github.com/rtc-tools/rtc-tools/zipball/2.7.3"]
        )
        self.assertIn("Downloaded archive is malformed or empty.", result["exception"])
        self.assertTrue(result["zip_removed"])
