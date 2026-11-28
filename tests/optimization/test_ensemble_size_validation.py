# SPDX-License-Identifier: LGPL-3.0-only

import pytest

from .test_integration import SingleShootingModel


@pytest.mark.parametrize("size", [0, -1])
def test_transcribe_rejects_nonpositive_ensemble_size(size):
    class InvalidEnsemble(SingleShootingModel):
        @property
        def ensemble_size(self):
            return size

    with pytest.raises(ValueError, match=rf"ensemble_size must be >= 1, got {size}"):
        InvalidEnsemble().transcribe()


def test_transcribe_accepts_default_ensemble_size():
    assert len(SingleShootingModel().transcribe()) == 7
