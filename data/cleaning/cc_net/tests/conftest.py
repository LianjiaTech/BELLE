# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest


def _request_is_disabled(self, *args, **kwargs):
    raise Exception(
        f"Your code tried to call 'request' with: {args}, {kwargs}. Unit test aren't allowed to reach internet."
    )


@pytest.fixture(autouse=True)
def no_requests(monkeypatch):
    """Remove requests.sessions.Session.request for all tests."""
    monkeypatch.setattr("requests.sessions.Session.request", _request_is_disabled)
