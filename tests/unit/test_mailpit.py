from unittest.mock import call, patch

import httpx
import pytest

from utilities import mailpit


@pytest.fixture
def mailpit_api(monkeypatch, tmp_path):
    monkeypatch.setattr(
        mailpit, "open", lambda _path, mode: (tmp_path / "release.lock").open(mode),
        raising=False,
    )
    with patch.object(mailpit, "request") as api:
        yield api


def search_results(*entries):
    return httpx.Response(200, json={"messages": entries})


def test_find_message_ignores_partial_message_id_matches(mailpit_api):
    partial = {"ID": "other", "MessageID": "another-complaint@demo.test"}
    exact = {"ID": "central-id", "MessageID": "complaint@demo.test"}
    mailpit_api.return_value = search_results(partial, exact)

    assert mailpit.find_message("central", "<complaint@demo.test>") == exact
    mailpit_api.assert_called_once_with(
        "central", "GET", "search", params={"query": "message-id:complaint@demo.test"},
    )


def test_pending_messages_skip_current_and_legacy_review_copies():
    pending = {"ID": "pending-id", "MessageID": "pending@demo.test"}
    central = [
        {"ID": "current-id", "MessageID": "current@demo.test"},
        pending,
        {"ID": "older-id", "MessageID": "older@demo.test"},
    ]
    review = [
        {"MessageID": "current@demo.test"},
        {"MessageID": "route-older-id-review@demo.test"},
    ]

    inboxes = {"central": central, "review": review}
    with patch.object(mailpit, "messages", side_effect=lambda mailbox: iter(inboxes[mailbox])):
        assert list(mailpit.pending_messages()) == [pending]


@pytest.mark.parametrize("message_id", ["latest", "../other"])
def test_forward_rejects_ids_that_do_not_identify_a_specific_message(mailpit_api, message_id):
    with pytest.raises(ValueError, match="database ID"):
        mailpit.forward_to_review(message_id)

    mailpit_api.assert_not_called()


@pytest.mark.parametrize("legacy", [False, True], ids=["preserved-id", "legacy-id"])
def test_forward_reuses_an_existing_review_copy(mailpit_api, legacy):
    if legacy:
        mailpit_api.side_effect = [
            search_results({"ID": "review-id", "MessageID": "route-central-id-review@demo.test"}),
        ]
    else:
        mailpit_api.side_effect = [
            search_results(),
            httpx.Response(200, json={"Message-Id": ["<complaint@demo.test>"]}),
            search_results({"ID": "review-id", "MessageID": "complaint@demo.test"}),
        ]

    assert mailpit.forward_to_review("central-id") == "review-id"
    assert all(request.args[1] == "GET" for request in mailpit_api.call_args_list)


def test_forward_releases_through_mailpit_and_finds_copy_by_original_id(mailpit_api):
    mailpit_api.side_effect = [
        search_results(),
        httpx.Response(200, json={"Message-Id": ["<complaint@demo.test>"]}),
        search_results(),
        httpx.Response(200),
        search_results({"ID": "review-id", "MessageID": "complaint@demo.test"}),
    ]

    assert mailpit.forward_to_review("central-id") == "review-id"
    assert mailpit_api.call_args_list[-2:] == [
        call("central", "POST", "message/central-id/release", json={"To": ["review@demo.test"]}),
        call("review", "GET", "search", params={"query": "message-id:complaint@demo.test"}),
    ]
