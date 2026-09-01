import os

from coding_agent.policy.memory import DecisionMemory, signature


def test_signature_normalizes_arguments():
    assert signature("write_file", {"path": "a", "content": "x"}) == (
        "write_file",
        '{"content": "x", "path": "a"}',
    )
    assert signature("write_file", {"content": "x", "path": "a"}) == signature(
        "write_file", {"path": "a", "content": "x"}
    )


def test_remember_and_lookup_scopes(tmp_path):
    mem = DecisionMemory(always_path=tmp_path / "approvals.json")
    sig = signature("run_command", {"command": "ls"})
    assert mem.lookup(sig) is None
    mem.remember(sig, "allow", scope="turn")
    assert mem.lookup(sig) == "allow"
    mem.clear_turn()
    assert mem.lookup(sig) is None
    mem.remember(sig, "deny", scope="session")
    assert mem.lookup(sig) == "deny"
    mem.clear_session()
    assert mem.lookup(sig) is None


def test_always_persists_to_file(tmp_path):
    path = tmp_path / "approvals.json"
    mem = DecisionMemory(always_path=path)
    sig = signature("write_file", {"path": "x"})
    mem.remember(sig, "allow", scope="always")
    mem.persist_always()
    loaded = DecisionMemory(always_path=path)
    loaded.load_always()
    assert loaded.lookup(sig) == "allow"
    assert (os.stat(path).st_mode & 0o777) == 0o600
