import pytest

from coding_agent.tui.app import CodingAgentApp
from coding_agent.tui.state import TranscriptItem, initial_state
from coding_agent.tui.widgets import TranscriptRow, TranscriptView


class _FakeRuntime:
    status = None

    def subscribe(self, callback):
        return lambda: None


def _make_app() -> CodingAgentApp:
    return CodingAgentApp(
        runtime=_FakeRuntime(),
        initial_state=initial_state("/tmp/project", "fake"),
        branch_detector=lambda workspace: None,
    )


def _assistant(i: int, text: str) -> TranscriptItem:
    return TranscriptItem(kind="assistant", item_id=f"m{i}", text=text)


@pytest.mark.asyncio
async def test_render_state_preserves_settled_rows_on_delta() -> None:
    app = _make_app()
    async with app.run_test():
        view = app.query_one("#transcript", TranscriptView)
        items = [_assistant(0, "hello one"), _assistant(1, "hello two")]
        await view.render_state(items)
        rows = list(view.query(TranscriptRow))
        assert len(rows) == 2
        settled_widget = rows[0]
        settled_renderable = rows[0]._renderable
        changed_renderable = rows[1]._renderable

        # A delta only grows the last row: the settled row must be untouched,
        # while the changed row is refreshed in place with a fresh renderable.
        items[1] = _assistant(1, "hello two plus more text")
        await view.render_state(items)
        rows_after = list(view.query(TranscriptRow))
        assert len(rows_after) == 2
        assert rows_after[0] is settled_widget
        assert rows_after[0]._renderable is settled_renderable
        assert rows_after[1] is rows[1]
        assert rows_after[1]._renderable is not changed_renderable


@pytest.mark.asyncio
async def test_render_state_mounts_only_appended_rows() -> None:
    app = _make_app()
    async with app.run_test():
        view = app.query_one("#transcript", TranscriptView)
        await view.render_state([_assistant(0, "a"), _assistant(1, "b")])
        before = list(view.query(TranscriptRow))
        assert len(before) == 2

        await view.render_state(
            [_assistant(0, "a"), _assistant(1, "b"), _assistant(2, "c")]
        )
        after = list(view.query(TranscriptRow))
        assert len(after) == 3
        assert after[0] is before[0]
        assert after[1] is before[1]
        assert after[2].item.item_id == "m2"


@pytest.mark.asyncio
async def test_render_state_full_replaces_on_material_change() -> None:
    app = _make_app()
    async with app.run_test():
        view = app.query_one("#transcript", TranscriptView)
        await view.render_state([_assistant(0, "a")])
        old = next(iter(view.query(TranscriptRow)))

        # A session switch replaces the id set entirely.
        await view.render_state([_assistant(9, "z")])
        new_rows = list(view.query(TranscriptRow))
        assert len(new_rows) == 1
        assert new_rows[0] is not old


@pytest.mark.asyncio
async def test_render_state_incremental_snapshot_matches_full_render() -> None:
    app = _make_app()
    async with app.run_test():
        view = app.query_one("#transcript", TranscriptView)
        await view.render_state([_assistant(0, "**a**"), _assistant(1, "b")])
        await view.render_state(
            [_assistant(0, "**a**"), _assistant(1, "b + delta"), _assistant(2, "c")]
        )
        incremental_text = view.renderable_text

        # A fresh full render of the same items yields the same snapshot.
        await view.render_state(
            [_assistant(0, "**a**"), _assistant(1, "b + delta"), _assistant(2, "c")]
        )
        assert view.renderable_text == incremental_text


@pytest.mark.asyncio
async def test_render_state_grows_changed_row_height() -> None:
    # An in-place update of a growing message must re-layout the row (refresh
    # with layout=True); a repaint-only refresh would keep the original height
    # and clip streamed content to one line.
    app = _make_app()
    async with app.run_test() as pilot:
        view = app.query_one("#transcript", TranscriptView)
        await view.render_state([_assistant(0, "one line")])
        await pilot.pause()
        row = next(iter(view.query(TranscriptRow)))
        assert row.region.height == 1

        await view.render_state(
            [_assistant(0, "\n".join(f"line {i}" for i in range(20)))]
        )
        await pilot.pause()
        row_after = next(iter(view.query(TranscriptRow)))
        assert row_after.region.height == 20
