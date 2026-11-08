import pytest

from gaze_mouse.cursor.controller import Click, PointerOutput
from gaze_mouse.evaluation.target_test import (
    HIT,
    MISS,
    TIMEOUT,
    WRONG_BUTTON,
    TargetTest,
    build_trials,
)
from gaze_mouse.os_io.mouse import DryRunMouse, MouseDriver, in_corner
from gaze_mouse.recording import Target

SECOND = 1_000_000_000
SCREEN = (1920, 1200)


class FakeMouse:
    def __init__(self, position=(500, 500)):
        self._position = position
        self.moves, self.clicks = [], []

    def position(self):
        return self._position

    def move(self, x, y):
        self.moves.append((x, y))

    def click(self, button, x, y):
        self.clicks.append((button, x, y))


@pytest.mark.parametrize("point", [(0, 0), (1919, 0), (0, 1199), (1919, 1199), (5, 5), (1914, 1194)])
def test_points_in_corners(point):
    assert in_corner(point, SCREEN, 5)


@pytest.mark.parametrize("point", [(500, 500), (0, 600), (960, 0), (6, 5), (1913, 1199)])
def test_points_outside_corners(point):
    assert not in_corner(point, SCREEN, 5)


def test_unknown_position_is_not_a_corner():
    assert not in_corner(None, SCREEN, 5)


def test_driver_moves_and_clicks():
    mouse = FakeMouse()
    driver = MouseDriver(mouse, SCREEN, 5, True)
    assert driver.apply(PointerOutput("tracking", (10, 20), Click("left", 30, 40))) is False
    assert mouse.moves == [(10, 20)] and mouse.clicks == [("left", 30, 40)]


def test_driver_sends_nothing_when_mouse_is_in_a_corner():
    mouse = FakeMouse((0, 0))
    driver = MouseDriver(mouse, SCREEN, 5, True)
    assert driver.apply(PointerOutput("tracking", (10, 20), Click("left", 30, 40))) is True
    assert mouse.moves == [] and mouse.clicks == []


def test_corner_pause_can_be_disabled():
    mouse = FakeMouse((0, 0))
    assert MouseDriver(mouse, SCREEN, 5, False).apply(PointerOutput("tracking", (10, 20))) is False
    assert mouse.moves == [(10, 20)]


def test_driver_leaves_cursor_alone_when_frozen():
    mouse = FakeMouse()
    MouseDriver(mouse, SCREEN, 5, True).apply(PointerOutput("frozen"))
    assert mouse.moves == [] and mouse.clicks == []


def test_dry_run_mouse_reports_no_position_and_sends_nothing():
    mouse = DryRunMouse()
    assert mouse.position() is None
    assert mouse.move(1, 2) is None and mouse.click("left", 1, 2) is None


def make_test(count=2) -> TargetTest:
    targets = [Target(0, 500, 500), Target(1, 1000, 800), Target(2, 200, 200)]
    return TargetTest(build_trials(targets, count), button_px=300, timeout_s=10.0)


def test_trials_alternate_buttons():
    assert [t.button for t in build_trials([Target(i, 0, 0) for i in range(4)], 4)] == ["left", "right", "left", "right"]


def test_click_before_start_is_ignored():
    test = make_test()
    assert test.on_click("left", 500, 500, 0) is None
    assert test.results == []


def test_hit_on_button_edge_and_duration():
    test = make_test()
    test.start(0)
    assert test.on_click("left", 650, 350, 2 * SECOND) == HIT
    assert test.results[0].duration_ms == pytest.approx(2000.0)
    assert test.current.target.position_id == 1


def test_miss_just_outside_button():
    test = make_test()
    test.start(0)
    assert test.on_click("left", 651, 500, SECOND) == MISS


def test_wrong_button_on_target():
    test = make_test()
    test.start(0)
    assert test.on_click("right", 500, 500, SECOND) == WRONG_BUTTON


def test_timeout_moves_on_and_restarts_the_clock():
    test = make_test()
    test.start(0)
    test.tick(9 * SECOND)
    assert test.results == []
    test.tick(10 * SECOND)
    assert test.results[0].outcome == TIMEOUT
    assert test.on_click("right", 1000, 800, 12 * SECOND) == HIT
    assert test.results[1].duration_ms == pytest.approx(2000.0)


def test_finishes_and_ignores_later_clicks():
    test = make_test(count=1)
    test.start(0)
    test.on_click("left", 500, 500, SECOND)
    assert test.finished
    assert test.on_click("left", 500, 500, 2 * SECOND) is None


def test_summary_counts():
    test = make_test(count=3)
    test.start(0)
    test.on_click("left", 500, 500, SECOND)
    test.on_click("left", 1000, 800, 3 * SECOND)
    test.on_click("left", 1900, 1100, 4 * SECOND)
    summary = test.summary()
    assert (summary["hit"], summary["wrong_button"], summary["miss"], summary["trials"]) == (1, 1, 1, 3)
    assert summary["hit_rate"] == pytest.approx(1 / 3)
    assert summary["mean_hit_ms"] == pytest.approx(1000.0)
