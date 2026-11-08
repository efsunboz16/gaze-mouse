import math

from gaze_mouse.perception.landmarks import StrictMsClock
from gaze_mouse.profiling import RateMeter, StageProfiler

MS = 1_000_000


def test_rate_meter_counts_dropped_frames_through_cumulative_count():
    meter = RateMeter(window_s=1.0)
    for i in range(10):
        meter.update(i * 100 * MS, i * 3)  # 3 events per 100 ms
    assert math.isclose(meter.rate(), 30.0, abs_tol=1e-6)


def test_rate_meter_forgets_samples_outside_window():
    meter = RateMeter(window_s=1.0)
    meter.update(0, 0)
    meter.update(100 * MS, 100)  # burst that must age out
    for i in range(2, 40):
        meter.update(i * 100 * MS, 100 + (i - 1))
    assert math.isclose(meter.rate(), 10.0, abs_tol=1e-6)


def test_rate_meter_needs_two_samples():
    meter = RateMeter(window_s=1.0)
    assert meter.rate() == 0.0
    meter.update(0, 0)
    assert meter.rate() == 0.0


def test_stage_profiler_window_and_percentiles():
    profiler = StageProfiler(window=3)
    for value in (100.0, 1.0, 2.0, 3.0):
        profiler.add("x", value)
    summary = profiler.summary()["x"]
    assert summary["n"] == 3
    assert summary["p50"] == 2.0
    assert profiler.percentile("x", 50) == 2.0


def test_stage_profiler_unknown_stage_is_nan():
    assert math.isnan(StageProfiler(window=3).percentile("missing", 50))


def test_strict_clock_never_repeats_within_a_millisecond():
    clock = StrictMsClock()
    values = [clock.to_ms(t) for t in (5 * MS, 5 * MS + 10, 5 * MS + 20, 40 * MS)]
    assert values == [1, 2, 3, 35]


def test_strict_clock_survives_timestamp_going_backwards():
    clock = StrictMsClock()
    values = [clock.to_ms(t) for t in (100 * MS, 150 * MS, 120 * MS)]
    assert values == sorted(set(values))
