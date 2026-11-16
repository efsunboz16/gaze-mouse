from gaze_mouse.config import load_config


def test_l2cs_gaze_model_is_configured_by_default():
    config = load_config()
    assert config["gaze_model"]["enabled"] is True
    assert config["gaze_model"]["path"] == "models/gaze.onnx"
    assert config["gaze_model"]["backend"] == "l2cs"
