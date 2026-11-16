import numpy as np
import pytest

from gaze_mouse.frame import FrameResult
from gaze_mouse.gaze.model import (
    GazeModelAdapter,
    L2CSOnnxGazeBackend,
    OnnxGazeBackend,
    angles_to_direction,
    apply_prediction,
    l2cs_angles_to_direction,
)


def test_angles_to_direction_points_forward_at_zero():
    assert np.allclose(angles_to_direction(0.0, 0.0), [0.0, 0.0, 1.0])


def test_l2cs_angles_point_toward_screen_at_zero():
    assert np.allclose(l2cs_angles_to_direction(0.0, 0.0), [0.0, 0.0, -1.0])


def test_model_adapter_converts_per_eye_angles_and_uncertainty():
    def backend(_image):
        return np.array([[0.0, 0.0, 0.1], [0.0, np.pi / 4, 0.3]])

    prediction = GazeModelAdapter(backend).predict(np.zeros((224, 224, 3), np.uint8))
    assert np.allclose(prediction.left_direction_camera, [0.0, 0.0, 1.0])
    assert np.allclose(prediction.right_direction_camera, [np.sqrt(0.5), 0.0, np.sqrt(0.5)])
    assert prediction.uncertainty_rad == pytest.approx(0.2)


@pytest.mark.parametrize("output", [np.zeros((3, 3)), np.zeros((2, 2))])
def test_model_adapter_rejects_wrong_backend_shape(output):
    with pytest.raises(ValueError, match="shape"):
        GazeModelAdapter(lambda _image: output).predict(np.zeros((10, 10), np.uint8))


def test_model_adapter_rejects_negative_uncertainty():
    with pytest.raises(ValueError, match="negative"):
        GazeModelAdapter(lambda _image: np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 0.0]])).predict(
            np.zeros((10, 10), np.uint8)
        )


def test_apply_prediction_writes_frame_contract():
    def backend(_image):
        return np.zeros((2, 3))

    prediction = GazeModelAdapter(backend).predict(np.zeros((10, 10), np.uint8))
    result = FrameResult(0, 0, 10, 10)
    apply_prediction(result, prediction)
    assert np.allclose(result.left_gaze_direction_camera, [0.0, 0.0, 1.0])
    assert result.gaze_uncertainty == 0.0


def test_onnx_backend_preprocesses_rgb_nchw(monkeypatch, tmp_path):
    captured = {}

    class Input:
        name = "image"

    class Session:
        def get_inputs(self):
            return [Input()]

        def run(self, _outputs, inputs):
            captured.update(inputs)
            return [np.zeros((2, 3), dtype=np.float32)]

    class Ort:
        InferenceSession = staticmethod(lambda _path, providers: Session())

    monkeypatch.setattr("importlib.import_module", lambda name: Ort())
    backend = OnnxGazeBackend(tmp_path / "gaze.onnx")
    output = backend(np.full((2, 3, 3), 255, dtype=np.uint8))
    assert output.shape == (2, 3)
    tensor = captured["image"]
    assert tensor.shape == (1, 3, 2, 3)
    assert np.allclose(tensor[0, :, 0, 0], 1.0)


def test_l2cs_backend_decodes_logits_and_duplicates_prediction(monkeypatch, tmp_path):
    captured = {}

    class Input:
        name = "image"

    class Session:
        def get_inputs(self):
            return [Input()]

        def run(self, _outputs, inputs):
            captured.update(inputs)
            yaw = np.full((1, 90), -100.0, dtype=np.float32)
            pitch = np.full((1, 90), -100.0, dtype=np.float32)
            yaw[0, 45] = 100.0
            pitch[0, 45] = 100.0
            return [yaw, pitch]

    class Ort:
        InferenceSession = staticmethod(lambda _path, providers: Session())

    monkeypatch.setattr("importlib.import_module", lambda name: Ort())
    backend = L2CSOnnxGazeBackend(tmp_path / "gaze.onnx")
    output = backend(np.zeros((12, 16, 3), dtype=np.uint8))

    assert output.shape == (2, 3)
    assert np.allclose(output[:, :2], 0.0, atol=1e-5)
    assert np.all(output[:, 2] == 0.0)
    assert captured["image"].shape == (1, 3, 448, 448)
