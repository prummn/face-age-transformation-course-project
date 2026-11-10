"""Meaningful API/geometry/session tests, with an explicit non-model test engine.

Run: FACE_AGE_SKIP_INIT=1 python -m unittest discover -s integration/tests -p test_backend.py -v
These tests do not establish model quality or claim to run Ascend inference.
"""
import importlib.util
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ["FACE_AGE_SKIP_INIT"] = "1"
import numpy as np
from PIL import Image
from fastapi import HTTPException
from fastapi.testclient import TestClient

module_path = Path(__file__).resolve().parents[1] / "backend.py"
sys.path.insert(0, str(module_path.parent))
spec = importlib.util.spec_from_file_location("face_age_backend", module_path)
backend = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = backend
spec.loader.exec_module(backend)


class GeometryTests(unittest.TestCase):
    def test_reference_samples_original_without_encoder_resolution_bottleneck(self):
        import cv2
        engine = backend.SamePersonEngine.__new__(backend.SamePersonEngine)
        engine.cv2 = cv2
        original = np.zeros((1024, 1024, 3), dtype=np.uint8)
        original[:, ::2] = 255
        # The corresponding encoder image would lose these one-pixel stripes.
        reference = engine.display_reference(original, {"matrix": [.25, 0, 0, .25, -.375, -.375]})
        np.testing.assert_array_equal(reference, original)

    def test_selection_never_falls_back_to_largest_person(self):
        selected = backend.SelectedFace(x=.05, y=.15, w=.2, h=.3)
        small = np.array([60, 170, 160, 250] + [0] * 10 + [.99], dtype=float)
        large = np.array([500, 100, 420, 700] + [0] * 10 + [.99], dtype=float)
        actual = backend.match_selected_face(np.stack([large, small]), selected, 1000, 1000)
        np.testing.assert_array_equal(actual, small)
        with self.assertRaises(HTTPException) as error:
            backend.match_selected_face(np.stack([large]), selected, 1000, 1000)
        self.assertEqual(error.exception.status_code, 422)

    def test_ffhq_mapping_matches_training_quad_and_round_trips(self):
        points = np.array([[300, 250], [220, 240], [260, 290], [285, 325], [230, 320]], dtype=float)
        matrix = backend.ffhq_affine(points)
        el, er = sorted(points[:2], key=lambda p: p[0])
        ml, mr = sorted(points[3:], key=lambda p: p[0])
        eye = (el + er) * .5
        e2e, e2m = er - el, (ml + mr) * .5 - eye
        x = e2e - np.flipud(e2m) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*e2e) * 2, np.hypot(*e2m) * 1.8)
        y = np.flipud(x) * [-1, 1]
        center = eye + e2m * .1
        quad = np.stack([center-x-y, center-x+y, center+x+y, center+x-y])
        homogeneous = np.column_stack([quad, np.ones(4)])
        mapped = (matrix @ homogeneous.T).T
        np.testing.assert_allclose(mapped, [[0, 0], [0, 256], [256, 256], [256, 0]], atol=1e-10)
        full = np.vstack([matrix, [0, 0, 1]])
        roundtrip = (np.linalg.inv(full) @ np.column_stack([mapped, np.ones(4)]).T).T
        np.testing.assert_allclose(roundtrip[:, :2], quad, atol=1e-10)
        a, b, c, d, e, f = backend.canvas_matrix(matrix)
        np.testing.assert_allclose([[a, c, e], [b, d, f]], matrix)


class StoreTests(unittest.TestCase):
    def test_expiration_cancels_queued_reference_and_releases_capacity(self):
        now = [0]
        store = backend.SessionStore(ttl=10, capacity=1, clock=lambda: now[0])
        first = store.create(np.zeros((256, 256, 3), dtype=np.uint8))
        held = store.get(first)
        with self.assertRaises(HTTPException) as full:
            store.create(held.aligned)
        self.assertEqual(full.exception.status_code, 429)
        now[0] = 11
        second = store.create(held.aligned)
        self.assertNotEqual(first, second)
        self.assertTrue(held.cancelled)
        with self.assertRaises(HTTPException) as late:
            store.check(first, held)
        self.assertEqual(late.exception.status_code, 410)

    def test_delete_invalidates_inflight_reference(self):
        store = backend.SessionStore()
        token = store.create(np.zeros((256, 256, 3), dtype=np.uint8))
        held = store.get(token)
        store.delete(token)
        store.delete(token)
        with self.assertRaises(HTTPException):
            store.check(token, held)


class FakeEngine:
    ready = True
    error = None
    def ensure_ready(self):
        pass
    def prepare(self, rgb, face):
        return np.zeros((256, 256, 3), dtype=np.uint8), {"matrix": [1,0,0,1,0,0], "inverse": [1,0,0,1,0,0], "sourceWidth": rgb.shape[1], "sourceHeight": rgb.shape[0]}, 37, np.array([10, 10, 100, 100])
    def transform(self, item, age, enhanced, assert_active):
        assert_active()
        result = np.full((1024, 1024, 3), 128, dtype=np.uint8)
        return result, result, .123, "native", None


class APIContractTests(unittest.TestCase):
    def setUp(self):
        self.engine_patch = patch.object(backend, "engine", FakeEngine())
        self.engine_patch.start()
        backend.store.clear()
        self.client = TestClient(backend.app)
        self.client.__enter__()
        self.image = backend.image_data_url(np.zeros((256, 256, 3), dtype=np.uint8))

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.engine_patch.stop()
        backend.store.clear()

    def prepare(self):
        return self.client.post("/api/prepare", json={"image": self.image, "face": {"x": .1, "y": .1, "w": .5, "h": .5, "id": "person-2"}})

    def test_prepare_transform_clear_lifecycle(self):
        response = self.prepare()
        self.assertEqual(response.status_code, 200)
        value = response.json()
        self.assertEqual(value["age"]["source"], "model")
        self.assertIsNone(value["age"]["regression"])
        self.assertEqual(value["alignment"], "sam_ffhq")
        self.assertEqual(response.headers["cache-control"], "no-store")
        token = value["session_id"]
        generated = self.client.post("/api/transform", json={"session_id": token, "target_age": 60, "request_id": "t-1"})
        self.assertEqual(generated.status_code, 200)
        output = generated.json()
        self.assertEqual((output["width"], output["height"]), (1024, 1024))
        self.assertEqual(output["meta"]["output_kind"], "native")
        self.assertIn("raw_image", output)
        self.assertIn("color_preservation", output["meta"])
        self.assertEqual(backend.decode_image(output["image"]).shape, (1024, 1024, 3))
        self.assertEqual(self.client.delete("/api/sessions/" + token).status_code, 200)
        self.assertEqual(self.client.post("/api/transform", json={"session_id": token, "target_age": 60}).status_code, 410)

    def test_age_is_strict_integer_and_validation_is_chinese(self):
        token = self.prepare().json()["session_id"]
        for invalid in [4, 91, 12.5, "30", True]:
            response = self.client.post("/api/transform", json={"session_id": token, "target_age": invalid})
            self.assertEqual(response.status_code, 422, invalid)
            self.assertIn("整数", response.json()["detail"])

    def test_invalid_file_and_body_limit(self):
        invalid = self.client.post("/api/prepare", json={"image": "data:image/png;base64,invalid", "face": {"x": 0, "y": 0, "w": .5, "h": .5}})
        self.assertEqual(invalid.status_code, 422)
        with patch.object(backend, "MAX_REQUEST_BYTES", 100):
            self.assertEqual(self.prepare().status_code, 413)

    def test_late_inference_result_is_discarded_after_delete(self):
        token = self.prepare().json()["session_id"]
        original = backend.engine.transform
        def deleted_inflight(*args):
            output = original(*args)
            backend.store.delete(token)
            return output
        with patch.object(backend.engine, "transform", deleted_inflight):
            response = self.client.post("/api/transform", json={"session_id": token, "target_age": 60})
        self.assertEqual(response.status_code, 410)
        self.assertNotIn("image", response.json())


if __name__ == "__main__":
    unittest.main()
