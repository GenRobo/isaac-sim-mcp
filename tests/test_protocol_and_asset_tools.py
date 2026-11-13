"""Protocol helpers and asset-tool scripts without Kit: framing, deadlines, bounds, and that every template compiles
with its arguments embedded as JSON (null/true/false never reach Python as names)."""
import ast
import importlib.util
import json
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "isaac_mcp"))


def load(name, path):
    # The extension package's __init__ imports Kit; the protocol module itself does not.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


protocol = load("protocol", ROOT / "isaac.sim.mcp_extension" / "isaac_sim_mcp_extension" / "protocol.py")
import asset_tools  # noqa: E402


class ProtocolTests(unittest.TestCase):
    def test_bounded_writer_keeps_the_head_and_counts_the_rest(self):
        writer = protocol.BoundedWriter(limit=10)
        writer.write("0123456789abcdef")
        writer.write("more")
        self.assertEqual(writer.getvalue(), "0123456789\n...[10 more characters]")

    def test_json_safe_bounds_results_and_falls_back_to_repr(self):
        self.assertEqual(protocol.json_safe({"a": [1, 2.5, None, True]}), {"a": [1, 2.5, None, True]})
        self.assertIsNone(protocol.json_safe(float("nan")))
        self.assertEqual(protocol.json_safe({"a": float("inf"), "b": (1, 2)}), {"a": None, "b": [1, 2]})
        self.assertEqual(protocol.json_safe(set([1])), [1])
        self.assertIn("object", protocol.json_safe({"o": object()})["o"])
        big = protocol.json_safe(list(range(100000)), limit=1000)
        self.assertIsInstance(big, str)
        self.assertLess(len(big), 1100)
        self.assertIn("truncated", big)

    def test_expired_only_for_a_past_numeric_deadline(self):
        now = time.time()
        self.assertTrue(protocol.expired({"type": "x", "deadline": now - 1}, now))
        self.assertFalse(protocol.expired({"type": "x", "deadline": now + 60}, now))
        self.assertFalse(protocol.expired({"type": "x"}, now))
        self.assertFalse(protocol.expired({"type": "x", "deadline": "soon"}, now))
        self.assertFalse(protocol.expired("not a dict", now))

    def test_frame_is_newline_terminated_json_even_for_unencodable_values(self):
        self.assertEqual(protocol.frame({"status": "success", "result": 1}), b'{"status": "success", "result": 1}\n')
        framed = protocol.frame({"status": "success", "result": float("inf")})
        self.assertTrue(framed.endswith(b"\n"))
        self.assertEqual(json.loads(framed)["status"], "success")

    def test_parse_buffer_waits_for_a_complete_object_and_drops_an_oversized_one(self):
        self.assertEqual(protocol.parse_buffer(b'{"type": "ping"'), (None, b'{"type": "ping"'))
        self.assertEqual(protocol.parse_buffer(b'{"type": "ping"}'), ({"type": "ping"}, b""))
        self.assertEqual(protocol.parse_buffer(b"x" * (protocol.MAX_REQUEST + 1)), (None, b""))


class AssetToolTests(unittest.TestCase):
    def test_every_template_compiles_with_json_arguments_embedded(self):
        for name in asset_tools.TEMPLATES:
            code = asset_tools.script(name, {"path": "/task/a.usd", "lookAt": None, "frames": 2, "flag": True})
            ast.parse(code)
            scope = {}
            exec(code.split("\n", 2)[0] + "\n" + code.split("\n", 2)[1], scope)
            self.assertEqual(scope["gr_args"], {"path": "/task/a.usd", "lookAt": None, "frames": 2, "flag": True})

    def test_scripts_advance_frames_only_through_the_update_helper(self):
        for name, template in asset_tools.TEMPLATES.items():
            self.assertNotIn("app.update()", template, name)
            self.assertIn("update()", template, name)


if __name__ == "__main__":
    unittest.main()
