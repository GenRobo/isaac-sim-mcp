"""The extension's command path with Kit stubbed out (omni, carb, pxr, isaacsim, numpy as MagicMock modules):
external-driver mode, the update() helper, deadlines, error passthrough with traceback and captured output,
SystemExit from a script, and per-element JSON-safe results."""
import sys
import time
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent


def stub_kit():
    """Install MagicMock modules for everything the extension imports from Kit; keep real omni.ext.IExt semantics."""
    names = ["carb", "carb.settings", "omni", "omni.usd", "omni.kit", "omni.kit.app", "omni.kit.commands", "omni.timeline",
             "omni.ext", "pxr", "numpy", "isaacsim", "isaacsim.core", "isaacsim.core.experimental",
             "isaacsim.core.experimental.utils", "isaacsim.core.experimental.utils.stage", "isaacsim.core.experimental.prims",
             "isaacsim.storage", "isaacsim.storage.native"]
    modules = {}
    for name in names:
        module = mock.MagicMock(name=name)
        module.__name__ = name
        modules[name] = module
    class IExt:
        pass
    modules["omni.ext"].IExt = IExt
    modules["omni"].ext = modules["omni.ext"]
    modules["omni"].kit = modules["omni.kit"]
    modules["omni.kit"].app = modules["omni.kit.app"]
    return modules


class ExtensionStubbedTests(unittest.TestCase):
    def setUp(self):
        self.modules = stub_kit()
        self.patcher = mock.patch.dict(sys.modules, self.modules)
        self.patcher.start()
        sys.path.insert(0, str(ROOT / "isaac.sim.mcp_extension"))
        for name in list(sys.modules):
            if name.startswith("isaac_sim_mcp_extension"):
                del sys.modules[name]
        from isaac_sim_mcp_extension import extension  # noqa: E402
        self.extension = extension
        self.ext = extension.MCPExtension()

    def tearDown(self):
        sys.path.remove(str(ROOT / "isaac.sim.mcp_extension"))
        self.patcher.stop()

    def run_command(self, command):
        """_process_command against a fake client; returns the parsed framed reply."""
        import json
        client = mock.MagicMock()
        self.ext._process_command(client, command)
        payload = client.sendall.call_args.args[0]
        self.assertTrue(payload.endswith(b"\n"))
        return json.loads(payload)

    def test_results_output_and_namespace(self):
        reply = self.run_command({"type": "execute_script", "params": {"code": "print('hi')\nresult = {'n': 1 + 1}"}})
        self.assertEqual(reply["status"], "success")
        self.assertEqual(reply["result"]["result"], {"n": 2})
        self.assertEqual(reply["result"]["output"], "hi\n")

    def test_errors_keep_traceback_and_captured_output_on_the_wire(self):
        reply = self.run_command({"type": "execute_script", "params": {"code": "print('before')\nraise ValueError('boom')"}})
        self.assertEqual((reply["status"], reply["message"]), ("error", "boom"))
        self.assertIn("ValueError: boom", reply["traceback"])
        self.assertEqual(reply["output"], "before\n")

    def test_system_exit_from_a_script_is_reported_not_raised(self):
        reply = self.run_command({"type": "execute_script", "params": {"code": "raise SystemExit(3)"}})
        self.assertEqual(reply["status"], "error")
        self.assertIn("SessionInterrupted", reply["message"])

    def test_update_is_refused_in_the_default_mode_and_advances_in_external_mode(self):
        reply = self.run_command({"type": "execute_script", "params": {"code": "update()"}})
        self.assertEqual(reply["status"], "error")
        self.assertIn("driven externally", reply["message"])
        self.assertIs(self.ext.drive_externally(), self.ext)
        self.assertIsNone(self.ext._update_sub)
        reply = self.run_command({"type": "execute_script", "params": {"code": "update()\nupdate()\nresult = 'advanced'"}})
        self.assertEqual(reply["result"]["result"], "advanced")
        self.assertEqual(self.modules["omni.kit.app"].get_app.return_value.update.call_count, 2)

    def test_an_expired_command_is_refused_and_a_running_loop_stops_at_its_deadline(self):
        reply = self.run_command({"type": "execute_script", "params": {"code": "result = 1"}, "deadline": time.time() - 1})
        self.assertEqual(reply["status"], "error")
        self.assertIn("deadline passed before execution", reply["message"])
        self.ext.drive_externally()
        code = "import time\nfor i in range(1000):\n    time.sleep(0.01)\n    update()\nresult = 'finished'"
        reply = self.run_command({"type": "execute_script", "params": {"code": code}, "deadline": time.time() + 0.1})
        self.assertEqual(reply["status"], "error")
        self.assertIn("deadline passed", reply["message"])

    def test_results_are_json_safe_per_element(self):
        reply = self.run_command({"type": "execute_script",
                                  "params": {"code": "result = {'prim': object(), 'n': 3, 'bad': float('nan'), 'items': [1, float('inf')]}"}})
        result = reply["result"]["result"]
        self.assertEqual(result["n"], 3)
        self.assertIsNone(result["bad"])
        self.assertEqual(result["items"], [1, None])
        self.assertIn("object", result["prim"])

    def test_the_queue_is_serviced_one_command_at_a_time_from_the_driver(self):
        self.ext.drive_externally()
        client = mock.MagicMock()
        self.ext._command_queue.put((client, {"type": "execute_script", "params": {"code": "result = 'a'"}}))
        self.ext._command_queue.put((client, {"type": "execute_script", "params": {"code": "result = 'b'"}}))
        self.assertEqual(self.ext.process_pending_commands(max_commands=1), 1)
        self.assertEqual(self.ext.process_pending_commands(max_commands=1), 1)
        self.assertEqual(self.ext.process_pending_commands(max_commands=1), 0)
        self.assertEqual(client.sendall.call_count, 2)


if __name__ == "__main__":
    unittest.main()
