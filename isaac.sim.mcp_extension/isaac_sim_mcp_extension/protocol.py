"""
MIT License

Copyright (c) 2023-2025 omni-mcp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""Pure helpers for the extension's socket protocol: no Kit imports, so they are testable anywhere.

Wire format: a client sends one JSON object {"type": <command>, "params": {...}, "deadline": <unix seconds, optional>}
and reads back one JSON object terminated by a newline ({"status": "success", "result": {...}} or
{"status": "error", "message": ...}). Older clients that parse the stream until the JSON is complete keep working: the
trailing newline is whitespace to them.
"""
import io
import json
import time

MAX_STREAM = 64 * 1024
MAX_RESULT = 256 * 1024
MAX_REQUEST = 1024 * 1024


class BoundedWriter(io.TextIOBase):
    """Keeps the first MAX_STREAM characters written; a runaway print cannot grow a reply without bound."""

    def __init__(self, limit=MAX_STREAM):
        self.limit = limit
        self.parts = []
        self.size = 0
        self.dropped = 0

    def writable(self):
        return True

    def write(self, text):
        text = str(text)
        keep = self.limit - self.size
        if keep > 0:
            self.parts.append(text[:keep])
            self.size += min(len(text), keep)
        self.dropped += max(0, len(text) - max(keep, 0))
        return len(text)

    def getvalue(self):
        suffix = f"\n...[{self.dropped} more characters]" if self.dropped else ""
        return "".join(self.parts) + suffix


def json_safe(value, limit=MAX_RESULT):
    """A JSON-encodable copy of `value` (repr for anything else) whose serialisation stays under `limit`."""
    try:
        text = json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        text = repr(value)
        return text[:limit] + ("..." if len(text) > limit else "")
    if len(text) > limit:
        return text[:limit] + f"...[result truncated at {limit} characters]"
    return json.loads(text)


def expired(command, now=None):
    """True when the command carries a Unix-time deadline that has passed (its caller gave up waiting)."""
    deadline = command.get("deadline") if isinstance(command, dict) else None
    if deadline is None:
        return False
    try:
        return (now if now is not None else time.time()) > float(deadline)
    except (TypeError, ValueError):
        return False


def frame(response):
    """Serialise one response for the wire, newline-terminated; unencodable values become their repr."""
    try:
        text = json.dumps(response, allow_nan=False)
    except (TypeError, ValueError):
        text = json.dumps({"status": response.get("status", "success") if isinstance(response, dict) else "success",
                           "message": repr(response)[:MAX_RESULT]})
    return (text + "\n").encode("utf-8")


def parse_buffer(buffer):
    """(command, remaining bytes) once the buffer holds one complete JSON object, else (None, buffer).

    A buffer over MAX_REQUEST that still does not parse is discarded (returns (None, b'')).
    """
    if not buffer:
        return None, buffer
    try:
        command = json.loads(buffer.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        if len(buffer) > MAX_REQUEST:
            return None, b""
        return None, buffer
    return command, b""
