"""Write-path tests for webapp/server.py, run against a temp copy of the repo.

Stdlib unittest; `python3 manage.py test webapp` picks it up, so does
`python3 -m unittest webapp.test_server`. Never touches the real topic files.
"""
import http.client
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerWriteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        # server.py serves the parent of its own folder, so a copied webapp/ serves the temp repo.
        shutil.copytree(HERE, os.path.join(cls.tmp, "webapp"), ignore=shutil.ignore_patterns("__pycache__"))
        os.makedirs(os.path.join(cls.tmp, "concepts"))
        cls.topic = os.path.join(cls.tmp, "concepts", "a.md")
        with open(cls.topic, "w", encoding="utf-8", newline="\n") as f:
            f.write("old\n")
        cls.port = free_port()
        cls.proc = subprocess.Popen([sys.executable, os.path.join(cls.tmp, "webapp", "server.py"),
                                     "--port", str(cls.port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", cls.port), timeout=0.1).close()
                break
            except OSError:
                time.sleep(0.1)
        cls.token = cls.request("GET", "/api/config")[1]["token"]

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.proc.wait(10)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def request(cls, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", cls.port, timeout=10)
        hdrs = {"Host": "localhost:%d" % cls.port, "Content-Type": "application/json", **(headers or {})}
        conn.request(method, path, json.dumps(body).encode() if body is not None else None, hdrs)
        r = conn.getresponse()
        data = json.loads(r.read() or b"null")
        conn.close()
        return r.status, data

    def put(self, body, **headers):
        return self.request("PUT", "/api/file", body, {"X-KB-Token": self.token, **headers})

    def version(self):
        return self.request("GET", "/api/file?path=concepts/a.md")[1]["version"]

    def read(self):
        with open(self.topic, encoding="utf-8") as f:
            return f.read()

    def test_no_token_rejected(self):
        status, _ = self.request("PUT", "/api/file", {"path": "concepts/a.md", "content": "x",
                                                      "version": self.version()})
        self.assertEqual(status, 403)
        self.assertEqual(self.read(), "old\n")

    def test_foreign_origin_rejected(self):
        status, _ = self.put({"path": "concepts/a.md", "content": "x", "version": self.version()},
                             Origin="http://evil.example")
        self.assertEqual(status, 403)
        self.assertEqual(self.read(), "old\n")

    def test_parent_path_rejected(self):
        status, _ = self.put({"path": "../escape.md", "content": "x", "version": "0"})
        self.assertEqual(status, 404)
        status, _ = self.request("POST", "/api/new", {"path": "concepts/../../escape.md", "content": "x"},
                                 {"X-KB-Token": self.token})
        self.assertEqual(status, 400)
        self.assertFalse(os.path.exists(os.path.join(os.path.dirname(self.tmp), "escape.md")))

    def test_stale_version_rejected(self):
        status, _ = self.put({"path": "concepts/a.md", "content": "x", "version": "1"})
        self.assertEqual(status, 409)
        self.assertEqual(self.read(), "old\n")

    def test_valid_save_writes_file(self):
        status, data = self.put({"path": "concepts/a.md", "content": "new\r\nline\n", "version": self.version()})
        self.assertEqual(status, 200)
        self.assertEqual(data["version"], self.version())
        with open(self.topic, "rb") as f:
            self.assertEqual(f.read(), b"new\nline\n")
        # Restore so test order does not matter.
        self.put({"path": "concepts/a.md", "content": "old\n", "version": self.version()})


if __name__ == "__main__":
    unittest.main()
