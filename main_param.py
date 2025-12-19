import sys
import json
import nltk
import docx
import re
from pathlib import Path
from collections import Counter
from langdetect import detect, DetectorFactory
import dateparser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

from llama_cpp import Llama  # GGUF / llama.cpp backend
from helper import process_meetings

DetectorFactory.seed = 0
nltk.download("punkt", quiet=False)

class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj, status=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path != "/process":
            self._send_json({"error": "Not found"}, status=404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            self._send_json({"error": "Invalid JSON"}, status=400)
            return

        try:
            input_folder = payload["input_folder"]
            output_folder = payload["output_folder"]
            prev = int(payload.get("previous_count", 0))
            curr = int(payload.get("current_count", 0))
        except KeyError as e:
            self._send_json({"error": f"Missing field {e}"}, status=400)
            return

        try:
            processed = process_meetings(input_folder, output_folder, prev, curr)
            self._send_json({"status": "ok", "processed": processed})
        except Exception as e:
            self._send_json({"status": "error", "message": str(e)}, status=500)


def run_server(port=8000):
    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"Server listening on http://127.0.0.1:{port}/process")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
