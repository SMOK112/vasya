"""Catch accidental corruption of the bundled Let's Encrypt PEM.

The original hand-copied certificate was missing a base64 character.
This test ensures all PEM bytes decode correctly before an ESP32 upload.
"""
import base64
from pathlib import Path
import re
import unittest

CERT_FILE = Path(__file__).resolve().parents[1] / "firmware/GlassesBridge/root_ca.h"


class RootCertificateTests(unittest.TestCase):
    def test_pem_base64_is_well_formed(self):
        text = CERT_FILE.read_text(encoding="utf-8")
        match = re.search(
            r"-----BEGIN CERTIFICATE-----\s+([A-Za-z0-9+/=\s]+?)\s+-----END CERTIFICATE-----",
            text)
        self.assertIsNotNone(match)
        encoded = "".join(match.group(1).split())
        decoded = base64.b64decode(encoded, validate=True)
        self.assertGreater(len(decoded), 1000)
        self.assertTrue(decoded.startswith(b"\x30"))
        self.assertIn(b"Internet Security Research Group", decoded)


if __name__ == "__main__":
    unittest.main()
