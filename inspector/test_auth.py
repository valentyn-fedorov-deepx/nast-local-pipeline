"""Checks for inspector/auth.py. Run from inspector/:  python -m unittest test_auth -v"""
import json
import tempfile
import unittest
from pathlib import Path

import auth

KEY = b"k" * 32
NOW = 1_800_000_000


class Passwords(unittest.TestCase):
    def test_right_password_verifies(self):
        self.assertTrue(auth.verify_password("s3cret", auth.hash_password("s3cret")))

    def test_wrong_password_fails(self):
        self.assertFalse(auth.verify_password("nope", auth.hash_password("s3cret")))

    def test_hash_is_salted(self):
        self.assertNotEqual(auth.hash_password("same"), auth.hash_password("same"))

    def test_stored_form_never_contains_the_password(self):
        self.assertNotIn("s3cret", auth.hash_password("s3cret"))

    def test_malformed_stored_value_fails_instead_of_raising(self):
        self.assertFalse(auth.verify_password("x", "garbage"))
        self.assertFalse(auth.verify_password("x", "md5$1$00$00"))


class Accounts(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / "users.json"

    def test_no_file_means_no_accounts(self):
        self.assertEqual(auth.load_users(self.path), {})

    def test_added_user_can_log_in(self):
        auth.add_user("op", "pw1", self.path)
        self.assertTrue(auth.check_login("op", "pw1", self.path))

    def test_wrong_password_and_unknown_user_both_fail(self):
        auth.add_user("op", "pw1", self.path)
        self.assertFalse(auth.check_login("op", "pw2", self.path))
        self.assertFalse(auth.check_login("nobody", "pw1", self.path))

    def test_file_holds_hashes_only(self):
        auth.add_user("op", "pw1", self.path)
        stored = json.loads(self.path.read_text())
        self.assertTrue(stored["op"].startswith("pbkdf2_sha256$"))
        self.assertNotIn("pw1", self.path.read_text())


class Tokens(unittest.TestCase):
    def token(self, **kw):
        return auth.issue_token("op", now=kw.get("now", NOW), ttl=kw.get("ttl", 3600), key=KEY)

    def test_valid_token_names_its_user(self):
        self.assertEqual(auth.check_token("Bearer " + self.token(), now=NOW + 10, key=KEY), "op")

    def test_scheme_is_case_insensitive(self):
        self.assertEqual(auth.check_token("bearer " + self.token(), now=NOW + 10, key=KEY), "op")

    def test_expired_token_is_refused(self):
        self.assertIsNone(auth.check_token("Bearer " + self.token(), now=NOW + 3600, key=KEY))

    def test_token_from_another_key_is_refused(self):
        self.assertIsNone(auth.check_token("Bearer " + self.token(), now=NOW + 10, key=b"x" * 32))

    def test_edited_claims_are_refused(self):
        payload, tag = self.token().split(".")
        forged = auth._b64(json.dumps({"sub": "admin", "iat": NOW, "exp": NOW + 99999}).encode())
        self.assertIsNone(auth.check_token(f"Bearer {forged}.{tag}", now=NOW + 10, key=KEY))

    def test_missing_or_malformed_header_is_refused(self):
        for header in (None, "", "Bearer", "Basic abc", "Bearer abc", "Bearer a.b.c", "Bearer !!.!!",
                       "Bearer é.é"):
            self.assertIsNone(auth.check_token(header, now=NOW + 10, key=KEY), header)


if __name__ == "__main__":
    unittest.main()
