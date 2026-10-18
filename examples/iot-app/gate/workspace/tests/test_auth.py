import unittest


def may_read_note(owner, subject):
    return owner == subject


class AuthTests(unittest.TestCase):
    def test_owner_can_read(self):
        self.assertTrue(may_read_note("alice", "alice"))

    def test_other_account_is_denied(self):
        self.assertFalse(may_read_note("alice", "bob"))
