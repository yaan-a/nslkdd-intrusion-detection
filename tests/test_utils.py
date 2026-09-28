import unittest

from utils import to_binary_label


class TestToBinaryLabel(unittest.TestCase):
    def test_normal_is_zero(self):
        self.assertEqual(to_binary_label("normal"), 0)

    def test_normal_is_case_insensitive(self):
        self.assertEqual(to_binary_label("Normal"), 0)
        self.assertEqual(to_binary_label("NORMAL"), 0)

    def test_dos_attack_is_one(self):
        self.assertEqual(to_binary_label("neptune"), 1)

    def test_r2l_attack_is_one(self):
        self.assertEqual(to_binary_label("guess_passwd"), 1)


if __name__ == "__main__":
    unittest.main()
