import unittest
from cplanning import test


class TestOobleckPlanning(unittest.TestCase):
    def test(self):
        self.assertEqual(test(), 42)


if __name__ == "__main__":
    unittest.main()
