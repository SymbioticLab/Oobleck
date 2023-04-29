import unittest
from oobleck.cplanning import test


class TestOobleckPlanning(unittest.TestCase):
    def test(self):
        self.assertEqual(test(), 0)


if __name__ == "__main__":
    unittest.main()
