from unittest import TestCase

from multiarm_bandit import Bandit


class TestBandit(TestCase):
    def test_init(self):
        b = Bandit(arms=[0.1, 0.2, 0.3, 0.4, 0.1, 0.9, 0.2])
        return

    def test_pull(self):
        b = Bandit(arms=[0.1, 0.9, 0.3])
        reward = b.pull(0)
        self.assertEqual(reward, 0.1)
        return

    def test_len(self):
        b = Bandit(arms=[0.1, 0.2, 0.3])
        self.assertEqual(len(b), 3)
        return

    def test_pull_invalid(self):
        b = Bandit([0.1, 0.9])
        with self.assertRaises(IndexError):
            b.pull(3)
        return
