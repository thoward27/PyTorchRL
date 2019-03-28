from unittest import TestCase

from hypothesis import strategies as st, given

from agent import *
from bandit import *

ARM = st.integers(min_value=0, max_value=100)


class TestAgent(TestCase):
    def test_init(self):
        a = Agent(in_dim=1, out_dim=4)
        self.assertIsInstance(a, Agent)
        return

    def test_forward(self):
        a = Agent(in_dim=1, out_dim=1)
        with self.assertRaises(AttributeError):
            a.forward(None)
        a.forward(randn(1, 1))
        return

    def test_predict(self):
        a = Agent(in_dim=1, out_dim=4)
        out = a.predict(randn(1, 1))
        self.assertIs(type(out), int)
        return

    def test_loss(self):
        b = Bandit([0, 0, 10, 0])
        a = Agent(in_dim=1, out_dim=len(b))

        context = randn(1, 1, requires_grad=True)
        predictions = a.forward(context)
        predictions[0][2] = b.pull(2)

        a.optimizer.zero_grad()
        loss = a.criterion(a.forward(context), predictions)
        loss.backward()
        a.optimizer.step()

        self.assertEqual(predictions[0][0], a.forward(context)[0][0])
        return

    def test_fit(self):
        b = Bandit([0, 0, 10, 0])
        a = Agent(in_dim=1, out_dim=len(b))

        context = randn(1, 1)
        arm = 2

        corrected = a.forward(context)
        corrected[0][arm] = b.pull(arm)

        a.fit(context, corrected)

        self.assertNotEqual(corrected[0][arm], a.forward(context)[0][arm])
        self.assertEqual(corrected[0][0], a.forward(context)[0][0])
        self.assertEqual(corrected[0][1], a.forward(context)[0][1])
        return

    def test_play(self):
        b = Bandit([0.1, 0.9, 0.4])
        a = Agent(in_dim=1, out_dim=len(b))
        arm = a.predict(None)
        self.assertLessEqual(arm, len(b))
        return

    @given(st.lists(ARM, min_size=2, max_size=10, unique=True))
    def test_learn(self, arms):
        b = Bandit(arms)
        a = Agent(in_dim=len(b.context), out_dim=len(b))
        a.learn(b, epochs=100)
        self.assertEqual(b.arms.index(max(b.arms)), a.predict(b.context))
        return
