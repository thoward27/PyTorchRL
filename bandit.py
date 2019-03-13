""" A multi-arm bandit.
"""


class Bandit:
    def __init__(self, arms):
        self.arms = arms

    def pull(self, arm):
        return self.arms[arm]

    def __len__(self):
        return len(self.arms)
