"""
Simple early stopping helper for validation metrics.
"""


class EarlyStopping:
    """
    Stop training when the monitored metric does not improve for `patience` epochs.
    """

    def __init__(self, patience: int = 10, delta: float = 0.001, mode: str = "max"):
        assert mode in ("max", "min"), "mode must be 'max' or 'min'."
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.delta
        else:
            improved = score < self.best_score - self.delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def __repr__(self) -> str:
        return (
            f"EarlyStopping(patience={self.patience}, delta={self.delta}, "
            f"mode={self.mode}, counter={self.counter}/{self.patience})"
        )
