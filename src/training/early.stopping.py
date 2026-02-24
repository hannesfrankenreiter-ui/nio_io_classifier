"""
src/training/early_stopping.py
Einfaches Early-Stopping nach Val-Accuracy (mode='max') oder Val-Loss (mode='min').
"""


class EarlyStopping:
    """
    Bricht das Training ab, wenn sich die überwachte Metrik
    für `patience` Epochen nicht um mindestens `delta` verbessert.

    Args:
        patience (int)  : Wartezeit in Epochen ohne Verbesserung
        delta    (float): Minimale Verbesserung, die zählt
        mode     (str)  : 'max' (z.B. Accuracy) | 'min' (z.B. Loss)
    """

    def __init__(self, patience: int = 10, delta: float = 0.001, mode: str = "max"):
        assert mode in ("max", "min"), "mode muss 'max' oder 'min' sein."
        self.patience   = patience
        self.delta      = delta
        self.mode       = mode
        self.best_score = None
        self.counter    = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """
        Registriert eine neue Metrik.
        Gibt True zurück, wenn das Training gestoppt werden soll.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.delta
        else:
            improved = score < self.best_score - self.delta

        if improved:
            self.best_score = score
            self.counter    = 0
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