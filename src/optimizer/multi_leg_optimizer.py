"""
Multi-leg parlay optimization utilities.

This module defines functions and classes for computing the expected value of
multi-leg sports outcome combinations (parlays) and searching for the most
advantageous combinations based on model probabilities and sportsbook odds.

The functions here assume independence between legs; correlated outcomes may
invalidate expected value calculations. Use with care in analytical contexts.
"""

from itertools import combinations
from typing import List, Dict, Tuple


def implied_probability_from_american_odds(odds: float) -> float:
    """Convert American odds to implied probability.

    Args:
        odds: American odds (e.g., +150, -110).

    Returns:
        Implied probability as a float between 0 and 1.
    """
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)


def parlay_expected_value(legs: List[Dict[str, float]]) -> float:
    """Compute expected value of a parlay given model probabilities and odds.

    The parlay EV is calculated as:
        EV = P(win) * (payout_multiplier - 1) - (1 - P(win))

    where payout_multiplier is the product of each leg's payout factor.

    Args:
        legs: A list of dictionaries, each containing:
            'model_prob': Probability assigned by the model for the leg to win (0-1).
            'odds': American odds offered by the sportsbook for the leg.

    Returns:
        Expected value of the parlay as a float.
    """
    win_prob = 1.0
    payout_multiplier = 1.0
    for leg in legs:
        model_prob = leg.get('model_prob')
        odds = leg.get('odds')
        if model_prob is None or odds is None:
            raise ValueError("Each leg must have 'model_prob' and 'odds' keys")
        win_prob *= model_prob
        # Convert American odds to payout multiplier: +150 -> 2.5 (stake + profit)
        if odds > 0:
            payout_multiplier *= (odds / 100 + 1)
        else:
            payout_multiplier *= (100 / -odds + 1)
    # EV formula: P(win) * (payout - stake) - (1 - P(win))
    return win_prob * (payout_multiplier - 1) - (1 - win_prob)


class ParlayOptimizer:
    """Optimize multi-leg combinations based on expected value.

    Attributes:
        events: A list of event dictionaries. Each dict should include:
            'id': Unique identifier of the market (to prevent same-game parlays).
            'model_prob': Model-derived probability for the selection.
            'odds': American odds offered by the sportsbook for the selection.

    Methods:
        find_best_parlays: Identify the top parlay combinations by expected value.
    """

    def __init__(self, events: List[Dict[str, float]]):
        self.events = events

    def _valid_combination(self, indices: Tuple[int, ...]) -> bool:
        """Check if a combination is valid (no duplicate event IDs)."""
        seen_ids = set()
        for i in indices:
            event_id = self.events[i].get('id')
            if event_id in seen_ids:
                return False
            seen_ids.add(event_id)
        return True

    def find_best_parlays(self, max_legs: int = 3, top_n: int = 5) -> List[Tuple[Tuple[int, ...], float]]:
        """Search for the best parlay combinations by expected value.

        Generates all combinations of events up to a given length and computes
        expected value for each. Only returns combinations where each leg
        corresponds to a distinct event ID to avoid obvious correlation.

        Args:
            max_legs: Maximum number of legs in the parlay (minimum is 2).
            top_n: Number of top parlays to return.

        Returns:
            A list of tuples: (indices, expected_value) sorted by EV descending.
        """
        best_parlays: List[Tuple[Tuple[int, ...], float]] = []
        n_events = len(self.events)
        for r in range(2, max_legs + 1):
            for indices in combinations(range(n_events), r):
                if not self._valid_combination(indices):
                    continue
                legs = [self.events[i] for i in indices]
                ev = parlay_expected_value(legs)
                best_parlays.append((indices, ev))
        best_parlays.sort(key=lambda x: x[1], reverse=True)
        return best_parlays[:top_n]
