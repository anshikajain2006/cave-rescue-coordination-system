from cave_hmm import CaveHMMLocaliser

ENTROPY_THRESHOLD = 3.5  # Nats; tune: log(N)/2 for N states is a good default (the cave's 247 cells give ~2.76)


def is_confident(hmm: CaveHMMLocaliser) -> bool:
    """Return True if the HMM belief is concentrated enough to navigate safely."""
    return hmm.belief_entropy() < ENTROPY_THRESHOLD


def confidence_status(hmm: CaveHMMLocaliser) -> dict:
    """Return entropy, threshold, and whether navigation is allowed."""
    entropy = hmm.belief_entropy()
    return {
        "entropy": round(entropy, 3),
        "threshold": ENTROPY_THRESHOLD,
        "confident": entropy < ENTROPY_THRESHOLD,
        "message": "Position known" if entropy < ENTROPY_THRESHOLD
        else f"Position uncertain (entropy={entropy:.2f}) — run SCAN first",
    }
