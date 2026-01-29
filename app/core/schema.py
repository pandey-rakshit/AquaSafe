from typing import List

def get_required_features() -> List[str]:
    return [
        "ph",
        "dissolved_o2",
        "bod",
        "conductivity",
        "turbidity",
        # keep EXACT order from training
    ]
