from traffic_logic import choose_emergency_lane, get_signal_state


def test_choose_emergency_lane_uses_highest_confidence_ambulance():
    detections = [
        {"lane": "North", "confidence": 0.7, "is_ambulance": True},
        {"lane": "South", "confidence": 0.9, "is_ambulance": True},
        {"lane": "East", "confidence": 0.99, "is_ambulance": False},
    ]

    assert choose_emergency_lane(detections) == "South"


def test_get_signal_state_all_red_when_no_ambulance():
    detections = [{"lane": "North", "confidence": 0.8, "is_ambulance": False}]

    assert get_signal_state(detections) == {
        "North": "red",
        "South": "red",
        "East": "red",
        "West": "red",
    }


def test_get_signal_state_sets_only_emergency_lane_green():
    detections = [{"lane": "West", "confidence": 0.91, "is_ambulance": True}]

    assert get_signal_state(detections) == {
        "North": "red",
        "South": "red",
        "East": "red",
        "West": "green",
    }
