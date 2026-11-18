ACTION_TYPES = {
    "LOOk_AT","POINT","WALK_TO","EXPLAIN","DEMONSTRATE", "IDLE", "NOD",
    "SHAKE_HEAD","WAVE", "HIGHLIGHT","WAIT", "ASK_CONFIRAMATION"
     }
EXPRESSION_TYPES = {"NEURAL", "SERIOUS","CONCERNED","FRIENDLY","ALERT"}

INTENT_ACTION_MAP = {
    "EMERGENCY": {"action_type":"ALERT_GESTURE","expression":"ALERT"},
    "SAFETY_QUESTION":{"action_type":"EXPLAIN","expression":"SERIOUS"},
    "START_PROCEDURE":{"action_type":"DEMONSTRATE","expression":"SERIOUS"},
    "STOP_PROCEDURE":{"action_type":"DEMONSTRATE","expression":"SERIOUS"},
    "TROUBLESHOOTING":{"action_type":"EXPLAIN","expression":"CONCERNED"},
    "MAINTENANCE":{"action_type":"EXPLAIN","expression":"NEUTRAL"},
    "MACHINE_INFORMATION":{"action_type":"EXPLAIN", "expression":"NEUTRAL"},
    "GREETING":{"action_type":"WAVE","expression":"FRIENDLY"},
    "UNKNOWN":{"action_type":"IDLE", "expression":"NEUTRAL"},
}

def decide_action(intent, safety_status="ALLOWED",target = None):

    mapping  = INTENT_ACTION_MAP.get(intent, INTENT_ACTION_MAP["UNKNOWN"])

    action_type = mapping["action_type"]
    expression = mapping["expression"]


    if safety_status == "HIGH_RISK":
        action_type = "ASK_CONFIRMATION"
        requires_confirmation = True

    else:
        requires_confirmation = False

    return {
        "intent":intent,
        "action":{
            "type":action_type, 
            "target":target,
        },
        "expression": expression,
        "requires_confirmation": requires_confirmation,
    }


if __name__ == "__main__":
    test_cases = [
        ("EMEGENCY","ALLOWED"),
        ("SAFETY_QUESTION","ALLOWED"),
        ("START_PROCEDURE", "HIGH_RISK"),
        ("GREETINGS","ALLOWED"),
        ("UNKNOWN","ALLOWED"),
    ]

    print("Testing actions decisions\n")
    for intent, safety_status in test_cases:
        result = decide_action(intent, safety_status)
        print(f"intent = {intent}, safety = {safety_status}")
        print(f" -> {result}\n")

        