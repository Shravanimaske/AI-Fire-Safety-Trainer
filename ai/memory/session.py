"""
Conversation memory: tracks recent turns in a session, bounded to avoid
unbounded growth (per original spec: "do not blindly send unlimited
conversation history to the LLM").

Usage:
    from ai.memory.session import ConversationMemory
    memory = ConversationMemory(max_turns=4)
    memory.add_turn("user", "what should I do if I discover a fire")
    memory.add_turn("assistant", "Stay calm, raise the alarm...")
    history_text = memory.get_history_text()
"""


class ConversationMemory:
    def __init__(self, max_turns=4):
        """
        max_turns: how many recent USER+ASSISTANT turn pairs to keep.
        This is a simple sliding window - older turns are dropped
        entirely, not summarized. Chosen deliberately as the simplest
        approach that could work; we test whether this is sufficient
        before adding complexity like summarization.
        """
        self.max_turns = max_turns
        self.history = []  # list of {"role": "user"/"assistant", "text": ...}

        # Session-level context beyond just message history - what
        # machine/scenario the conversation is currently about, useful
        # for later phases (e.g. Unity needs to know "current machine"
        # per the original spec's session structure).
        self.current_machine = None
        self.current_scenario = None

    def add_turn(self, role, text):
        self.history.append({"role": role, "text": text})

        # Enforce the bound: keep only the last max_turns USER turns
        # (and their corresponding assistant replies). We count by
        # user turns since that's the natural unit of "a turn" here.
        user_turn_count = sum(1 for h in self.history if h["role"] == "user")
        while user_turn_count > self.max_turns:
            # Drop from the front until we're back within bounds
            removed = self.history.pop(0)
            if removed["role"] == "user":
                user_turn_count -= 1

    def get_history_text(self):
        """Format recent history as text to prepend to a new prompt."""
        if not self.history:
            return ""

        lines = []
        for turn in self.history:
            label = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{label}: {turn['text']}")

        return "\n".join(lines)

    def turn_count(self):
        return sum(1 for h in self.history if h["role"] == "user")


if __name__ == "__main__":
    # Self-test: verify the bound actually works
    memory = ConversationMemory(max_turns=2)

    for i in range(5):
        memory.add_turn("user", f"question {i}")
        memory.add_turn("assistant", f"answer {i}")

    print("History after adding 5 turns with max_turns=2:")
    print(memory.get_history_text())
    print(f"\nTurn count: {memory.turn_count()} (expected: 2)")
    assert memory.turn_count() == 2, "Bound not enforced correctly!"
    print("Bound correctly enforced.")