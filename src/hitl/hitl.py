"""
Lab 11 — Part 4: Human-in-the-Loop Design
  TODO 12: Confidence Router
  TODO 13: Design 3 HITL decision points
"""
from dataclasses import dataclass


# ============================================================
# TODO 12: ConfidenceRouter
#
# Why this component is needed:
#   Not every AI response needs a human reviewer, but some do.
#   A confidence-based router automates the triage:
#   - High-confidence, low-risk responses go directly to the user.
#   - Medium-confidence responses are queued for async human review.
#   - Low-confidence responses are escalated immediately.
#   - High-risk actions (money transfers, account deletions) ALWAYS
#     require a human, regardless of confidence, because the cost
#     of an error is too high.
# ============================================================

HIGH_RISK_ACTIONS = [
    "transfer_money",
    "close_account",
    "change_password",
    "delete_data",
    "update_personal_info",
]


@dataclass
class RoutingDecision:
    """Result of the confidence router."""
    action: str          # "auto_send", "queue_review", "escalate"
    confidence: float
    reason: str
    priority: str        # "low", "normal", "high"
    requires_human: bool


class ConfidenceRouter:
    """Route agent responses based on confidence and risk level.

    Decision logic:
      1. High-risk action → always escalate (Human-as-tiebreaker).
      2. confidence >= 0.9 → auto-send (Human-on-the-loop).
      3. 0.7 <= confidence < 0.9 → queue for review (Human-in-the-loop).
      4. confidence < 0.7 → escalate immediately (Human-as-tiebreaker).

    The three thresholds create three HITL models as sliding doors:
    most traffic auto-sends, edge cases get reviewed, uncertain or
    risky decisions always land with a human.
    """

    HIGH_THRESHOLD = 0.9
    MEDIUM_THRESHOLD = 0.7

    def route(self, response: str, confidence: float,
              action_type: str = "general") -> RoutingDecision:
        """Route a response based on confidence score and action type.

        Args:
            response: The agent's response text
            confidence: Confidence score between 0.0 and 1.0
            action_type: Type of action (e.g., "general", "transfer_money")

        Returns:
            RoutingDecision with routing action and metadata
        """
        # 1. High-risk actions always require a human regardless of confidence
        if action_type in HIGH_RISK_ACTIONS:
            return RoutingDecision(
                action="escalate",
                confidence=confidence,
                reason=f"High-risk action: {action_type}",
                priority="high",
                requires_human=True,
            )

        # 2. High confidence → auto-send (human monitors after the fact)
        if confidence >= self.HIGH_THRESHOLD:
            return RoutingDecision(
                action="auto_send",
                confidence=confidence,
                reason="High confidence",
                priority="low",
                requires_human=False,
            )

        # 3. Medium confidence → queue for async human review
        if confidence >= self.MEDIUM_THRESHOLD:
            return RoutingDecision(
                action="queue_review",
                confidence=confidence,
                reason="Medium confidence — needs review",
                priority="normal",
                requires_human=True,
            )

        # 4. Low confidence → escalate immediately
        return RoutingDecision(
            action="escalate",
            confidence=confidence,
            reason="Low confidence — escalating to human",
            priority="high",
            requires_human=True,
        )


# ============================================================
# TODO 13: Design 3 HITL decision points
#
# Three real banking scenarios where human judgment is critical.
# Each decision point maps to one of the three HITL models:
#   - Human-in-the-loop:   agent proposes, human approves before acting
#   - Human-on-the-loop:   agent acts, human monitors & can override
#   - Human-as-tiebreaker: agent is uncertain, human makes final call
# ============================================================

hitl_decision_points = [
    {
        "id": 1,
        "name": "Large Money Transfer Approval",
        # A transfer above 50 M VND is irreversible and high-value.
        # The agent can draft the instruction but must not execute it
        # without a human banker reviewing the transaction details first.
        "trigger": "Customer requests a transfer > 50,000,000 VND",
        "hitl_model": "human-in-the-loop",
        "context_needed": (
            "Customer transaction history (last 30 days), current account balance, "
            "KYC verification status, recipient account details, and fraud-risk score."
        ),
        "example": (
            "Customer asks to transfer 200 M VND to an unfamiliar account. "
            "The agent drafts the transfer request and puts it in a review queue. "
            "A human banker reviews the risk score and KYC status before approving."
        ),
    },
    {
        "id": 2,
        "name": "Account Closure Request",
        # Closing an account is irreversible. Even if the AI is 99% confident
        # the customer wants to close, a human must confirm — the downside of
        # a false positive (an account wrongly closed) is too severe.
        "trigger": "Any intent to permanently close or delete a bank account detected",
        "hitl_model": "human-as-tiebreaker",
        "context_needed": (
            "Outstanding loan balances, remaining account balance, standing orders, "
            "linked cards, and identity verification confirmation."
        ),
        "example": (
            "Customer says 'I want to close my account'. The AI detects account-closure "
            "intent and immediately escalates to a senior banker who contacts the customer "
            "by phone to confirm intent and walk through the closure checklist."
        ),
    },
    {
        "id": 3,
        "name": "Ambiguous High-Value Loan Application",
        # The AI can pre-screen a loan application and compute a credit score,
        # but for loans above 500 M VND the final approval requires a human
        # credit officer to review edge cases the model may misjudge.
        "trigger": "Loan amount > 500,000,000 VND OR credit-score confidence < 0.75",
        "hitl_model": "human-in-the-loop",
        "context_needed": (
            "Applicant's credit score, monthly income evidence, total existing debt, "
            "debt-to-income ratio, collateral details, and employment status."
        ),
        "example": (
            "Customer applies for a 700 M VND business loan. The AI pre-fills the "
            "application form and adds a risk summary. A credit officer then reviews "
            "the supporting documents and makes the final approve/reject decision."
        ),
    },
]


# ============================================================
# Quick tests
# ============================================================

def test_confidence_router():
    """Test ConfidenceRouter with sample scenarios."""
    router = ConfidenceRouter()

    test_cases = [
        ("Balance inquiry",       0.95, "general"),
        ("Interest rate question", 0.82, "general"),
        ("Ambiguous request",      0.55, "general"),
        ("Transfer $50,000",       0.98, "transfer_money"),
        ("Close my account",       0.91, "close_account"),
    ]

    print("Testing ConfidenceRouter:")
    print("=" * 80)
    print(f"{'Scenario':<25} {'Conf':<6} {'Action Type':<18} {'Decision':<15} {'Priority':<10} {'Human?'}")
    print("-" * 80)

    for scenario, conf, action_type in test_cases:
        decision = router.route(scenario, conf, action_type)
        print(
            f"{scenario:<25} {conf:<6.2f} {action_type:<18} "
            f"{decision.action:<15} {decision.priority:<10} "
            f"{'Yes' if decision.requires_human else 'No'}"
        )

    print("=" * 80)


def test_hitl_points():
    """Display HITL decision points."""
    print("\nHITL Decision Points:")
    print("=" * 60)
    for point in hitl_decision_points:
        print(f"\n  Decision Point #{point['id']}: {point['name']}")
        print(f"    Trigger:  {point['trigger']}")
        print(f"    Model:    {point['hitl_model']}")
        print(f"    Context:  {point['context_needed'][:100]}...")
        print(f"    Example:  {point['example'][:100]}...")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_confidence_router()
    test_hitl_points()
