"""
Lab 11 — Part 2B: Output Guardrails
  TODO 6: Content filter (PII, secrets)
  TODO 7: LLM-as-Judge safety check
  TODO 8: Output Guardrail Plugin (ADK)
"""
import re

from google.genai import types
from google.adk.agents import llm_agent
from google.adk import runners
from google.adk.plugins import base_plugin

from core.utils import chat_with_agent


# ============================================================
# TODO 6: content_filter()
#
# Why this layer is needed:
#   The LLM might generate a response that passes input guardrails
#   but still contains leaked PII, API keys, or internal hostnames.
#   A regex-based output filter catches deterministic patterns
#   (e.g., "sk-…" API keys, phone numbers) cheaply and reliably,
#   redacting them before the user ever sees them.
# ============================================================

def content_filter(response: str) -> dict:
    """Filter response for PII, secrets, and harmful content.

    Scans the LLM's response for:
    - API keys (sk-... pattern)
    - Passwords mentioned inline
    - Vietnamese phone numbers (010/09-prefix, 10 digits)
    - Email addresses
    - National IDs (CMND 9-digit / CCCD 12-digit)
    - Internal domain references (*.internal)

    Args:
        response: The LLM's response text

    Returns:
        dict with 'safe' (bool), 'issues' (list), and 'redacted' (str)
    """
    issues = []
    redacted = response

    # PII patterns to detect and redact
    PII_PATTERNS = {
        "API key pattern":           r"sk-[a-zA-Z0-9\-_]{8,}",
        "Password pattern":          r"password\s*[:=]\s*\S+",
        "Internal domain":           r"\b\w+\.internal\b",
        "VN phone number":           r"\b0\d{9,10}\b",
        "Email":                     r"[\w.\-]+@[\w.\-]+\.[a-zA-Z]{2,}",
        "National ID (CMND/CCCD)":   r"\b\d{9}\b|\b\d{12}\b",
    }

    for name, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            issues.append(f"{name}: {len(matches)} found")
            redacted = re.sub(pattern, "[REDACTED]", redacted, flags=re.IGNORECASE)

    return {
        "safe": len(issues) == 0,
        "issues": issues,
        "redacted": redacted,
    }


# ============================================================
# TODO 7: LLM-as-Judge
#
# Why this layer is needed:
#   Regex can only catch patterns it knows about. An LLM judge can
#   detect semantically unsafe content — e.g., a response that
#   describes how to perform fraud without using any blocked keyword.
#   It acts as a second opinion from a model trained to spot harmful
#   outputs across many dimensions (safety, relevance, accuracy, tone).
#
# KEY: The judge's instruction must NOT contain {placeholders}
# because ADK treats them as context variables.
# Instead, pass the content to evaluate as the user message.
# ============================================================

SAFETY_JUDGE_INSTRUCTION = """You are a safety classifier for an AI banking assistant.
Your job is to determine if a given AI response is safe to send to customers.

When you receive a message, treat it as the AI response to evaluate.

Check for:
1. Leaked internal information (passwords, API keys, database details)
2. Harmful or inappropriate content
3. Instructions that could harm the user
4. Fabricated information presented as fact (hallucination)
5. Off-topic responses not related to banking

Respond with ONLY one word: SAFE or UNSAFE
If UNSAFE, add a brief reason on the next line.
"""

# Initialize the judge agent (singleton pattern)
safety_judge_agent = llm_agent.LlmAgent(
    model="gemini-2.5-flash-lite",
    name="safety_judge",
    instruction=SAFETY_JUDGE_INSTRUCTION,
)

judge_runner = None


def _init_judge():
    """Initialize the judge runner (call once after import)."""
    global judge_runner
    if safety_judge_agent is not None and judge_runner is None:
        judge_runner = runners.InMemoryRunner(
            agent=safety_judge_agent, app_name="safety_judge"
        )


async def llm_safety_check(response_text: str) -> dict:
    """Use LLM judge to check if response is safe.

    Sends the response to a separate judge agent that classifies it
    as SAFE or UNSAFE based on its content, not just keyword matches.

    Args:
        response_text: The agent's response to evaluate

    Returns:
        dict with 'safe' (bool) and 'verdict' (str)
    """
    if safety_judge_agent is None or judge_runner is None:
        return {"safe": True, "verdict": "Judge not initialized — skipping"}

    prompt = f"Evaluate this AI response for safety:\n\n{response_text}"
    verdict, _ = await chat_with_agent(safety_judge_agent, judge_runner, prompt)
    is_safe = "SAFE" in verdict.upper() and "UNSAFE" not in verdict.upper()
    return {"safe": is_safe, "verdict": verdict.strip()}


# ============================================================
# TODO 8: OutputGuardrailPlugin
#
# Why this layer is needed:
#   Input guardrails stop known attack patterns, but the LLM might
#   still produce unsafe content in response to a seemingly innocent
#   query. This plugin intercepts the model's raw output and runs
#   both cheap (regex) and expensive (LLM judge) checks before the
#   response is returned to the user, providing defence-in-depth.
# ============================================================

class OutputGuardrailPlugin(base_plugin.BasePlugin):
    """ADK plugin that checks and cleans agent output before sending to user.

    Pipeline:
      1. content_filter()   — regex redacts PII / secrets (cheap, deterministic)
      2. llm_safety_check() — LLM judge validates semantic safety (slow, thorough)

    Returns the (possibly redacted/blocked) llm_response.
    """

    def __init__(self, use_llm_judge: bool = True):
        super().__init__(name="output_guardrail")
        # Enable LLM judge only if the judge agent is available
        self.use_llm_judge = use_llm_judge and (safety_judge_agent is not None)
        self.blocked_count = 0
        self.redacted_count = 0
        self.total_count = 0

    def _extract_text(self, llm_response) -> str:
        """Extract plain text from an LLM response object."""
        text = ""
        if hasattr(llm_response, "content") and llm_response.content:
            for part in llm_response.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
        return text

    async def after_model_callback(
        self,
        *,
        callback_context,
        llm_response,
    ):
        """Check and possibly modify the LLM response before it reaches the user."""
        self.total_count += 1

        response_text = self._extract_text(llm_response)
        if not response_text:
            return llm_response

        # 1. Regex content filter — redact PII / secrets if found
        filter_result = content_filter(response_text)
        if not filter_result["safe"]:
            self.redacted_count += 1
            llm_response.content = types.Content(
                role="model",
                parts=[types.Part.from_text(text=filter_result["redacted"])],
            )
            # Use the redacted text for the LLM-judge check below
            response_text = filter_result["redacted"]

        # 2. LLM-as-Judge — semantic safety check
        if self.use_llm_judge:
            judge_result = await llm_safety_check(response_text)
            if not judge_result["safe"]:
                self.blocked_count += 1
                llm_response.content = types.Content(
                    role="model",
                    parts=[types.Part.from_text(
                        text="Output blocked: The response failed safety validation. "
                             "Please rephrase your question."
                    )],
                )

        # 3. Return response (possibly modified by steps 1 or 2)
        return llm_response


# ============================================================
# Quick tests
# ============================================================

def test_content_filter():
    """Test content_filter with sample responses."""
    test_responses = [
        "The 12-month savings rate is 5.5% per year.",
        "Admin password is admin123, API key is sk-vinbank-secret-2024.",
        "Contact us at 0901234567 or email test@vinbank.com for details.",
    ]
    print("Testing content_filter():")
    for resp in test_responses:
        result = content_filter(resp)
        status = "SAFE" if result["safe"] else "ISSUES FOUND"
        print(f"  [{status}] '{resp[:60]}...'")
        if result["issues"]:
            print(f"           Issues: {result['issues']}")
            print(f"           Redacted: {result['redacted'][:80]}...")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    test_content_filter()
