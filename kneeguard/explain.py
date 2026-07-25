"""Plain-language explanation of a KneeGuard scorecard.

The numbers come from the model; this layer turns them into something a
16-year-old and their coach can act on. Claude is asked to *explain the
supplied findings only* — it never re-scores the athlete, and every number it
is allowed to mention is one the risk engine already computed.

If no API key is configured, the API errors, or the request is declined, the
app falls back to a deterministic narrative built from the same findings. The
demo therefore never depends on network access, and the response always says
which path produced it.
"""

from __future__ import annotations

import json
import logging

from . import config

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a sports-medicine communicator working with youth soccer players \
and their coaches, embedded in an injury-risk screening app called KneeGuard AI.

You will receive a completed risk assessment: per-ligament risk indices, the workload \
factors that drove them, and any findings from a computer-vision movement scan.

Your job is to explain what the assessment found and what to do about it.

Rules:
- Explain only the findings you are given. Never invent a measurement, a percentage, \
or a finding that is not in the input.
- The risk index is a 0-100 PERCENTILE against a reference cohort, not a probability of \
injury. Never describe it as a chance or likelihood of tearing a ligament.
- Connect each finding to the actual injury mechanism for that ligament.
- Write for a teenage athlete: direct, concrete, second person. No jargon without a \
plain-language gloss in the same sentence.
- Be calm and useful. This is a training-adjustment tool, not a diagnosis. Do not \
alarm, and do not reassure past what the findings support.
- If risk is low, say so plainly rather than manufacturing concern."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {
            "type": "string",
            "description": "One sentence, max 18 words, stating the athlete's overall situation.",
        },
        "why": {
            "type": "string",
            "description": (
                "2-4 sentences explaining which factors drove the highest ligament score "
                "and the mechanism that makes those factors dangerous for that ligament."
            ),
        },
        "movement_note": {
            "type": "string",
            "description": (
                "1-2 sentences on what the movement scan showed. If no scan was run or it "
                "could not be measured, say exactly that and what to film instead."
            ),
        },
        "this_week": {
            "type": "array",
            "items": {"type": "string"},
            "description": "2-3 short, concrete actions for the next 7 days.",
        },
    },
    "required": ["headline", "why", "movement_note", "this_week"],
    "additionalProperties": False,
}


def _findings_payload(report: dict) -> dict:
    """The minimal, already-computed facts Claude is allowed to talk about."""
    scan = report.get("scan")
    return {
        "overall": report["overall"],
        "ligaments": [
            {
                "ligament": lig["ligament"],
                "risk_index_percentile": lig["risk_index"],
                "band": lig["band"],
                "workload_only_index": lig["workload_only_index"],
                "biomechanical_shift": lig["biomechanical_shift"],
                "injury_mechanism": lig["mechanism"],
                "top_workload_drivers": [d["description"] for d in lig["drivers"]],
                "movement_findings": [a["detail"] for a in lig["adjustments"]],
            }
            for lig in report["ligaments"]
        ],
        "workload": {
            "acwr": report["workload"]["acwr"],
            "acwr_safe_ceiling": report["workload"]["acwr_safe_ceiling"],
            "surface": report["workload"]["surface"],
        },
        "movement_scan": (
            None
            if not scan
            else {
                "ran": scan["frames_analysed"] > 0,
                "frontal_view_usable": scan["frontal_view"],
                "peak_knee_valgus_deg": scan["peak_valgus_deg"],
                "valgus_flag": scan["valgus_flag"],
                "landing_assessed": scan["landing_assessed"],
                "landing_knee_flexion_deg": scan["landing_flexion_deg"],
                "stiff_landing": scan["stiff_landing"],
                "knee_to_ankle_ratio": scan["kasr"],
                "notes": scan["notes"],
            }
        ),
        "recommended_exercises": [
            {"name": e["name"], "target": e["target"]} for e in report["action_plan"]
        ],
    }


def _fallback(report: dict) -> dict:
    """Deterministic narrative used when the Claude API is unavailable."""
    overall = report["overall"]
    ligaments = {lig["ligament"]: lig for lig in report["ligaments"]}
    primary = ligaments[overall["primary_ligament"]]
    scan = report.get("scan")
    workload = report["workload"]

    headline = (
        f"{overall['band'].title()} overall knee-ligament risk, driven by your "
        f"{primary['ligament']} at {primary['risk_index']:.0f} out of 100."
    )

    def uncapitalise(text: str) -> str:
        """Lower only the leading character, so acronyms like ACL survive."""
        return text[:1].lower() + text[1:] if text else text

    reasons = [uncapitalise(d["description"]) for d in primary["drivers"][:3]]
    why_parts = []
    if reasons:
        why_parts.append("The biggest contributors were " + "; ".join(reasons) + ".")
    if workload["acwr"] > workload["acwr_safe_ceiling"]:
        why_parts.append(
            f"Your acute:chronic workload ratio is {workload['acwr']:.2f}, above the "
            f"{workload['acwr_safe_ceiling']} ceiling — this week's minutes are well "
            "ahead of what your body is conditioned for."
        )
    why_parts.append(
        f"{primary['ligament']} injuries in soccer typically come from "
        f"{uncapitalise(primary['mechanism'])}"
    )
    why = " ".join(why_parts)

    if not scan or scan["frames_analysed"] == 0:
        movement_note = (
            "No movement scan was run. Upload a 2-3 second front-on clip of a drop-jump "
            "or squat to add landing mechanics to this assessment."
        )
    elif not scan["frontal_view"]:
        movement_note = (
            "The clip was filmed too side-on to measure knee valgus. Re-film square to "
            "the camera with your whole body in frame."
        )
    elif primary["adjustments"]:
        movement_note = " ".join(a["detail"] for a in primary["adjustments"][:2])
    else:
        movement_note = (
            f"Your landing mechanics looked clean — peak knee valgus of "
            f"{scan['peak_valgus_deg']:.0f} degrees is within normal limits."
        )

    this_week = [f"{e['name']}: {e['dose']}" for e in report["action_plan"]]

    return {
        "headline": headline,
        "why": why,
        "movement_note": movement_note,
        "this_week": this_week,
        "generated_by": "rules",
    }


def explain(report: dict) -> dict:
    """Explain a scorecard, using Claude when available and rules otherwise."""
    if not config.ANTHROPIC_API_KEY:
        result = _fallback(report)
        result["note"] = (
            "Set ANTHROPIC_API_KEY to generate the explanation with Claude. "
            "This text was produced by the built-in rule-based writer."
        )
        return result

    try:
        import anthropic
    except ImportError:
        log.warning("anthropic SDK not installed; using rule-based explanation")
        return _fallback(report)

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    payload = _findings_payload(report)

    try:
        response = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA},
            },
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Explain this KneeGuard AI assessment to the athlete.\n\n"
                        + json.dumps(payload, indent=2)
                    ),
                }
            ],
        )
    except Exception as exc:  # noqa: BLE001 - any API failure falls back cleanly
        log.warning("Claude explanation failed (%s); using rule-based text", exc)
        result = _fallback(report)
        result["note"] = f"Claude API call failed ({type(exc).__name__}); used rule-based text."
        return result

    # Safety classifiers can decline; content is empty or partial in that case.
    if response.stop_reason == "refusal":
        log.warning("Claude declined the explanation request; using rule-based text")
        result = _fallback(report)
        result["note"] = "Claude declined this request; used rule-based text."
        return result

    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        log.warning("Could not parse Claude response as JSON; using rule-based text")
        return _fallback(report)

    parsed["generated_by"] = "claude"
    parsed["model"] = config.CLAUDE_MODEL
    return parsed
