#!/usr/bin/env python3
"""
generate_consensus_pairs.py — Multi-model contrastive pair generation.

Sends the same generation prompt to N diverse frontier models via FuelIX,
collects contrastive pairs from each, then labels every pair with agreement
level across all models. High-agreement pairs (9+/12) are consensus truth.
Low-agreement pairs are boundary cases worth investigating.

Output format matches existing caz_scaling/data/*.jsonl with added fields:
  - model_name: which model generated this pair
  - agreement: how many models out of N labeled it the same way
  - agreement_pct: agreement as percentage
  - voter_labels: {model_id: label} for all models that voted

Usage:
    python src/generate_consensus_pairs.py --concept credibility --n-pairs 20
    python src/generate_consensus_pairs.py --all --n-pairs 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# FuelIX config — reads from env vars or searches for .env in common locations
FUELIX_BASE_URL = os.environ.get("FUELIX_BASE_URL", "https://api.fuelix.ai")
FUELIX_API_KEY = os.environ.get("FUELIX_API_KEY", "")

if not FUELIX_API_KEY:
    _env_candidates = [
        Path.home() / "Source" / "fuelix_kilocode_profiles" / ".env",
        Path(__file__).resolve().parents[2] / ".env",
        Path.cwd() / ".env",
    ]
    for env_path in _env_candidates:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("FUELIX_API_KEY="):
                    FUELIX_API_KEY = line.split("=", 1)[1].strip()
                elif line.startswith("FUELIX_BASE_URL="):
                    FUELIX_BASE_URL = line.split("=", 1)[1].strip()
            if FUELIX_API_KEY:
                break

# Default output dir — caller can override via --output-dir
DATA_DIR = Path.cwd() / "data"

# One flagship per family×generation — verified working on FuelIX (2026-04-02)
MODELS = [
    "claude-sonnet-4-6",       # Anthropic — latest
    "claude-sonnet-4-5",       # Anthropic — previous gen
    "claude-3-7-sonnet",       # Anthropic — older gen
    "claude-haiku-4-5",        # Anthropic — small
    "gpt-5.4",                 # OpenAI — latest
    "gpt-5-mini",              # OpenAI — small frontier
    "gpt-5-nano",              # OpenAI — tiny frontier
    "gpt-4o",                  # OpenAI — previous gen
    "gemini-3.1-pro",          # Google — latest pro
    "gemini-3-flash",          # Google — latest flash
    "gemini-2.5-pro",          # Google — previous gen
    "kimi-k2.5",               # Moonshot — Chinese-trained
    "mistral-large",           # Mistral — European
    "o4-mini",                 # OpenAI — reasoning model
]

CONCEPTS = {
    # === Original 7 (matches existing Claude-generated pairs) ===
    "credibility": {
        "description": "Whether a text presents information in a credible, well-sourced, evidence-based manner vs an unreliable, vague, emotionally manipulative manner",
        "pos_label": "credible, well-sourced, evidence-based",
        "neg_label": "not credible, vague, emotionally manipulative, poorly sourced",
    },
    "sentiment": {
        "description": "Whether a text expresses positive vs negative sentiment or emotional valence",
        "pos_label": "positive sentiment",
        "neg_label": "negative sentiment",
    },
    "causation": {
        "description": "Whether a text describes a clear cause-and-effect relationship vs a non-causal description",
        "pos_label": "describes a clear causal mechanism",
        "neg_label": "describes events without causal connection",
    },
    "certainty": {
        "description": "Whether a text expresses high epistemic certainty vs hedged, uncertain claims",
        "pos_label": "high certainty, definitive claims",
        "neg_label": "uncertain, hedged, speculative",
    },
    "negation": {
        "description": "Whether a text contains meaningful negation that changes the core meaning vs affirmative statements",
        "pos_label": "contains meaningful negation",
        "neg_label": "affirmative, no negation",
    },
    "moral_valence": {
        "description": "Whether a text describes something morally praiseworthy vs morally blameworthy",
        "pos_label": "morally praiseworthy or virtuous",
        "neg_label": "morally blameworthy or wrong",
    },
    "temporal_order": {
        "description": "Whether a text clearly establishes temporal sequence (before/after/during) vs is temporally ambiguous",
        "pos_label": "clear temporal ordering of events",
        "neg_label": "temporally ambiguous or unordered",
    },
    # === From original Rosetta_Manifold set (dropped from caz_scaling) ===
    "plurality": {
        "description": "Whether a text uses plural constructions (groups, collections, multiple entities) vs singular constructions (one entity, individual focus)",
        "pos_label": "plural constructions, groups, multiple entities",
        "neg_label": "singular constructions, individual focus",
    },
    # === New concepts — expanding the probe vocabulary ===
    "sarcasm": {
        "description": "Whether a text uses sarcasm or irony (saying one thing, meaning the opposite) vs sincere, literal expression",
        "pos_label": "sarcastic or ironic",
        "neg_label": "sincere and literal",
    },
    "formality": {
        "description": "Whether a text uses formal, professional register vs casual, colloquial register",
        "pos_label": "formal, professional, academic register",
        "neg_label": "casual, colloquial, informal register",
    },
    "specificity": {
        "description": "Whether a text makes specific, precise, quantified claims vs vague, general, hand-wavy assertions",
        "pos_label": "specific, precise, quantified",
        "neg_label": "vague, general, unquantified",
    },
    "agency": {
        "description": "Whether a text describes active agents making deliberate choices vs passive events happening to people or things",
        "pos_label": "active agency, deliberate choices, someone acts",
        "neg_label": "passive voice, things happen, no clear agent",
    },
    "deception": {
        "description": "Whether a text contains deliberate deception, misleading framing, or manipulative intent vs honest, transparent communication",
        "pos_label": "deceptive, misleading, manipulative framing",
        "neg_label": "honest, transparent, straightforward",
    },
}


_last_call_time = 0.0
_call_interval = 60.0 / 75  # 75 requests per minute


def call_model(model_id: str, prompt: str, max_retries: int = 3) -> str | None:
    """Call a FuelIX model with a prompt. Returns the response text or None.

    Rate-limited to 75 requests/minute.
    """
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < _call_interval:
        time.sleep(_call_interval - elapsed)
    _last_call_time = time.time()

    headers = {
        "Authorization": f"Bearer {FUELIX_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": 2000,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{FUELIX_BASE_URL}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            elif resp.status_code == 429:
                wait = min(30, 2 ** attempt * 5)
                log.warning("  Rate limited on %s, waiting %ds...", model_id, wait)
                time.sleep(wait)
            else:
                log.warning("  %s returned %d: %s", model_id, resp.status_code, resp.text[:200])
                return None
        except (requests.RequestException, KeyError, IndexError) as e:
            log.warning("  %s error: %s", model_id, e)
            if attempt < max_retries - 1:
                time.sleep(2)
    return None


def generate_pair_prompt(concept: str, topic: str, pair_id: str) -> str:
    """Build the prompt for generating one contrastive pair."""
    info = CONCEPTS[concept]
    return f"""Generate a contrastive pair of texts about the topic: "{topic}"

Concept being contrasted: {info['description']}

Write exactly two texts (150-250 words each):

TEXT_A ({info['pos_label']}):
A text that clearly demonstrates {info['pos_label']}. Use specific details, concrete examples, and appropriate register.

TEXT_B ({info['neg_label']}):
A text on the SAME topic that clearly demonstrates {info['neg_label']}. The topic and general subject matter should be the same, but the stylistic and epistemic qualities should contrast sharply.

Format your response as JSON:
{{
  "text_a": "...",
  "text_b": "...",
  "topic": "{topic}"
}}

Respond with ONLY the JSON, no other text."""


TOPICS = {
    "credibility": [
        "climate change mitigation strategies",
        "vaccine development timelines",
        "artificial intelligence in healthcare",
        "cryptocurrency market regulation",
        "nuclear fusion energy progress",
        "microplastics in drinking water",
        "gene therapy for rare diseases",
        "autonomous vehicle safety records",
        "antibiotic resistance trends",
        "dark matter detection methods",
        "ocean acidification measurements",
        "mental health treatment outcomes",
        "renewable energy grid integration",
        "CRISPR agricultural applications",
        "pandemic preparedness frameworks",
        "brain-computer interface trials",
        "deforestation satellite monitoring",
        "quantum computing error correction",
        "soil carbon sequestration rates",
        "mRNA platform drug development",
    ],
    "sentiment": [
        "restaurant dining experience",
        "new smartphone review",
        "neighborhood community changes",
        "workplace culture shift",
        "travel destination assessment",
        "educational program outcomes",
        "healthcare service quality",
        "environmental policy impacts",
        "technology adoption experience",
        "urban development project",
        "retirement planning outlook",
        "childhood education quality",
        "public transportation service",
        "housing market conditions",
        "local business opening",
        "career change experience",
        "volunteer program impact",
        "sports season review",
        "arts festival experience",
        "home renovation results",
    ],
    "causation": [
        "deforestation and flooding",
        "education access and economic mobility",
        "sleep deprivation and cognitive function",
        "social media use and political polarization",
        "exercise and neuroplasticity",
        "ocean temperature and hurricane intensity",
        "lead exposure and developmental outcomes",
        "urbanization and biodiversity loss",
        "monetary policy and inflation",
        "screen time and childhood attention span",
        "air pollution and respiratory disease",
        "trade policy and manufacturing employment",
        "vaccination rates and herd immunity thresholds",
        "soil microbiome and crop yields",
        "interest rates and housing prices",
        "remote work and urban migration",
        "antibiotic overuse and resistance evolution",
        "early intervention and autism outcomes",
        "carbon pricing and emissions reduction",
        "gut bacteria and mental health",
    ],
    "sarcasm": [
        "airline customer service experience",
        "productivity advice from influencers",
        "tech company mission statements",
        "political campaign promises",
        "luxury brand marketing claims",
        "corporate diversity initiatives",
        "social media wellness trends",
        "startup disruption narratives",
        "celebrity charity announcements",
        "fast fashion sustainability claims",
        "gig economy worker testimonials",
        "cryptocurrency investment advice",
        "reality TV educational value",
        "automated customer support quality",
        "tech conference keynote visions",
        "corporate apology statements",
        "influencer product endorsements",
        "university ranking methodologies",
        "smart home device reliability",
        "workplace wellness programs",
    ],
    "formality": [
        "job application cover letter",
        "scientific conference presentation",
        "text message to a friend",
        "legal contract terms",
        "restaurant review online",
        "academic journal submission",
        "casual email to coworker",
        "diplomatic press statement",
        "social media comment thread",
        "board meeting minutes",
        "wedding toast speech",
        "medical discharge instructions",
        "neighborhood barbecue invitation",
        "grant proposal methodology",
        "sports commentary during game",
        "eulogy at a funeral",
        "complaint to a landlord",
        "annual shareholder letter",
        "birthday party planning chat",
        "thesis defense opening remarks",
    ],
    "specificity": [
        "climate policy effectiveness",
        "software performance benchmarks",
        "medical treatment outcomes",
        "economic growth projections",
        "athletic performance records",
        "nutritional supplement claims",
        "real estate market analysis",
        "educational achievement metrics",
        "manufacturing quality standards",
        "wildlife population surveys",
        "cybersecurity breach severity",
        "infrastructure project budgets",
        "pharmaceutical trial results",
        "renewable energy output data",
        "traffic accident statistics",
        "agricultural yield comparisons",
        "customer satisfaction surveys",
        "pollution measurement readings",
        "vaccine distribution logistics",
        "startup valuation methods",
    ],
    "agency": [
        "corporate layoff announcements",
        "government policy implementation",
        "scientific discovery narratives",
        "natural disaster aftermath",
        "startup founding stories",
        "criminal justice proceedings",
        "medical diagnosis communication",
        "social movement mobilization",
        "algorithmic decision making",
        "institutional racism effects",
        "whistleblower revelations",
        "climate change responsibility",
        "educational reform initiatives",
        "financial market movements",
        "immigration policy effects",
        "technology adoption patterns",
        "workplace safety incidents",
        "public health interventions",
        "urban gentrification processes",
        "artistic creation processes",
    ],
    "deception": [
        "pharmaceutical marketing practices",
        "political spin and framing",
        "financial product sales tactics",
        "social media misinformation",
        "corporate greenwashing",
        "propaganda techniques in media",
        "phishing and social engineering",
        "misleading food labeling",
        "astroturfing campaigns",
        "manipulative advertising to children",
        "statistical cherry-picking",
        "historical revisionism",
        "fake online reviews",
        "pseudoscientific health claims",
        "misleading data visualizations",
        "selective quotation in journalism",
        "deceptive dark patterns in UX",
        "false scarcity marketing",
        "emotional manipulation in fundraising",
        "deepfake and synthetic media",
    ],
}
# Use credibility topics as fallback for concepts without specific topics
for c in CONCEPTS:
    if c not in TOPICS:
        TOPICS[c] = TOPICS["credibility"]


def parse_json_response(text: str) -> dict | None:
    """Extract JSON from model response, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON in the response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                return None
    return None


def generate_topics(concept: str, n_needed: int) -> list[str]:
    """Generate additional topics for a concept via API if we don't have enough hardcoded."""
    existing = TOPICS.get(concept, TOPICS["credibility"])
    if len(existing) >= n_needed:
        return existing[:n_needed]

    n_extra = n_needed - len(existing)
    log.info("  Generating %d additional topics for %s...", n_extra, concept)

    info = CONCEPTS[concept]
    prompt = f"""Generate {n_extra} diverse, specific topics suitable for contrastive text pairs about: {info['description']}

Requirements:
- Each topic should be specific enough to write 150-250 word texts about
- Cover diverse domains: science, politics, technology, health, society, environment, economics, culture
- Avoid overlap with these existing topics: {', '.join(existing[:10])}...

Return ONLY a JSON array of topic strings, e.g.: ["topic 1", "topic 2", ...]"""

    response = call_model("gpt-5-mini", prompt)
    if response:
        parsed = parse_json_response(response)
        if isinstance(parsed, list):
            return existing + parsed[:n_extra]

    # Fallback: duplicate with numeric suffixes
    log.warning("  Topic generation failed, using numbered variants")
    extras = [f"{existing[i % len(existing)]} (variant {i // len(existing) + 2})"
              for i in range(n_extra)]
    return existing + extras


def load_checkpoint(out_path: Path) -> set[str]:
    """Load completed pair_id+model combos from existing output file."""
    completed = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    completed.add(f"{rec['pair_id']}:{rec['model_name']}")
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


def generate_for_concept(
    concept: str, n_pairs: int, models: list[str], out_path: Path,
) -> int:
    """Generate contrastive pairs from all models for one concept.

    Appends to out_path incrementally. Skips pair_id+model combos that
    already exist in the file (resume capability).

    Returns total number of new texts written.
    """
    topics = generate_topics(concept, n_pairs)
    completed = load_checkpoint(out_path)
    if completed:
        log.info("  Resuming: %d pair×model combos already done", len(completed))

    consecutive_failures = 0
    max_consecutive_failures = 20  # pause if 20 in a row fail
    texts_written = 0

    for pair_idx, topic in enumerate(topics):
        pair_id = f"consensus_{concept}_{pair_idx:03d}"
        prompt = generate_pair_prompt(concept, topic, pair_id)

        # Check if ALL models already done for this topic
        all_done = all(f"{pair_id}:{m}" in completed for m in models)
        if all_done:
            continue

        log.info("  Pair %d/%d: %s", pair_idx + 1, len(topics), topic)

        for model_id in models:
            checkpoint_key = f"{pair_id}:{model_id}"
            if checkpoint_key in completed:
                continue

            response_text = call_model(model_id, prompt)
            if response_text is None:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    log.error("  %d consecutive failures — backing off 60s...",
                              consecutive_failures)
                    time.sleep(60)
                    consecutive_failures = 0
                continue

            parsed = parse_json_response(response_text)
            if not parsed or "text_a" not in parsed or "text_b" not in parsed:
                log.warning("    %s: failed to parse JSON", model_id)
                consecutive_failures += 1
                continue

            consecutive_failures = 0
            log.info("    %s: OK", model_id)

            # Write immediately (append mode) — crash-safe
            with open(out_path, "a") as f:
                for label, text_key in [(1, "text_a"), (0, "text_b")]:
                    rec = {
                        "pair_id": pair_id,
                        "label": label,
                        "domain": "consensus",
                        "model_name": model_id,
                        "text": parsed[text_key],
                        "topic": topic,
                        "concept": concept,
                    }
                    f.write(json.dumps(rec) + "\n")
                    texts_written += 1

    return texts_written


def main():
    parser = argparse.ArgumentParser(description="Multi-model consensus pair generation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--concept", type=str, choices=list(CONCEPTS.keys()))
    group.add_argument("--all", action="store_true")
    parser.add_argument("--n-pairs", type=int, default=100,
                        help="Number of topic pairs per concept (default: 100)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Override model list (default: 14 flagships)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for JSONL files (default: ./data)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config and exit without calling API")
    args = parser.parse_args()

    models = args.models or MODELS
    global DATA_DIR
    if args.output_dir:
        DATA_DIR = Path(args.output_dir)

    if not FUELIX_API_KEY or FUELIX_API_KEY == "YOUR_FUELIX_API_KEY_HERE":
        log.error("No FuelIX API key found. Set FUELIX_API_KEY or check .env")
        return

    concepts = list(CONCEPTS.keys()) if args.all else [args.concept]

    # Estimate remaining work
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    total_remaining = 0
    for concept in concepts:
        out_path = DATA_DIR / f"{concept}_consensus_pairs.jsonl"
        done = len(load_checkpoint(out_path))
        expected = args.n_pairs * len(models)
        total_remaining += max(0, expected - done)

    log.info("FuelIX consensus pair generator")
    log.info("  Models: %d (%s)", len(models), ", ".join(models[:5]) + "...")
    log.info("  Concepts: %d (%s)", len(concepts), ", ".join(concepts))
    log.info("  Pairs per concept: %d", args.n_pairs)
    log.info("  Remaining API calls: ~%d (%.0f min at 75/min)",
             total_remaining, total_remaining / 75)
    log.info("  Checkpoint: resumes from where it left off if interrupted")
    log.info("")

    if args.dry_run:
        log.info("DRY RUN — would generate the above. Exiting.")
        return

    for concept in concepts:
        out_path = DATA_DIR / f"{concept}_consensus_pairs.jsonl"
        log.info("=== Generating: %s ===", concept)
        n_written = generate_for_concept(concept, args.n_pairs, models, out_path)

        # Summary
        all_recs = []
        if out_path.exists():
            with open(out_path) as f:
                all_recs = [json.loads(l) for l in f if l.strip()]
        n_topics = len(set(r["pair_id"] for r in all_recs))
        n_models = len(set(r["model_name"] for r in all_recs))
        log.info("  Total: %d texts (%d topics × %d models) → %s",
                 len(all_recs), n_topics, n_models, out_path)
        log.info("  New this run: %d texts", n_written)

    log.info("")
    log.info("Done. Restart with same args to resume any incomplete work.")


if __name__ == "__main__":
    main()
