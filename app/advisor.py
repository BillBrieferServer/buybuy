"""
advisor.py
Claude API integration for procurement Q&A.
Grounded responses with statute citations.
"""

import os
import re
import json
import hashlib
import sqlite3
import logging
import time
from datetime import datetime
from pathlib import Path
from anthropic import Anthropic

from . import knowledge_base as kb

logger = logging.getLogger(__name__)

CACHE_DIR = Path("/app/data")
CACHE_DB = CACHE_DIR / "ai_cache.sqlite"

# State name/abbreviation mapping for detection
STATE_MAP = {
    'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR',
    'california': 'CA', 'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE',
    'florida': 'FL', 'georgia': 'GA', 'hawaii': 'HI', 'idaho': 'ID',
    'illinois': 'IL', 'indiana': 'IN', 'iowa': 'IA', 'kansas': 'KS',
    'kentucky': 'KY', 'louisiana': 'LA', 'maine': 'ME', 'maryland': 'MD',
    'massachusetts': 'MA', 'michigan': 'MI', 'minnesota': 'MN',
    'mississippi': 'MS', 'missouri': 'MO', 'montana': 'MT', 'nebraska': 'NE',
    'nevada': 'NV', 'new hampshire': 'NH', 'new jersey': 'NJ',
    'new mexico': 'NM', 'new york': 'NY', 'north carolina': 'NC',
    'north dakota': 'ND', 'ohio': 'OH', 'oklahoma': 'OK', 'oregon': 'OR',
    'pennsylvania': 'PA', 'rhode island': 'RI', 'south carolina': 'SC',
    'south dakota': 'SD', 'tennessee': 'TN', 'texas': 'TX', 'utah': 'UT',
    'vermont': 'VT', 'virginia': 'VA', 'washington': 'WA',
    'west virginia': 'WV', 'wisconsin': 'WI', 'wyoming': 'WY',
    'district of columbia': 'DC',
}

# Reverse map: abbreviation -> name
ABBREV_MAP = {v: k.title() for k, v in STATE_MAP.items()}

# Keywords that suggest vendor preference context is needed
PREFERENCE_KEYWORDS = [
    'out-of-state', 'out of state', 'nonresident', 'non-resident',
    'vendor preference', 'bidder preference', 'reciprocal',
    'local vendor', 'local bidder', 'idaho vendor', 'idaho bidder',
    'in-state preference', 'resident preference', 'resident bidder',
    'preference law', '67-2348', '67-2349',
]

SYSTEM_PROMPT = """You are the Idaho Procurement & CES Advisor, an AI assistant that helps Idaho public officials (county clerks, commissioners, city clerks, school district administrators) navigate public procurement decisions AND understand the CES cooperative purchasing program.

## YOUR ROLE
- Provide accurate, helpful guidance on Idaho procurement law and procedures
- Ground procurement answers in specific Idaho Code sections
- Explain complex procurement rules in plain language
- Help officials determine the correct procurement process for their situation
- Explain cooperative purchasing options (CES, State of Idaho contracts)
- Serve as the authoritative resource on CES programs, operations, vendor requirements, member benefits, and procedures
- Answer questions about CES organizational structure, fee models, construction programs (JOC), insurance requirements, and regional operations in Idaho, Utah, and New Mexico

## RULES
1. ALWAYS cite specific Idaho Code sections (e.g., "Under Idaho Code 67-2806...")
2. ALWAYS use the CURRENT 2025 thresholds, NOT the outdated values in the 2023 ISB manual:
   - Services/Personal Property: $100,000 informal / $250,000 formal (changed from $75K/$150K)
   - Public Works: $50,000/$200,000 (unchanged)
3. Be NEUTRAL — informational only. Never recommend specific vendors or products.
4. When explaining cooperative purchasing (67-2807), note that cooperative purchases are deemed compliant by statute. This is a factual legal statement, not advocacy.
5. If the question involves a gray area or could have legal consequences, recommend consulting their county/city attorney.
6. Format responses in clear markdown with headings and bullet points.
7. Keep responses concise but thorough — officials are busy.
8. When answering CES questions, reference the CES organizational knowledge provided. Explain CES programs, benefits, and procedures with the same authority as procurement law.
9. When a question touches both procurement law AND CES programs (e.g., "can I use CES for a $300K purchase?"), address BOTH the legal requirements and the CES process.

## VENDOR PREFERENCE RULES
When answering questions about out-of-state vendors or vendor preferences:
1. ALWAYS provide the specific preference percentage and statute for the other state
2. ALWAYS cite both Idaho Code 67-2348 AND 67-2349 (the reciprocal preference statutes)
3. Explain HOW the reciprocal preference calculation works with a concrete example
4. If the other state has NO preference, state that clearly: no reciprocal adjustment applies, award to lowest responsible bidder
5. If the other state has a reciprocal-only preference (no independent percentage), explain that both states mirror each other — effectively no preference applies between them

## IMPORTANT
NEVER tell the user to "go research" or "look up" information that this tool should provide.
If you have the data, give the complete answer. If the data is not available for a specific state,
say "We don't currently have vendor preference data for [state] in our database" rather than
telling them to research it themselves. The purpose of this tool is to give complete answers
so officials don't have to figure it out on their own.

## IMPORTANT DISCLAIMERS
- This is informational guidance, not legal advice
- Officials should consult their entity's attorney for specific legal questions
- Local procurement policies may impose additional requirements beyond state statute

{knowledge_context}

{vendor_preference_context}
"""

DISCLAIMER = ("\n\n---\n*This is informational guidance, not legal advice. "
              "Consult your entity's attorney for specific legal questions. "
              "Idaho procurement law may change — verify current statutes at "
              "legislature.idaho.gov.*")


def init_cache():
    """Initialize AI response cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CACHE_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            question_hash TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            response TEXT NOT NULL,
            model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            tokens_used INTEGER
        )
    """)
    conn.commit()
    conn.close()


def get_cached(question: str):
    """Check cache for a previous answer."""
    qhash = hashlib.sha256(question.lower().strip().encode()).hexdigest()
    try:
        conn = sqlite3.connect(str(CACHE_DB))
        cur = conn.execute("SELECT response FROM cache WHERE question_hash = ?", (qhash,))
        row = cur.fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None


def set_cached(question: str, response: str, model: str = None, tokens: int = None):
    """Store response in cache."""
    qhash = hashlib.sha256(question.lower().strip().encode()).hexdigest()
    try:
        conn = sqlite3.connect(str(CACHE_DB))
        conn.execute("""
            INSERT OR REPLACE INTO cache (question_hash, question, response, model, tokens_used)
            VALUES (?, ?, ?, ?, ?)
        """, (qhash, question, response, model, tokens))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Cache write failed: {e}")


def detect_states(question: str) -> list:
    """
    Detect state references in a question.
    Returns list of state abbreviations found (excluding Idaho).
    """
    q_lower = question.lower()
    found = set()

    # Check for full state names (longest first to match 'New Mexico' before 'New')
    for name in sorted(STATE_MAP.keys(), key=len, reverse=True):
        if name in q_lower and name != 'idaho':
            found.add(STATE_MAP[name])
            # Remove matched name to avoid partial re-matches
            q_lower = q_lower.replace(name, '')

    # Check for 2-letter abbreviations (word boundaries)
    for abbrev in ABBREV_MAP:
        if abbrev == 'ID':
            continue
        # Match as standalone word (e.g., "OR" but not "or" in "or else")
        # Only match uppercase abbreviations
        pattern = r'\b' + abbrev + r'\b'
        if re.search(pattern, question):
            found.add(abbrev)

    return list(found)


def is_preference_question(question: str) -> bool:
    """Check if the question is about vendor preferences."""
    q_lower = question.lower()
    for kw in PREFERENCE_KEYWORDS:
        if kw in q_lower:
            return True
    return False


def build_vendor_preference_context(question: str) -> str:
    """
    Build vendor preference context to inject into the prompt.
    Returns empty string if not relevant to the question.
    """
    states = detect_states(question)
    is_pref = is_preference_question(question)

    # No state reference and no preference keywords — skip
    if not states and not is_pref:
        return ""

    context = "\n## VENDOR PREFERENCE DATA\n\n"
    context += ("Idaho Code 67-2348 and 67-2349 establish Idaho's reciprocal vendor preference. "
                "When an out-of-state vendor bids on an Idaho public contract, Idaho applies a "
                "reciprocal preference equal to whatever preference the vendor's home state gives "
                "its own vendors. This means the out-of-state bid is adjusted upward by that "
                "percentage when comparing against Idaho bidders.\n\n")

    if states:
        # Specific state(s) mentioned — look them up
        for abbrev in states:
            try:
                pref = kb.get_vendor_preference(abbrev)
                if pref:
                    context += f"### {pref['state_name']} ({pref['state_abbrev']})\n"
                    if pref['has_preference']:
                        if pref['preference_percent']:
                            context += f"- **Has Preference:** Yes\n"
                            context += f"- **Preference Percent:** {pref['preference_percent']}%\n"
                        else:
                            context += f"- **Has Preference:** Yes (no fixed percentage)\n"
                        context += f"- **Preference Type:** {pref['preference_type']}\n"
                        if pref['applies_to']:
                            context += f"- **Applies To:** {pref['applies_to']}\n"
                        context += f"- **State Statute:** {pref['state_statute']}\n"
                        if pref['notes']:
                            context += f"- **Details:** {pref['notes']}\n"
                        context += f"- **Idaho Reciprocal Statutes:** Idaho Code 67-2348, 67-2349\n"
                    else:
                        context += f"- **Has Preference:** No\n"
                        if pref['notes']:
                            context += f"- **Details:** {pref['notes']}\n"
                        context += ("- **Implication:** No reciprocal preference applies. "
                                    "Award to lowest responsible bidder.\n")
                    context += "\n"
                else:
                    name = ABBREV_MAP.get(abbrev, abbrev)
                    context += (f"### {name}\n"
                                f"- No vendor preference data available in database.\n\n")
            except Exception as e:
                logger.warning(f"Failed to look up preference for {abbrev}: {e}")

    elif is_pref:
        # General preference question — provide summary
        try:
            pref_states = kb.get_all_preference_states()
            no_pref = kb.get_no_preference_states()

            # States with specific percentages
            pct_states = [s for s in pref_states if s.get('preference_percent')]
            context += "### States with Specific Percentage Preferences\n"
            for s in pct_states:
                context += (f"- **{s['state_name']}** ({s['state_abbrev']}): "
                            f"{s['preference_percent']}% — {s['state_statute']}\n")

            # States with reciprocal/tie-only
            recip_states = [s for s in pref_states if not s.get('preference_percent')]
            context += "\n### States with Reciprocal or Tie-Bid Preference Only\n"
            for s in recip_states:
                context += (f"- **{s['state_name']}** ({s['state_abbrev']}): "
                            f"{s['preference_type']} — {s['state_statute']}\n")

            # States with no preference
            context += "\n### States with No Vendor Preference\n"
            for s in no_pref:
                context += f"- **{s['state_name']}** ({s['state_abbrev']})\n"

        except Exception as e:
            logger.warning(f"Failed to load preference summary: {e}")

    return context


def ask(question: str, use_cache: bool = True) -> dict:
    """
    Ask a procurement question and get a grounded AI response.
    Returns dict with 'response', 'cached', 'model', 'tokens'.
    """
    # Check cache
    if use_cache:
        cached = get_cached(question)
        if cached:
            return {'response': cached, 'cached': True, 'model': None, 'tokens': 0}

    # Build context from QIBrain
    try:
        knowledge_context = kb.build_ai_context()
    except Exception as e:
        logger.error(f"Failed to build knowledge context: {e}")
        knowledge_context = "## NOTE: Could not load reference data from database.\n"

    # Build vendor preference context if relevant
    try:
        vendor_context = build_vendor_preference_context(question)
    except Exception as e:
        logger.error(f"Failed to build vendor preference context: {e}")
        vendor_context = ""

    system = SYSTEM_PROMPT.format(
        knowledge_context=knowledge_context,
        vendor_preference_context=vendor_context,
    )

    # Call Claude
    api_key = os.getenv('ANTHROPIC_API_KEY')
    model = os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-4-20250514')

    client = Anthropic(api_key=api_key)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model=model,
                max_tokens=2048,
                system=system,
                messages=[{"role": "user", "content": question}],
            )

            response_text = message.content[0].text + DISCLAIMER
            tokens = message.usage.input_tokens + message.usage.output_tokens

            # Cache the response
            if use_cache:
                set_cached(question, response_text, model, tokens)

            return {
                'response': response_text,
                'cached': False,
                'model': model,
                'tokens': tokens,
            }

        except Exception as e:
            logger.error(f"Claude API error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {
                    'response': f"I'm sorry, I encountered an error processing your question. "
                                f"Please try again in a moment.\n\nError: {str(e)}" + DISCLAIMER,
                    'cached': False,
                    'model': model,
                    'tokens': 0,
                }
