"""
advisor.py
Claude API integration for procurement Q&A.
Grounded responses with statute citations.
"""

import os
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

SYSTEM_PROMPT = """You are the Idaho Procurement Advisor, an AI assistant that helps Idaho public officials (county clerks, commissioners, city clerks, school district administrators) navigate public procurement decisions.

## YOUR ROLE
- Provide accurate, helpful guidance on Idaho procurement law and procedures
- Ground every answer in specific Idaho Code sections
- Explain complex procurement rules in plain language
- Help officials determine the correct procurement process for their situation
- Explain cooperative purchasing options (CES, State of Idaho contracts)

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

## IMPORTANT DISCLAIMERS
- This is informational guidance, not legal advice
- Officials should consult their entity's attorney for specific legal questions
- Local procurement policies may impose additional requirements beyond state statute

{knowledge_context}
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

    system = SYSTEM_PROMPT.format(knowledge_context=knowledge_context)

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
