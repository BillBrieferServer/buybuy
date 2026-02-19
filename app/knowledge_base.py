"""
knowledge_base.py
QIBrain PostgreSQL queries for procurement reference data.
"""

import os
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DB_CONFIG = {
    'dbname': os.getenv('DB_NAME', 'qibrain'),
    'user': os.getenv('DB_USER', 'quietimpact_user'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST', 'host.docker.internal'),
    'port': os.getenv('DB_PORT', '5432'),
}


@contextmanager
def get_db():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        yield conn
    finally:
        conn.close()


def get_threshold(purchase_type: str, amount: int, entity_type: str = 'all'):
    """Look up the required procurement process for a purchase."""
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT process_required, description, statute_section, category,
                   min_amount, max_amount, notes
            FROM procurement_thresholds
            WHERE purchase_type = %s
              AND (entity_type = %s OR entity_type = 'all')
              AND min_amount <= %s
              AND (max_amount >= %s OR max_amount IS NULL)
            ORDER BY entity_type DESC
            LIMIT 1
        """, (purchase_type, entity_type, amount, amount))
        return cur.fetchone()


def get_all_thresholds():
    """Get all threshold entries for context."""
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT entity_type, purchase_type, category, min_amount, max_amount,
                   process_required, description, statute_section, notes
            FROM procurement_thresholds
            ORDER BY purchase_type, min_amount
        """)
        return cur.fetchall()


def get_compliance_requirements(amount: int = None, entity_type: str = 'all',
                                 purchase_type: str = None):
    """Get applicable compliance requirements for a purchase scenario."""
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT requirement_name, trigger_condition, required_action,
                   statute_section, penalty_text, notes
            FROM compliance_requirements
            WHERE (applies_to = %s OR applies_to = 'all')
            ORDER BY requirement_name
        """, (entity_type,))
        all_reqs = cur.fetchall()

    # Filter based on specifics
    applicable = []
    for req in all_reqs:
        name = req['requirement_name']
        # Israel boycott only for $100K+
        if name == 'Anti-Boycott Israel Certification' and amount and amount < 100000:
            continue
        # Prevailing wage only for public works
        if name == 'Prevailing Wage' and purchase_type and purchase_type != 'public_works':
            continue
        # QBS only for professional services
        if name == 'Design Professional QBS' and purchase_type and purchase_type != 'professional_services':
            continue
        applicable.append(req)

    return applicable


def get_statute(section: str):
    """Get a specific statute summary."""
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT statute_section, title, subject_area, summary, key_points
            FROM procurement_statutes
            WHERE statute_section = %s
        """, (section,))
        return cur.fetchone()


def get_all_statutes():
    """Get all statutes for context."""
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT statute_section, title, subject_area, summary, key_points
            FROM procurement_statutes
            ORDER BY statute_section
        """)
        return cur.fetchall()


def get_ces_categories():
    """Get CES cooperative contract categories."""
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT category_name, description, ordering_method
            FROM ces_contract_categories
            ORDER BY category_name
        """)
        return cur.fetchall()


def build_ai_context():
    """Build the full knowledge context for AI prompts."""
    thresholds = get_all_thresholds()
    statutes = get_all_statutes()
    ces = get_ces_categories()

    context = "## IDAHO PROCUREMENT THRESHOLDS (Current as of 2025)\n\n"
    context += "IMPORTANT: The Idaho Legislature amended 67-2806 in 2025. "
    context += "Services/Personal Property thresholds changed from $75K/$150K to $100K/$250K. "
    context += "Public works thresholds ($50K/$200K) are unchanged.\n\n"

    for t in thresholds:
        max_str = f"${t['max_amount']:,}" if t['max_amount'] else "no limit"
        context += f"- {t['purchase_type']}/{t['category']}: "
        context += f"${t['min_amount']:,} - {max_str} → {t['process_required']} "
        context += f"({t['statute_section']})\n"
        if t.get('notes'):
            context += f"  Note: {t['notes']}\n"

    context += "\n## IDAHO PROCUREMENT STATUTES\n\n"
    for s in statutes:
        context += f"### {s['statute_section']} — {s['title']}\n"
        context += f"{s['summary']}\n"
        if s.get('key_points'):
            for p in s['key_points']:
                context += f"  - {p}\n"
        context += "\n"

    context += "## CES COOPERATIVE CONTRACT CATEGORIES\n\n"
    context += "CES (Cooperative Educational Services) provides pre-competed contracts "
    context += "available to Idaho counties, cities, school districts, and other public entities. "
    context += "Under Idaho Code 67-2807, purchases through CES are deemed compliant.\n\n"
    for c in ces:
        context += f"- **{c['category_name']}** ({c['ordering_method']}): {c['description']}\n"

    return context


def get_vendor_preference(state_input: str) -> dict:
    """
    Look up a state's vendor preference policy.
    Accept state name or abbreviation, case-insensitive.
    """
    state_input = state_input.strip()
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        if len(state_input) == 2:
            cur.execute(
                'SELECT * FROM vendor_preference_states WHERE UPPER(state_abbrev) = UPPER(%s)',
                (state_input,))
        else:
            cur.execute(
                'SELECT * FROM vendor_preference_states WHERE LOWER(state_name) = LOWER(%s)',
                (state_input,))
        return cur.fetchone()


def get_all_preference_states() -> list:
    """
    Return list of all states that have vendor preferences.
    Used for general vendor preference questions.
    """
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            'SELECT state_name, state_abbrev, preference_percent, preference_type, '
            'applies_to, state_statute, notes '
            'FROM vendor_preference_states WHERE has_preference = TRUE '
            'ORDER BY state_name')
        return cur.fetchall()


def get_no_preference_states() -> list:
    """Return list of states with no vendor preference."""
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            'SELECT state_name, state_abbrev, notes '
            'FROM vendor_preference_states WHERE has_preference = FALSE '
            'ORDER BY state_name')
        return cur.fetchall()
