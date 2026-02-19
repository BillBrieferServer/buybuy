"""
procurement.py
Business logic for procurement decision tree and compliance checklists.
All data driven from QIBrain — no hardcoded thresholds.
"""

from . import knowledge_base as kb


PURCHASE_TYPES = {
    'goods_services': {
        'label': 'Goods, Services, or Personal Property',
        'db_type': 'services_personal_property',
        'description': 'Equipment, supplies, technology, maintenance, consulting, etc.'
    },
    'public_works': {
        'label': 'Public Works / Construction',
        'db_type': 'public_works',
        'description': 'Building construction, renovation, infrastructure, road work, etc.'
    },
    'professional_services': {
        'label': 'Professional Services (Design)',
        'db_type': 'professional_services',
        'description': 'Architects, engineers, surveyors, construction managers'
    },
    'cooperative': {
        'label': 'Cooperative Purchase (CES, Sourcewell, etc.)',
        'db_type': 'cooperative_purchase',
        'description': 'Purchase through a cooperative purchasing program'
    },
}

ENTITY_TYPES = [
    ('county', 'County'),
    ('city', 'City'),
    ('school_district', 'School District'),
    ('special_district', 'Special District / Other'),
]


def evaluate_procurement(purchase_type: str, amount: int, entity_type: str = 'all',
                          is_cooperative: bool = False):
    """
    Evaluate a procurement scenario and return the required process.
    Returns dict with process, description, statute, compliance items.
    """
    result = {
        'purchase_type': purchase_type,
        'amount': amount,
        'entity_type': entity_type,
        'is_cooperative': is_cooperative,
    }

    # If cooperative, that overrides everything
    if is_cooperative:
        threshold = kb.get_threshold('cooperative_purchase', amount, entity_type)
    else:
        db_type = PURCHASE_TYPES.get(purchase_type, {}).get('db_type', purchase_type)
        threshold = kb.get_threshold(db_type, amount, entity_type)

    if threshold:
        result['process'] = threshold['process_required']
        result['process_description'] = threshold['description']
        result['statute'] = threshold['statute_section']
        result['category'] = threshold['category']
        result['notes'] = threshold.get('notes')
    else:
        result['process'] = 'unknown'
        result['process_description'] = 'Could not determine required process. Consult your attorney.'
        result['statute'] = None

    # Get compliance requirements
    db_type = PURCHASE_TYPES.get(purchase_type, {}).get('db_type', purchase_type)
    result['compliance'] = kb.get_compliance_requirements(
        amount=amount, entity_type=entity_type, purchase_type=db_type
    )

    return result


def generate_checklist(purchase_type: str, amount: int, entity_type: str = 'all',
                        is_cooperative: bool = False):
    """Generate a compliance checklist for a procurement scenario."""
    evaluation = evaluate_procurement(purchase_type, amount, entity_type, is_cooperative)

    checklist = []

    # Threshold determination
    checklist.append({
        'item': 'Threshold Determination',
        'description': f"Purchase amount: ${amount:,}. "
                       f"Process required: {evaluation.get('process', 'unknown')}.",
        'statute': evaluation.get('statute'),
        'status': 'info',
    })

    # Bidding requirement
    process = evaluation.get('process', '')
    if process == 'exempt' or process == 'cooperative_exempt':
        checklist.append({
            'item': 'Competitive Bidding',
            'description': 'Not required for this purchase.',
            'statute': evaluation.get('statute'),
            'status': 'not_required',
        })
    elif 'bid' in process or 'competitive' in process:
        checklist.append({
            'item': 'Competitive Bidding',
            'description': evaluation.get('process_description', ''),
            'statute': evaluation.get('statute'),
            'status': 'required',
        })
    elif process == 'qbs_selection':
        checklist.append({
            'item': 'Qualifications-Based Selection (QBS)',
            'description': evaluation.get('process_description', ''),
            'statute': '67-2320',
            'status': 'required',
        })

    # Cooperative purchasing note
    if is_cooperative:
        checklist.append({
            'item': 'Cooperative Program Authorization',
            'description': 'Board must authorize participation in the cooperative program. '
                           'Once authorized, purchase is deemed compliant per 67-2807.',
            'statute': '67-2807',
            'status': 'required',
        })

    # Compliance requirements
    for req in evaluation.get('compliance', []):
        status = 'required'
        checklist.append({
            'item': req['requirement_name'],
            'description': req['required_action'],
            'statute': req['statute_section'],
            'status': status,
            'penalty': req.get('penalty_text'),
        })

    # Board approval
    if amount and amount >= 50000:
        checklist.append({
            'item': 'Board/Council Approval',
            'description': 'Purchases of this size typically require governing body approval. '
                           'Check your entity\'s local procurement policy for specific thresholds.',
            'statute': None,
            'status': 'recommended',
        })

    return {
        'evaluation': evaluation,
        'checklist': checklist,
    }
