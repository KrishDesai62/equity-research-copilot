"""Admit independently reviewed outcomes; never infer health from registry absence."""
from datetime import date
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlparse
import json

import pandas as pd


def verify_review(record, *, as_of):
    """Validate a review contract, not the truth of a human's legal conclusion.

    Two reviewers attest company identity and outcome coverage. Evidence files
    must already have been acquired and hashed; this function has no network IO.
    """
    cik = str(record.get('cik', ''))
    if len(cik) != 10 or not cik.isdigit() or int(cik) <= 0:
        raise ValueError('Reviewed outcomes need a ten-digit positive CIK')
    prediction = date.fromisoformat(record['prediction_date'])
    horizon = (pd.Timestamp(prediction) + pd.DateOffset(years=1)).date()
    cutoff = date.fromisoformat(as_of)
    label = record.get('target')
    if type(label) is not int or label not in {0, 1}:
        raise ValueError('Outcome must be reviewed binary 0/1, never unknown')
    reviewers = record.get('reviewers', [])
    if not isinstance(reviewers, list) or any(not isinstance(r, str) or not r.strip() for r in reviewers) or len(reviewers) < 2 or len(set(reviewers)) != len(reviewers):
        raise ValueError('Two distinct named reviewers are required')
    reviewed = date.fromisoformat(record['reviewed_on'])
    if reviewed > cutoff or not record.get('identity_verified') is True:
        raise ValueError('Review must be completed by as-of and verify the legal entity')
    if record.get('prior_bankruptcy_excluded') is not True:
        raise ValueError('Prior cases must be explicitly checked')
    if label == 1:
        event = date.fromisoformat(record['event_date'])
        if not prediction < event <= min(horizon, cutoff, reviewed) or record.get('chapter') not in {'7', '11'}:
            raise ValueError('Positive needs a dated Chapter 7/11 event inside the horizon')
        available = event
    else:
        through = date.fromisoformat(record['coverage_through'])
        if horizon > min(cutoff, reviewed, through) or record.get('complete_event_coverage') is not True:
            raise ValueError('Negative requires completed full-year event coverage')
        if not str(record.get('coverage_method', '')).strip() or record.get('event_date'):
            raise ValueError('Negative needs documented coverage method and no bankruptcy event')
        available = horizon
    sources = record.get('evidence', [])
    if not sources or not any(e.get('kind') == ('bankruptcy_event' if label else 'complete_outcome_coverage') for e in sources):
        raise ValueError('Missing outcome-specific evidence')
    for evidence in sources:
        url = urlparse(evidence['url'])
        if url.scheme != 'https' or not url.netloc:
            raise ValueError('Evidence needs a public HTTPS source URL')
        if date.fromisoformat(evidence['retrieved_on']) > min(cutoff, reviewed):
            raise ValueError('Evidence was unavailable when reviewed')
        actual = sha256(Path(evidence['path']).read_bytes()).hexdigest()
        if evidence.get('sha256') != actual:
            raise ValueError('Reviewed evidence hash mismatch')
    return {'unit_id': f'{cik}@{prediction}', 'cik': cik, 'prediction_date': prediction.isoformat(),
            'target_bankruptcy_next_1yr': label, 'label_available_date_1yr': available.isoformat(),
            'review_sha256': sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()}


def load_reviews(path, *, as_of):
    rows = [verify_review(json.loads(line), as_of=as_of) for line in Path(path).read_text().splitlines() if line.strip()]
    frame = pd.DataFrame(rows)
    if frame.empty or frame.unit_id.duplicated().any():
        raise ValueError('Reviews must be nonempty and have unique company/prediction units')
    return frame
