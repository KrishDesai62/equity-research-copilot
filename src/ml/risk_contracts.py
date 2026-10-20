"""Explicit prediction meanings for production risk tracks."""
TARGETS = {
    'distress': {
        'target_bankruptcy_next_1yr': 'Company bankruptcy within the next year',
        'target_distressed_1yr': 'One-year distress outcome as defined by the reviewed release dataset',
        'target_sec_disclosed_petition_1yr': 'Issuer Chapter 7/11 petition disclosed in SEC filings within one year; not exhaustive court-bankruptcy coverage',
    },
    'manipulation': {
        'target_aaer_enforcement_flag': 'SEC accounting-enforcement outcome; not proof of every accounting manipulation',
    },
}


def target_definition(track, target):
    try:
        return TARGETS[track][target]
    except (KeyError, TypeError) as exc:
        raise ValueError(f'Unsupported prediction target for {track}: {target}') from exc
