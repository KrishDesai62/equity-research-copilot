"""Audited Chapter 7/name-annotation corrections without rewriting prior studies."""
from hashlib import sha256
import json
from pathlib import Path
import re
from unittest.mock import patch

import pandas as pd

from src.data_layer.filing_changes import normalize_filing
from src.data_layer.research_dataset import write_json
from src.validation import sec_disclosure_admission as previous
from src.validation.adjudicate_public_events import DATE, OTHER_ACTION, parent_in_subject
from src.validation.cohort_v3_data import digest
from src.validation.sec_disclosure_data import TARGET

CONFIG = Path('config/sec_disclosure_expansion/admission_v3.json')
ROOT = Path('data/processed/validation/sec_disclosure_admission_v3')
BASE_PARSER = previous.corroborate_v2
LIQUIDATION = re.compile(r'\bon\s+(?P<date>' + DATE + r')\b(?P<subject>.{0,1500}?)\bfiled\s+for\s+liquidation\s+under\s+chapter\s+7\b', re.I)


def legal_name(name):
    # These are SEC name annotations, not part of the issuer's legal suffix.
    return re.sub(r'\s*/(?:DE|NV)/?\s*$', '', name, flags=re.I).strip()


def corroborate_v3(text, *, filed, names, maximum_lag=200):
    names = [legal_name(n) for n in names]
    assertions = BASE_PARSER(text, filed=filed, names=names, maximum_lag=maximum_lag)
    plain = re.sub(r'\s+', ' ', normalize_filing(text))
    for anchor in re.finditer(r'\bon\s+' + DATE, plain, re.I):
        match = LIQUIDATION.match(plain, anchor.start())
        if not match or OTHER_ACTION.search(match.group('subject')) or re.search(DATE, match.group('subject'), re.I):
            continue
        day = pd.Timestamp(match.group('date'))
        if not pd.Timestamp(filed) - pd.Timedelta(days=maximum_lag) <= day <= pd.Timestamp(filed):
            continue
        end = min(len(plain), match.end() + 240)
        assertions.append({'petition_date': day.date().isoformat(), 'chapter': '7',
            'issuer_explicitly_included': parent_in_subject(match.group('subject'), names),
            'statement_start': match.start(), 'statement_end': end,
            'statement_sha256': sha256(plain[match.start():end].encode()).hexdigest()})
    return assertions


def run():
    base = previous.run()  # Verify, do not overwrite earlier source admissions.
    config = json.loads(CONFIG.read_text())
    ROOT.mkdir(parents=True, exist_ok=True)
    saved_path = ROOT / 'report.json'
    if saved_path.exists():
        saved = json.loads(saved_path.read_text())
        if saved['previous_report_sha256'] != digest(previous.ROOT / 'report.json') or saved['implementation_sha256'] != digest(__file__) or saved['config_sha256'] != digest(CONFIG):
            raise ValueError('Frozen V3 admission changed')
        for role, expected in saved['output_hashes'].items():
            if digest(ROOT / (role + '.csv')) != expected:
                raise ValueError('Frozen V3 admission output changed')
        if digest(ROOT / 'evidence.json') != saved['evidence_sha256']:
            raise ValueError('Frozen V3 admission evidence changed')
        return saved
    events, proofs = {}, []
    # Only swap the parser during this sequential source review. It is restored
    # immediately; the previous parser's file, data and report are untouched.
    with patch.object(previous, 'corroborate_v2', corroborate_v3):
        for cik, source in base['checkpoint_inputs'].items():
            events[cik], evidence = previous.company_proofs(cik, Path(source['path']), config['maximum_statement_lag_days'])
            proofs.extend(evidence)
    counts, changes, outputs = {}, {}, {}
    for role in ['training', 'calibration', 'final']:
        frame = pd.read_csv(previous.ROOT / (role + '.csv'), dtype={'cik': str})
        updated, changes[role] = previous.apply_evidence(frame, events)
        updated.to_csv(ROOT / (role + '.csv'), index=False)
        outputs[role] = digest(ROOT / (role + '.csv'))
        counts[role] = {'rows': len(updated), 'positive_companies': int(updated.loc[updated[TARGET].eq(1), 'cik'].nunique()),
            'positive_rows': int(updated[TARGET].eq(1).sum()), 'negative_rows': int(updated[TARGET].eq(0).sum()),
            'unknown_rows': int(updated[TARGET].isna().sum())}
    write_json(ROOT / 'evidence.json', proofs)
    result = {'config_sha256': digest(CONFIG), 'implementation_sha256': digest(__file__),
        'previous_report_sha256': digest(previous.ROOT / 'report.json'), 'output_hashes': outputs,
        'evidence_sha256': digest(ROOT / 'evidence.json'), 'counts': counts, 'changes': changes,
        'new_negative_labels': 0, 'independent_human_review': False, 'final_predictions_created': False,
        'production_ready': False}
    write_json(saved_path, result)
    return result


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
