"""Prepare a sampled source-review queue without changing frozen label approvals."""
import argparse
import hashlib
import html
import json
from pathlib import Path

from html.parser import HTMLParser


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def plain_text(value):
    parser = PlainText()
    parser.feed(value)
    return " ".join(" ".join(parser.parts).split())

PACKET = Path('data/processed/validation/final_outcome_audit_v2_20261019')
BACKLOG = Path('data/processed/validation/reconciled_coverage_20261019/remaining_current_reports.json')
DEFAULT_OUTPUT = Path('data/processed/validation/rapid_review_v1')


def sample_units(units, size, seed):
    """Stable uniform hash ordering, independent of labels' model predictions."""
    if size < 1:
        raise ValueError('Sample size must be positive')
    return sorted(units, key=lambda u: hashlib.sha256(f'{seed}:{u}'.encode()).hexdigest())[:size]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(packet=PACKET, output=DEFAULT_OUTPUT, sample_size=60, seed='rapid-review-v1'):
    manifest = json.loads((packet / 'manifest.json').read_text())
    source = packet / 'pending_final_reviews.jsonl'
    if sha(source) != manifest['packet_sha256']:
        raise ValueError('Frozen packet hash mismatch')
    tasks = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    negatives = [t['unit_id'] for t in tasks if t['automated_proposed_label'] == 0]
    sampled = set(sample_units(negatives, sample_size, seed))
    backlog = json.loads(BACKLOG.read_text()) if BACKLOG.exists() else None
    affected = {}
    for filing in backlog or []:
        for unit in filing['affected_units']:
            affected.setdefault(unit, []).append(filing)
    rows = []
    for task in tasks:
        unit = task['unit_id']
        checkpoint = Path(task['source_checkpoint']['path'])
        issues = []
        content = {}
        if not checkpoint.exists() or sha(checkpoint) != task['source_checkpoint']['sha256']:
            issues.append('checkpoint missing or hash mismatch')
        else:
            content = json.loads(checkpoint.read_text())
            if content.get('row', {}).get('unit_id') != unit:
                issues.append('checkpoint unit mismatch')
            if content.get('unresolved_accessions'):
                issues.append('unresolved accessions')
        if not task['prediction_date'] < task['horizon_end'] <= task['evidence_cutoff']:
            issues.append('invalid horizon dates')
        evidence = []
        for item in task.get('positive_statement_evidence', []):
            assertion = item['assertion']
            path = Path(item['source']['path'])
            excerpt = ''
            if not path.exists() or sha(path) != item['source']['sha256']:
                issues.append('positive document missing or hash mismatch')
            else:
                text = plain_text(path.read_text())
                pos = text.casefold().find('petition date')
                if pos < 0:
                    pos = text.casefold().find('chapter 11')
                excerpt = text[max(0, pos - 350):max(0, pos) + 1100] if pos >= 0 else text[:1400]
            if not task['prediction_date'] < assertion['petition_date'] <= task['horizon_end']:
                issues.append('petition outside horizon')
            if assertion.get('chapter') not in ('7', '11') or assertion.get('issuer_explicitly_included') is not True:
                issues.append('chapter or issuer assertion requires resolution')
            if assertion.get('filing_date', '9999') > task['evidence_cutoff']:
                issues.append('source after cutoff')
            evidence.append({'url': item['source']['url'], 'excerpt': excerpt, 'automated_assertion': assertion})
        positive = task['automated_proposed_label'] == 1
        if positive and not evidence:
            issues.append('missing positive evidence')
        if not positive and content.get('reporting_followup_complete') is not True:
            issues.append('incomplete reporting follow-up')
        reasons = (['proposed bankruptcy'] if positive else []) + (['random negative audit'] if unit in sampled else [])
        if issues:
            reasons.append('automated check exception')
        rows.append({**task, 'review_reasons': reasons, 'automated_check_exceptions': issues,
                     'selected_for_human_review': bool(reasons), 'evidence_preview': evidence,
                     'known_unreviewed_current_reports': len(affected.get(unit, [])) if backlog is not None else None,
                     'coverage_status': 'not established by this workflow',
                     'negative_coverage_examples': affected.get(unit, [])[:3]})
    output.mkdir(parents=True, exist_ok=True)
    (output / 'queue.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    selected = [r for r in rows if r['selected_for_human_review']]
    summary = {'policy': 'sampled quality-control v1; advisory, not label admission',
               'packet_sha256': manifest['packet_sha256'], 'seed': seed,
               'negative_population': len(negatives), 'negative_sample': len(sampled),
               'total_units': len(rows), 'human_review_units': len(selected),
               'second_review_policy': 'Disagreements, uncertain cases, and all sampled negatives for independent quality control; positives need one source review unless disputed.',
               'expansion_rule': 'Any sample error or uncertain result requires investigation and expanded review before trusting negative labels.',
               'coverage_backlog_sha256': sha(BACKLOG) if BACKLOG.exists() else None,
               'production_approval': False, 'labels_approved_by_sampling': 0}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    body = ['<html><head><meta charset="utf-8"><title>Source review queue</title></head><body>',
            '<h1>Source review queue</h1><p>Automated excerpts and assertions are navigation aids. Open primary sources. No labels are approved here. Full-horizon negative coverage remains required; missing evidence means uncertain.</p>',
            '<table border="1"><tr><th>Company / unit</th><th>Why review</th><th>Compare with primary source</th><th>Coverage / exceptions</th></tr>']
    for row in selected:
        links = ''.join('<p><a href="' + html.escape(e['url'], quote=True) + '">SEC source</a></p><blockquote>' + html.escape(e['excerpt']) + '</blockquote><p>Automated assertion: ' + html.escape(str({k:e['automated_assertion'].get(k) for k in ('petition_date','chapter','issuer_explicitly_included')})) + '</p>' for e in row['evidence_preview'])
        if not links:
            links = '<a href="https://www.sec.gov/edgar/browse/?CIK=' + row['cik'] + '">SEC issuer filings</a>'
        links += ''.join('<p><a href="' + html.escape(f['filing_index_url'], quote=True) + '">Known coverage gap: ' + html.escape(f['filing_date']) + '</a></p>' for f in row['negative_coverage_examples'])
        body.append('<tr><td>' + html.escape(row['company_name'] + ' / ' + row['unit_id']) + '</td><td>' + html.escape(', '.join(row['review_reasons'])) + '</td><td>' + links + '</td><td>' + html.escape(str(row['automated_check_exceptions'])) + '<p>Known unreviewed current reports: ' + str(row['known_unreviewed_current_reports']) + '. This is not an exhaustive coverage audit.</p></td></tr>')
    (output / 'review.html').write_text(''.join(body) + '</table></body></html>')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--sample-size', type=int, default=60)
    parser.add_argument('--seed', default='rapid-review-v1')
    args = parser.parse_args()
    print(json.dumps(build(output=args.output, sample_size=args.sample_size, seed=args.seed), indent=2))
