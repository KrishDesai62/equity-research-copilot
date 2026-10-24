"""Publish supplemental source-screening results without converting them to approvals."""
import html
import json
from pathlib import Path

from src.validation.coverage_completion import ROOT
from src.validation.cohort_v3_data import digest


def run():
    summary=json.loads((ROOT/'summary.json').read_text())
    records=[json.loads(p.read_text()) for p in sorted((ROOT/'companies').glob('*.json'))]
    queue=[]
    pages=['<html><meta charset="utf-8"><title>Supplemental exhibit review</title>',
           '<h1>Supplemental exhibit review</h1>',
           '<p>2025 issuer petition horizon; evidence through July 19, 2026. Automated screening only. '
           'Previous human reviews excluded these exhibits. All keyword matches are shown; absence is not approval.</p>']
    for company in records:
        entries=[]
        for document in company['documents']:
            for exhibit in document.get('exhibits',[]):
                if exhibit['scope']!='exhibit':continue
                if not exhibit.get('keyword_matches',0) and exhibit['status']=='screened':continue
                entry=dict(cik=company['cik'],unit_id=company['unit_id'],accession=document['accession'],
                           submission_url=document['url'],submission_sha256=document.get('sha256'),
                           **exhibit,human_review_status='pending',approved_label=None)
                entries.append(entry);queue.append(entry)
        if entries:
            pages.append('<details><summary>'+html.escape(company['cik'])+f' — {len(entries)} exhibits requiring interpretation</summary>')
            for entry in entries:
                pages.append('<p><a href="'+html.escape(entry['submission_url'],quote=True)+'">'+html.escape(entry['accession']+' '+entry['type']+' '+entry['filename'])+'</a> — '+html.escape(entry['status'])+'</p>')
                for excerpt in entry.get('excerpts',[]):
                    pages.append('<blockquote>'+html.escape(excerpt)+'</blockquote>')
            pages.append('</details>')
    pages.append('</html>')
    (ROOT/'exhibit_review.html').write_text('\n'.join(pages))
    (ROOT/'exhibit_review_queue.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in queue))
    result=dict(summary,exhibits_requiring_interpretation=len(queue),
                unique_exhibits_requiring_interpretation=len({e['content_sha256'] for e in queue}),
                displayed_excerpt_count=sum(len(e.get('excerpts',[])) for e in queue),
                source_summary_sha256=digest(ROOT/'summary.json'),
                review_html_sha256=digest(ROOT/'exhibit_review.html'),
                review_queue_sha256=digest(ROOT/'exhibit_review_queue.jsonl'))
    (ROOT/'review_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
