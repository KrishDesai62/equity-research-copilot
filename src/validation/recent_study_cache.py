"""Reuse exactly matching, source-verified legacy feature caches atomically."""
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from src.validation.cohort_v3_data import digest
from src.validation import filing_text_signals as text
from src.validation import contextual_filing_signals as context


def run():
    root=Path('data/processed/validation/recent_study_features_20261019')
    example=next((root/'documents').glob('*.json'))
    inputs=json.loads(example.read_text())['inputs']
    for p,sha in inputs.items():
        if digest(p)!=sha:raise ValueError('Newer extraction input changed')
    market=Path('data/processed/validation/market_context_signals_20261019')
    report=json.loads((market/'report.json').read_text());corrections=market/'liquidity_corrections.json'
    if digest(corrections)!=report['outputs'][corrections.name]:raise ValueError('Archived correction table changed')
    changes={r['document']:r for r in json.loads(corrections.read_text())}
    counts={'reused':0,'already_present':0,'not_matching':0}
    for marker in sorted((context.ROOT/'documents').glob('*.json')):
        dest=root/'documents'/marker.name
        if dest.exists():counts['already_present']+=1;continue
        original=text.ROOT/'documents'/marker.name
        record=json.loads(marker.read_text());base=json.loads(original.read_text())
        expected={'implementation_sha256':digest(context.__file__),'source_checkpoint_sha256':digest(original),'text_kernel_sha256':digest(text.__file__)}
        if record['inputs']!=expected or base['implementation_sha256']!=digest(text.__file__) or base['config_sha256']!=digest(text.CONFIG):
            counts['not_matching']+=1;continue
        source=base['source']
        if source!=record['source'] or digest(source['path'])!=source['sha256']:raise ValueError('Archived filing source changed')
        if record['evidence']['document']!=base['document']:raise ValueError('Archived document identity mismatch')
        values=dict(record['values'])
        if marker.stem in changes:
            correction=changes[marker.stem]
            if correction['source']!=source or any(values[k]!=v for k,v in correction['old'].items()):raise ValueError('Archived liquidity correction identity changed')
            values.update(correction['new'])
        result={'inputs':inputs,'document':base['document'],'source':source,'values':{**base['values'],**values}}
        # Atomic publication is safe alongside the sequential reader: it sees
        # either the absent file or the complete equivalent cache record.
        with NamedTemporaryFile(mode='w',dir=dest.parent,delete=False) as handle:
            json.dump(result,handle,indent=2);temporary=Path(handle.name)
        os.replace(temporary,dest);counts['reused']+=1
    return counts

if __name__=='__main__':print(json.dumps(run()))
