"""Protect the new study's timing, sampling, and immutable evidence boundaries."""
import json
from pathlib import Path
import pytest
from src.validation.recent_study_sources import chronology,CONFIG
from src.validation.recent_study_data import inclusion_probability,cached_fetch
from src.validation.cohort_v3_data import digest


def test_outcomes_mature_before_next_forecast():
    config=json.loads(CONFIG.read_text())
    assert chronology(config)==('2022-12-17','2024-12-17')
    config['calibration_anchor']='2023-12-31'
    with pytest.raises(ValueError,match='postdate'):chronology(config)


def test_three_stage_sampling_union():
    old={'companies':[]};history={'sec_inventory_proxies':[],'extension_frame':['a']}
    assert inclusion_probability('a',old,history,{'a'},set(),.05)==pytest.approx(1-.9*.9*.95)
    assert inclusion_probability('a',old,history,{'a'},{'a'},.05)==1
    old['companies']=[{'cik':'a','inclusion_probability':1}]
    assert inclusion_probability('a',old,history,set(),set(),.05)==1


def test_cached_source_is_verified_before_linking(tmp_path):
    source=tmp_path/'source';source.write_bytes(b'public evidence')
    pool={'url':{'url':'url','path':str(source),'sha256':digest(source)}}
    target=tmp_path/'new'/'source'
    proof=cached_fetch('url',target,pool)
    assert target.is_symlink() and proof['sha256']==digest(source)
    source.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):cached_fetch('url',target,pool)


def test_new_plan_company_reservations():
    plan=json.loads(Path('config/recent_study_20261019/acquisition.json').read_text())
    train={c['cik'] for c in plan['training']};cal={c['cik'] for c in plan['calibration']}
    final=set(plan['reserved_final_ids'])
    assert not train&cal and not (train|cal)&final
    assert not train&set(plan['prior_development_excluded_from_training'])
    assert all(0<c['inclusion_probability']<=1 for role in ['training','calibration'] for c in plan[role])
