"""Public source preparation for the fixed, chronologically valid newer study."""
import json
from pathlib import Path
from unittest.mock import patch
import zipfile

import pandas as pd

from src.data_layer.research_dataset import write_json
from src.validation.cohort_v3_data import digest
from src.validation.public_source_fetch import coordinated_fetch
from src.validation import sec_event_inventory as search

ROOT=Path('data/processed/validation/recent_study_sources_20261019')
RAW=Path('data/raw/recent_study_20261019')
CONFIG=Path('config/recent_study_20261019/protocol.json')


def chronology(config):
    grace=pd.Timedelta(days=config['reporting_grace_days'])
    training_cutoff=(pd.Timestamp(config['new_training_anchor'])+pd.DateOffset(years=1)+grace).date().isoformat()
    calibration_cutoff=(pd.Timestamp(config['calibration_anchor'])+pd.DateOffset(years=1)+grace).date().isoformat()
    if not training_cutoff<config['calibration_anchor'] or not calibration_cutoff<config['reserved_final_anchor']:
        raise ValueError('Outcome admission would postdate the next forecast')
    return training_cutoff,calibration_cutoff


def inventory(label,start,end):
    folder=ROOT/label;folder.mkdir(parents=True,exist_ok=True)
    query={'start':start,'end':end,'endpoint':'https://efts.sec.gov/LATEST/search-index','forms':'8-K','item':'1.03','page_size':100}
    saved=folder/'report.json'
    if saved.exists():
        record=json.loads(saved.read_text())
        if record['query']!=query or record['implementation_sha256']!=digest(__file__):raise ValueError('Frozen recent discovery changed')
        if digest(folder/'candidates.csv')!=record['candidates_sha256']:raise ValueError('Recent candidates changed')
        for source in record['sources']:
            if digest(source['path'])!=source['sha256']:raise ValueError('Recent search bytes changed')
        return record
    sources=[]
    with patch.object(search,'RAW',RAW/'discovery'/label):
        page,proof=search.get_page(query,0);sources.append(proof)
        total=search.validate_page(page,query,0);hits=list(page['hits']['hits'])
        for offset in range(100,total,100):
            page,proof=search.get_page(query,offset);sources.append(proof)
            if search.validate_page(page,query,offset)!=total:raise ValueError('Search count changed')
            hits.extend(page['hits']['hits'])
    if len(hits)!=total or len({h['_id'] for h in hits})!=total:raise ValueError('Incomplete discovery')
    frame=pd.DataFrame(search.candidates_from_hits(hits,query)).drop_duplicates(['cik','accession_number'])
    frame.to_csv(folder/'candidates.csv',index=False)
    result={'query':query,'implementation_sha256':digest(__file__),'sources':sources,
            'candidates_sha256':digest(folder/'candidates.csv'),'candidate_companies':int(frame.cik.nunique()),'search_hits':total}
    write_json(saved,result);return result


def prepare():
    config=json.loads(CONFIG.read_text());train_cutoff,cal_cutoff=chronology(config)
    ROOT.mkdir(parents=True,exist_ok=True)
    discoveries={label:inventory(label,start,end) for label,start,end in [
        ('training_extension','2022-07-20',train_cutoff),
        ('new_calibration','2023-06-01',cal_cutoff)]}
    blocks=[];sources=[]
    for quarter in range(1,5):
        path=RAW/'bulk'/f'2023q{quarter}.zip'
        source=coordinated_fetch(f'https://www.sec.gov/files/dera/data/financial-statement-data-sets/2023q{quarter}.zip',path)
        sources.append(source)
        with zipfile.ZipFile(path) as archive:
            frame=pd.read_csv(archive.open('sub.txt'),sep='\t',dtype={'cik':str,'filed':str},low_memory=False)
        frame=frame.loc[frame.form.eq('10-K'),['cik','name','filed','sic']].copy()
        frame['cik']=frame.cik.str.zfill(10);frame['filed']=pd.to_datetime(frame.filed,format='%Y%m%d').dt.strftime('%Y-%m-%d')
        blocks.append(frame.loc[frame.filed.lt(config['calibration_anchor'])])
    annual=pd.concat(blocks,ignore_index=True).drop_duplicates()
    sic=pd.to_numeric(annual.sic,errors='coerce');annual=annual.loc[sic.notna() & ~sic.between(6000,6999)]
    annual.to_csv(ROOT/'annual_2023.csv',index=False)
    result={'config_sha256':digest(CONFIG),'implementation_sha256':digest(__file__),'sources':sources,
            'discoveries':discoveries,'annual_frame_sha256':digest(ROOT/'annual_2023.csv'),
            'eligible_2023_companies':int(annual.cik.nunique()),'training_cutoff':train_cutoff,'calibration_cutoff':cal_cutoff,
            'reserved_final_rows_read':False,'production_approval':False}
    saved=ROOT/'report.json'
    if saved.exists() and json.loads(saved.read_text())!=result:raise ValueError('Frozen newer source frame changed')
    if not saved.exists():write_json(saved,result)
    return result


if __name__=='__main__':print(json.dumps(prepare(),indent=2))
