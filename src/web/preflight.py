"""Check a Ledger backend installation without downloading data or scoring companies."""
import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from src.web.company_imports import CompanyImports
from src.web.research_service import ResearchService, REPO


def run(root='data/processed/company_imports'):
    checks={};errors=[]
    root=Path(root)
    try:
        root.mkdir(parents=True,exist_ok=True)
        with NamedTemporaryFile(dir=root):pass
        checks['persistent_directory_writable']=True
    except OSError:
        checks['persistent_directory_writable']=False;errors.append('Import directory is not writable.')
    try:
        from src.data_layer.edgar_ingest import sec_headers
        sec_headers()
        checks['sec_contact_configured']=True
    except ValueError:
        checks['sec_contact_configured']=False;errors.append('Configure EDGAR_USER_AGENT with a real application/contact on the server.')
    service=CompanyImports(root=root,frontend=REPO/'frontend')
    try:
        catalog=service.catalog()
        checks['catalog_companies']=len(catalog['companies'])
        service.base_retriever()
        checks['base_retrieval_index_ready']=True
    except (OSError,ValueError,KeyError):
        checks['base_retrieval_index_ready']=False;errors.append('Public frontend data could not be indexed.')
    model=ResearchService(service).model_status()
    service.executor.shutdown()
    return dict(backend_ready=not errors,checks=checks,errors=errors,model=model,
                production_model_ready=model.get('production_approval',False),
                inference_executed=False,aws_required_for_local_run=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--import-root',default='data/processed/company_imports')
    result=run(parser.parse_args().import_root)
    print(json.dumps(result,indent=2))
    if not result['backend_ready']:raise SystemExit(1)
