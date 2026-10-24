from io import BytesIO
import pytest
from src.web.backups import replicate
from src.web.operations import notify
from scripts.load_check import run


def test_off_server_backup_verifies_encrypted_bytes(tmp_path):
    path=tmp_path/'backup.fernet';path.write_bytes(b'encrypted-example')
    class Store:
        def put_object(self,**kwargs):self.obj=kwargs
        def get_object(self,**kwargs):return {'Body':BytesIO(self.obj['Body'])}
    store=Store()
    assert replicate(path,'ledger-test-bucket',client=store)=='verified'
    assert store.obj['ServerSideEncryption']=='AES256'
    assert store.obj['Key']=='ledger/backup.fernet'
    store.get_object=lambda **kwargs:{'Body':BytesIO(b'corrupt')}
    with pytest.raises(ValueError):replicate(path,'ledger-test-bucket',client=store)


def test_no_destinations_mean_no_external_calls(tmp_path,monkeypatch):
    monkeypatch.delenv('LEDGER_ALERT_TOPIC_ARN',raising=False)
    assert notify('ledger-health',tmp_path)['status']=='not_configured'
    assert replicate(tmp_path/'does-not-exist','')=='not_configured'


def test_load_check_is_bounded_before_any_credentials_or_network():
    with pytest.raises(ValueError):run('https://example.com')
    with pytest.raises(ValueError):run(count=49)
    with pytest.raises(ValueError):run(concurrency=3)
