from spendlens.config import settings
def test_channel_order_is_fixed():
    assert settings()['channels'] == [
        'rep_visits','speaker_programs','hcp_email','programmatic_display',
        'paid_search','paid_social','medical_journals','patient_digital'
    ]
def test_fabric_is_not_a_dependency():
    text = open('requirements.txt', encoding='utf-8').read().lower()
    assert 'fabric' not in text
def test_sql_server_is_not_a_dependency():
    text = open('requirements.txt', encoding='utf-8').read().lower()
    assert 'pyodbc' not in text and 'pymssql' not in text
