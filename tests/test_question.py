import pytest
import pandas as pd
import numpy as np
import requests
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tasks.task_manager import *

@pytest.fixture
def dummy_df():
    return pd.DataFrame({
        "Name": ["Light", "L", "Misa"],
        "Justice Score": [80, None, 60],
        "IQ": [140, 180, 100],
        "Aggression": [0.8, 0.2, 0.6],
        "City": ["Tokyo", "Kyoto", "Tokyo"],
        "Is Suspect": [1, 0, 1]
    })

def test_clean_column_names(dummy_df):
    df = clean_column_names(dummy_df)
    assert "justice_score" in df.columns

def test_fill_missing_scores(dummy_df):
    df = clean_column_names(dummy_df)
    df = fill_missing_scores(df, "justice_score")
    assert df["justice_score"].isnull().sum() == 0

def test_encode_city_column(dummy_df):
    df = clean_column_names(dummy_df)
    df = encode_city_column(df)
    assert "city_tokyo" in df.columns

def test_detect_outliers_iq(dummy_df):
    df = clean_column_names(dummy_df)
    df = detect_outliers_iq(df)
    assert "is_outlier" in df.columns

def test_generate_suspect_score(dummy_df):
    df = clean_column_names(dummy_df)
    df["aggression_scaled"] = df["aggression"]  # dummy normalization
    df = generate_suspect_score(df)
    assert "suspect_score" in df.columns

def send_post_request(url: str, data: dict, headers: dict = None):
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()  # hata varsa exception fırlatır
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - Status Code: {response.status_code}")
    except Exception as err:
        print(f"Other error occurred: {err}")

class ResultCollector:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1

def run_tests():
    collector = ResultCollector()
    pytest.main(["tests"], plugins=[collector])
    print(f"\nToplam Başarılı: {collector.passed}")
    print(f"Toplam Başarısız: {collector.failed}")
    
    user_score = (collector.passed / (collector.passed + collector.failed)) * 100
    print(round(user_score, 2))
    
    url = "https://kaizu-api-8cd10af40cb3.herokuapp.com/projectLog"
    payload = {
        "user_id": 403,
        "project_id": 699,
        "user_score": round(user_score, 2),
        "is_auto": False
    }
    headers = {
        "Content-Type": "application/json"
    }
    send_post_request(url, payload, headers)

if __name__ == "__main__":
    run_tests()
