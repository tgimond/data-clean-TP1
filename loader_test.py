import pytest
import pandas as pd

# This script contains tests for the cleaning functions and should be
# completed by you


@pytest.fixture
def sample_dirty_fname() -> str:
    fname = 'data-clean-TP1/sample_dirty.csv'
    return fname


def sample_formatted() -> pd.DataFrame:
    # TODO Complete the test case bellow
    df = pd.DataFrame(
        data='sample_dirty.csv',
        dtype=...
    )

    return df


@pytest.fixture
def sample_sanitized() -> pd.DataFrame:
    # TODO Complete the test case bellow
    df = pd.DataFrame(
        data=...,
        dtype=...
    )
    return df


@pytest.fixture
def sample_framed() -> pd.DataFrame:
    # TODO Complete the test case bellow
    df = pd.DataFrame(
        data=...,
        dtype=...
    )
    return df


def test_load_formatted_data(sample_dirty_fname, sample_formatted):
    from loader import load_formatted_data
    assert load_formatted_data(sample_dirty_fname).equals(sample_formatted)


def test_sanitize_data(sample_formatted, sample_sanitized):
    from loader import sanitize_data
    assert sanitize_data(sample_formatted).equals(sample_sanitized)


def test_frame_data(sample_sanitized, sample_framed):
    from loader import frame_data
    assert frame_data(sample_sanitized).equals(sample_framed)


def test_load_clean_data(sample_dirty_fname, sample_framed):
    from loader import load_clean_data
    assert load_clean_data(sample_dirty_fname).equals(sample_framed)


def assert_column_equal(clean, target, column):
    # utility function if you which to implement column-specific assertion tests
    assert clean[column].equals(
        target[column]), f"Result should be {clean[column]} but was {target[column]}"
