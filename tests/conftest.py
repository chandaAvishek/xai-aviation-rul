# standard
import matplotlib
import pytest

# uses matplotlib to non-interactive backend before importing pyplot
matplotlib.use('Agg')

# 3rd party
import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def close_plots():
    """Automatically close all matplotlib figures after each test."""
    yield
    plt.close('all')
