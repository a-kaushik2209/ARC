import arc
from importlib.metadata import version

def test_version():
    metadata_version = version("arc-training")
    assert arc.__version__ == metadata_version