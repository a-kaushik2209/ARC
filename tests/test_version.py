import arc
from importlib.metadata import version

def test_version():
    metadata_version = version("arc")
    assert arc.__version__ == metadata_version