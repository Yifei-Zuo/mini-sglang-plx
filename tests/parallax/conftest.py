import pytest


@pytest.fixture(scope="session", autouse=True)
def tp_info():
    from minisgl.distributed import set_tp_info

    try:
        set_tp_info(rank=0, size=1)
    except AssertionError:
        pass  # already set by another test module in this session
