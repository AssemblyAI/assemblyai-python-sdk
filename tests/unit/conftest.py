import pytest


@pytest.fixture(scope="session", autouse=True)
def faker_seed():
    """
    Seeds the faker library with a constant value.

    See: https://faker.readthedocs.io/en/master/pytest-fixtures.html
    """
    return 12345
