import concurrent.futures
import itertools

import pytest


@pytest.fixture(scope="session", autouse=True)
def faker_seed():
    """
    Seeds the faker library with a constant value.

    See: https://faker.readthedocs.io/en/master/pytest-fixtures.html
    """
    return 12345


@pytest.fixture
def futures_hash_in_descending_order(monkeypatch):
    """
    Gives each `concurrent.futures.Future` a small hash, descending in the
    order the futures are first hashed.

    A set of futures iterates by hash. A future's default hash comes from its
    memory address, so a set of them can iterate in creation order by chance.
    With these hashes it never does: a test of result order then fails for
    any code that takes its order from a set of futures.
    """
    next_hash = itertools.count(1000, -1)

    def descending_hash(future):
        if not hasattr(future, "_order_test_hash"):
            future._order_test_hash = next(next_hash)

        return future._order_test_hash

    monkeypatch.setattr(concurrent.futures.Future, "__hash__", descending_hash)
