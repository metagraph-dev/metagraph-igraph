import pytest
from metagraph.core.resolver import Resolver


@pytest.fixture(scope="session")
def default_plugin_resolver():
    from metagraph.plugins import find_plugins
    from metagraph_igraph import registry

    res = Resolver()
    res.register(**find_plugins())  # from metagraph
    res.register(**registry.find_plugins())  # from metagraph-igraph
    return res
