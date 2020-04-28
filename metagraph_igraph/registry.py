from metagraph import PluginRegistry

# Use this as the entry_point object
registry = PluginRegistry()


def find_plugins():
    # Ensure we import all items we want registered
    from . import types, translators, algorithms

    registry.register_from_modules(types, translators, algorithms)
    return registry.plugins


################
# Import guards
################
