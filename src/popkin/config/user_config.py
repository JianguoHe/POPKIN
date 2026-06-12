"""User configuration loading helpers."""

from importlib import import_module


def import_optional_config(module_name):
    try:
        return import_module(module_name)
    except ModuleNotFoundError as e:
        if e.name == module_name:
            return None
        raise


def apply_user_config(namespace, *module_names):
    loaded = []

    for module_name in module_names:
        module = import_optional_config(module_name)
        if module is None:
            continue

        for name, value in vars(module).items():
            if not name.startswith("_"):
                namespace[name] = value

        loaded.append(module_name)

    return loaded


def log_loaded_configs(logger, loaded_configs):
    if loaded_configs:
        logger.info(
            f"Successfully loaded user config: "
            f"{', '.join(name + '.py' for name in loaded_configs)}",
            extra={"console": True},
        )
    else:
        logger.warning(
            "No user config found, using default parameters",
            extra={"console": True},
        )