"""Run POPKIN from a user workspace."""
import sys
from pathlib import Path
from popkin import paths
from popkin.config.logger import info, error, warning, timer
from popkin.config.logger import LoggerConfig
from popkin.config.user_config import import_optional_config


PROGRAM_RUNNERS = {
    "sse": "run_sse",
    "bse": "run_bse",
    "popsin": "run_popsin",
    "popbin": "run_popbin",
}


def apply_custom_data_dir(config_name, attr_name):
    """Apply a program-specific custom data directory if configured."""
    config = import_optional_config(config_name)
    if config is None:
        warning(f"{config_name}.py not found; using default data directory")
        return

    if hasattr(config, attr_name):
        paths.data = Path(getattr(config, attr_name))
        info(f"Using custom data directory: {paths.data}")


@timer("POPKIN main program", console=True)
def main():
    """Main program entry point."""
    inlist = import_optional_config("inlist")
    if inlist is None:
        error("User config file not found: inlist.py", console=True)
        raise ModuleNotFoundError("User config file not found: inlist.py")

    info("Successfully loaded user config: inlist.py")

    if not hasattr(inlist, "program"):
        error("Missing required variable in inlist.py: program", console=True)
        raise ValueError("Missing required variable in inlist.py: program")

    program = inlist.program
    valid_programs = list(PROGRAM_RUNNERS)

    if program not in valid_programs:
        error(
            f"Invalid program value: {program}",
            context={"valid": valid_programs},
            console=True
        )
        raise ValueError(f"Unknown program: {program}. Expected one of: {', '.join(valid_programs)}")

    info(f"Selected program: {program}", console=True)

    program_data_dirs = {
        "sse": paths.workspaces / "data" / "sse",
        "bse": paths.workspaces / "data" / "bse",
        "popsin": paths.workspaces / "data" / "popsin",
        "popbin": paths.workspaces / "data" / "popbin",
    }
    paths.data = program_data_dirs[program]

    if program == "popsin":
        apply_custom_data_dir("inlist_popsin", "popsin_data_dir")

    if program == "popbin":
        apply_custom_data_dir("inlist_popbin", "popbin_data_dir")

    paths.data.mkdir(parents=True, exist_ok=True)

    try:
        import popkin.drivers as drivers

        runner = getattr(drivers, PROGRAM_RUNNERS[program])
        runner()

        success_msg = f"{program} completed successfully"
        info(success_msg, console=True)
    except Exception as e:
        error(
            f"Program failed: {e}",
            context={"program": program, "data_dir": str(paths.data)},
            console=True
        )
        raise


if __name__ == "__main__":
    _work_dir = Path(__file__).resolve().parent
    if str(_work_dir) not in sys.path:
        sys.path.insert(0, str(_work_dir))

    paths.workspaces = _work_dir
    paths.logs = paths.workspaces / "logs"

    LoggerConfig.setup(
        logs_dir=paths.logs,
        log_filename="popkin.log",
        level="INFO",
        also_console=True,
    )
    main()
