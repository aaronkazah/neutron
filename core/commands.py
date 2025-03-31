import os
import sys
import asyncio
import time
import unittest
import logging
import uvicorn
from typing import List, Callable, Dict, Union, Coroutine, Any
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Type definitions
CommandFunction = Union[
    Callable[[List[str]], None], Callable[[List[str]], Coroutine[Any, Any, None]]
]
CommandRegistry = Dict[str, CommandFunction]

# Global settings
COMMANDS: CommandRegistry = {}
BASE_DIR = Path(__file__).parent
RESTART_FILE = BASE_DIR / ".restart"


def register_command(name: str) -> Callable[[CommandFunction], CommandFunction]:
    """Decorator to register a command function."""

    def decorator(func: CommandFunction) -> CommandFunction:
        COMMANDS[name] = func
        return func

    return decorator


class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, restart_callback: Callable[[], None]):
        self.restart_callback = restart_callback
        self.last_restart = 0
        self._restart_delay = 1

    def on_any_event(self, event):
        if event.is_directory or not event.src_path.endswith(".py"):
            return

        current_time = time.time()
        if current_time - self.last_restart < self._restart_delay:
            return

        self.last_restart = current_time
        print(f"\nDetected change in {event.src_path}")
        request_restart()
        self.restart_callback()


def needs_restart() -> bool:
    """Check if server needs to restart."""
    return RESTART_FILE.exists()


def request_restart():
    """Request a server restart."""
    RESTART_FILE.parent.mkdir(exist_ok=True)
    RESTART_FILE.touch()


def clear_restart():
    """Clear the restart flag."""
    if RESTART_FILE.exists():
        RESTART_FILE.unlink()


def restart_server():
    """Restart the server process."""
    python = sys.executable
    os.execl(python, python, *sys.argv)


async def watch_for_changes(
    paths: List[str], restart_callback: Callable[[], None]
) -> None:
    observer = Observer()
    handler = FileChangeHandler(restart_callback)

    for path in paths:
        if os.path.exists(path):
            observer.schedule(handler, path, recursive=True)
            print(f"Watching {path} for changes...")

    observer.start()
    try:
        while True:
            if needs_restart():
                clear_restart()
                restart_callback()
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        observer.stop()
    observer.join()


@register_command("makemigrations")
async def makemigrations_command(args: List[str]) -> None:
    """Create new migrations based on changes detected to your models."""
    try:
        # Import MigrationManager here to allow the command to work without it
        from core.db.migrations import MigrationManager

        manager = MigrationManager(base_dir="apps")
        app_labels = [args[0]] if args else manager.apps

        for app_label in app_labels:
            models = manager._discover_models(app_label)
            if not models:
                continue

            operations = await manager.makemigrations(
                app_label=app_label, models=models, return_ops=False, clean=False
            )


    except ImportError:
        print(
            "Error: MigrationManager not found. Please ensure core.db.migrations is available."
        )
        sys.exit(1)
    except Exception as e:
        print(f"Error creating migrations: {str(e)}")
        raise


@register_command("migrate")
async def migrate_command(args: List[str]) -> None:
    """Apply database migrations."""
    try:
        # Import required components here to allow the command to work without them
        from core.db.migrations import MigrationManager
        from core.db.base import DATABASES

        manager = MigrationManager(base_dir="apps")
        app_label = args[0] if args and not args[0].startswith("--") else None

        try:
            if app_label:
                connection = await DATABASES.get_connection(app_label)
                await manager.migrate(
                    app_label=app_label, connection=connection, operations=None
                )
            else:
                await manager.bootstrap_all(test_mode=False)

        except Exception as e:
            print(f"Error during migration: {str(e)}")
            raise
        finally:
            await DATABASES.shutdown()

    except ImportError:
        print(
            "Error: Required migration components not found. Please ensure core.db is available."
        )
        sys.exit(1)


@register_command("start")
async def start_command(args: List[str]) -> None:
    """Start the development server with auto-reload."""
    try:
        from core.db.migrations import MigrationManager
        from core.db.base import DATABASES

        os.environ.setdefault("ENV", ".env")

        clear_restart()

        watch_paths = ["apps", "core", ".env"]
        watch_task = asyncio.create_task(watch_for_changes(watch_paths, restart_server))

        # Initialize database
        manager = MigrationManager(base_dir="apps")
        await manager.bootstrap_all(test_mode=False)
        PORT = os.getenv("PORT", 7777)

        # Configure server
        config = uvicorn.Config(
            app="apps.entry:app",
            host="0.0.0.0",
            port=PORT,
            log_level="debug",
            reload=False,
        )
        server = uvicorn.Server(config=config)
        server_task = asyncio.create_task(server.serve())

        await asyncio.gather(watch_task, server_task)
    except ImportError:
        print("Error: Required components not found. Running without database support.")
        # Run without database support
        await watch_for_changes([".", ".env"], restart_server)
    except asyncio.CancelledError:
        print("\nShutting down...")
    finally:
        watch_task.cancel()
        try:
            await watch_task
        except asyncio.CancelledError:
            pass
        try:
            await DATABASES.shutdown()
        except NameError:
            pass


@register_command("test")
def test_command(args: List[str]) -> None:
    """Run the test suite."""
    verbosity = 1
    test_args = []

    # Check for -v or --verbose flags
    for i, arg in enumerate(args):
        if arg == "-v" or arg.startswith("--verbose"):
            if arg == "-v" and i + 1 < len(args) and args[i + 1].isdigit():
                verbosity = int(args[i + 1])
                # Skip the next argument since it's the verbosity level
                continue
            elif arg.startswith("-v") and len(arg) > 2 and arg[2:].isdigit():
                verbosity = int(arg[2:])
            else:
                verbosity = 3  # Max verbosity for --verbose
        else:
            test_args.append(arg)

    # Set logging level based on verbosity
    if verbosity >= 3:
        logging.basicConfig(level=logging.DEBUG)
        print(f"Running tests with DEBUG logging level (verbosity {verbosity})")
    elif verbosity == 2:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.ERROR)

    # Always keep subprocess logging at WARNING level to avoid noise
    logging.getLogger("asyncio.subprocess").setLevel(logging.WARNING)

    try:
        os.environ.setdefault("ENV", ".env")
        from apps.entry import app
        from core.db.base import DATABASES

        DATABASES.configure_app("test", memory=True)
    except ImportError:
        print("Warning: Running tests without database support")

    loader = unittest.TestLoader()
    runner = unittest.TextTestRunner(verbosity=verbosity)

    try:
        if test_args:
            if test_args[0].startswith("apps.") or test_args[0].startswith("core."):
                suite = loader.loadTestsFromName(test_args[0])
            else:
                pattern = test_args[0] if test_args[0].endswith(".py") else "test*.py"
                start_dir = test_args[0] if os.path.isdir(test_args[0]) else "."
                suite = loader.discover(start_dir=start_dir, pattern=pattern)
        else:
            suite = unittest.TestSuite()
            for directory in ["apps", "core"]:
                if os.path.exists(directory):
                    discovered = loader.discover(
                        start_dir=directory, pattern="test*.py", top_level_dir="."
                    )
                    suite.addTests(discovered)

        result = runner.run(suite)
        if not result.wasSuccessful():
            sys.exit(1)
    except Exception as e:
        print(f"Error running tests: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        try:
            asyncio.run(DATABASES.shutdown())
        except NameError:
            pass