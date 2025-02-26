#!/usr/bin/env python
import os
import sys
import asyncio

from core.commands import COMMANDS


def main() -> None:
    """Main entry point for the command line interface."""
    os.environ.setdefault("DATABASE_PATH", ".data/databases")
    try:
        if len(sys.argv) < 2:
            print("Available commands:", ", ".join(COMMANDS.keys()))
            return

        command = sys.argv[1]
        args = sys.argv[2:]

        if command not in COMMANDS:
            print(f"Unknown command: {command}")
            print("Available commands:", ", ".join(COMMANDS.keys()))
            sys.exit(1)

        task = COMMANDS[command](args)
        if asyncio.iscoroutine(task):
            asyncio.run(task)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        if os.getenv("DEBUG", "False").lower() == "true":
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
