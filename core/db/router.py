# core/db/router.py (new file)

from abc import ABC, abstractmethod


class DatabaseRouter(ABC):
    """
    Abstract base class for database routing decisions.

    Router classes define how database operations are routed to different database connections.
    """

    @abstractmethod
    def db_for_app(self, app_label: str) -> str:
        """
        Return the database alias to use for the given app label.

        Args:
            app_label: The application label requesting a database.

        Returns:
            The database alias that should be used.
        """
        pass

    def allow_migrate(self, db: str, app_label: str, model_name: str = None) -> bool:
        """
        Determine if migration operations should be run on the specified database.

        Args:
            db: Database alias being migrated.
            app_label: Application label of the model being migrated.
            model_name: Name of the model being migrated (optional).

        Returns:
            True if migration should proceed, False otherwise.
        """
        # Default implementation: allow all migrations
        return True


# Continue in core/db/router.py

class PerAppRouter(DatabaseRouter):
    """Routes each application to its own database alias."""

    def db_for_app(self, app_label: str) -> str:
        if not app_label:
            raise ValueError("app_label cannot be empty for PerAppRouter")
        return app_label


class SingleDatabaseRouter(DatabaseRouter):
    """Routes all applications to a single, specified database alias."""

    def __init__(self, default_alias: str = "default"):
        """
        Initialize router with the default database alias.

        Args:
            default_alias: The database alias to route all operations to.
        """
        if not default_alias:
            raise ValueError("default_alias cannot be empty for SingleDatabaseRouter")
        self.default_alias = default_alias

    def db_for_app(self, app_label: str) -> str:
        """Always return the default database alias regardless of app."""
        return self.default_alias