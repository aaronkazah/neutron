# core/db/base.py
import asyncio
import logging
import os
from enum import Enum
from typing import Dict, Optional, List, Any
import aiosqlite

from core.db.router import DatabaseRouter, PerAppRouter, SingleDatabaseRouter

BASE_PRAGMAS = {
    "journal_mode": "WAL",
    "synchronous": "NORMAL",
    "mmap_size": 17179869184,
    "cache_size": 5242880,
    "temp_store": "MEMORY",
    "journal_size_limit": 104857600,
}

class DatabaseType(Enum):
    SQLITE = "sqlite"
    POSTGRES = "postgres"

class DatabaseInfo:
    """Stores database connection and related information."""
    # ... (keep existing DatabaseInfo class content) ...
    def __init__(
        self,
        name: str, # This 'name' is the alias the connection is stored under
        connection: aiosqlite.Connection,
        db_type: DatabaseType,
        config: Dict,
    ):
        self.name = name
        self.connection = connection
        self.db_type = db_type
        self.config = config

    def __repr__(self):
        return f"DatabaseInfo(name={self.name}, db_type={self.db_type}, config={self.config})"

    def __str__(self):
        return f"DatabaseInfo(name={self.name}, db_type={self.db_type}, config={self.config})"

    async def executemany(self, query: str, values: List[Any]) -> None:
        """Execute many queries."""
        await self.connection.executemany(query, values)
        await self.connection.commit()

    async def execute(self, query: str, values: Optional[List] = None) -> Any:
        """Execute a query with proper parameter binding"""
        return await self.connection.execute(query, values or [])

    async def fetch_one(
        self, query: str, values: Optional[List] = None
    ) -> Optional[Any]:
        """Fetch one row with proper parameter binding"""
        cursor = await self.execute(query, values)
        try:
            return await cursor.fetchone()
        finally:
            await cursor.close()

    async def fetch_all(self, query: str, values: Optional[List] = None) -> List[Any]:
        """Fetch all rows with proper parameter binding"""
        cursor = await self.execute(query, values)
        try:
            return await cursor.fetchall()
        finally:
            await cursor.close()

    async def commit(self):
        """Commit the current transaction."""
        await self.connection.commit()

    async def rollback(self):
        """Rollback the current transaction."""
        await self.connection.rollback()

    async def close(self):
        """Close the connection."""
        await self.connection.close()


class BaseDatabase:
    # Update in core/db/base.py - the BaseDatabase class

    def __init__(self, database_config: Dict = None, router: Optional[DatabaseRouter] = None):
        self.CONNECTIONS: Dict[str, DatabaseInfo] = {}
        self.CONFIG = {}
        # Ensure DATABASE_PATH is set, default if not
        self.base_path = os.environ.get("DATABASE_PATH", os.path.join(os.getcwd(), "db_files"))
        os.makedirs(self.base_path, exist_ok=True)

        # Set the router, default to PerAppRouter
        from core.db.router import PerAppRouter
        self.router: DatabaseRouter = router or PerAppRouter()

        if database_config:
            self.apply_config(database_config)

    def set_router(self, router: 'DatabaseRouter'):
        """Sets the database routing strategy."""
        from core.db.router import DatabaseRouter
        if not isinstance(router, DatabaseRouter):
            raise TypeError("Router must be an instance of DatabaseRouter")
        self.router = router

    def get_default_config(self, db_alias: str, test_mode: bool = False) -> Dict:
        """Generate default configuration for a database alias."""
        # Check if this is a postgres:// or postgresql:// URI
        is_postgres_uri = (
                isinstance(db_alias, str) and
                (db_alias.startswith("postgres://") or db_alias.startswith("postgresql://"))
        )

        if is_postgres_uri:
            # Parse the postgres URI into components
            from urllib.parse import urlparse
            parsed = urlparse(db_alias)

            # Extract parts
            username = parsed.username or "postgres"
            password = parsed.password or ""
            hostname = parsed.hostname or "localhost"
            port = parsed.port or 5432
            database = parsed.path.lstrip('/') or db_alias.split('/')[-1] or "postgres"

            return {
                "ENGINE": "asyncpg",
                "HOST": hostname,
                "PORT": port,
                "USER": username,
                "PASSWORD": password,
                "NAME": database,
                "SETTINGS": {},  # PostgreSQL settings
                "MIN_CONNECTIONS": 1,
                "MAX_CONNECTIONS": 10,
            }

        # For SQLite, use existing logic
        if test_mode or db_alias == ":memory:":
            return {
                "ENGINE": "aiosqlite",
                "NAME": ":memory:",
                "PRAGMAS": BASE_PRAGMAS.copy(),
            }

        # Default SQLite database path
        db_path = os.path.join(self.base_path, f"{db_alias}.sqlite3")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        return {"ENGINE": "aiosqlite", "NAME": db_path, "PRAGMAS": BASE_PRAGMAS.copy()}

    async def get_connection(
            self, app_label_or_alias: str, retry: bool = True, timeout: int = 30, test_mode: bool = False
    ) -> DatabaseInfo:
        """
        Initialize or retrieve a database connection pool based on the configured router.
        Returns the DatabaseInfo object wrapping the pool/connection.
        """
        if not app_label_or_alias:
            raise ValueError("app_label_or_alias cannot be empty")

        # Use router to find the target alias
        target_alias = self.router.db_for_app(app_label_or_alias)

        # Return existing connection info if available for the target alias
        if target_alias in self.CONNECTIONS:
            # Ensure the connection info for the original alias also points to the same object
            if app_label_or_alias != target_alias:
                self.CONNECTIONS[app_label_or_alias] = self.CONNECTIONS[target_alias]
            return self.CONNECTIONS[target_alias]

        # Create a new connection/pool if needed
        if target_alias not in self.CONFIG:
            self.CONFIG[target_alias] = self.get_default_config(target_alias, test_mode=test_mode)

        config = self.CONFIG[target_alias]
        db_info = None # Initialize

        try:
            engine = config.get("ENGINE", "").lower()

            if engine == "aiosqlite":
                if retry: await self.wait_for_database(config, timeout)
                connection = await self._init_sqlite_connection(config)
                db_info = DatabaseInfo( # Use the base DatabaseInfo for SQLite
                    name=target_alias,
                    connection=connection, # SQLite holds the direct connection
                    db_type=DatabaseType.SQLITE,
                    config=config,
                )
            elif engine == "asyncpg":
                from core.db.postgres import PostgreSQLDatabaseInfo, PostgresConnectionManager
                # Call create_connection, which now returns only the pool
                pool = await PostgresConnectionManager.create_connection(config)
                # Instantiate PostgreSQLDatabaseInfo with the pool
                db_info = PostgreSQLDatabaseInfo(
                    name=target_alias,
                    pool=pool, # Pass the pool
                    config=config,
                    # No initial connection passed here anymore
                )
            else:
                raise ValueError(f"Unsupported database engine: {engine}")

            # Store the connection info under the target alias
            self.CONNECTIONS[target_alias] = db_info
            # Also store under the original alias if different, pointing to the same object
            if app_label_or_alias != target_alias:
                self.CONNECTIONS[app_label_or_alias] = db_info

            return db_info

        except Exception as e:
            # Log with more detail if possible
            logging.getLogger("BaseDatabase").error(f"Error initializing database for alias '{target_alias}': {e}", exc_info=True)
            raise # Re-raise the original exception

    async def _init_sqlite_connection(self, config: Dict) -> aiosqlite.Connection:
        """Initialize SQLite connection."""
        db_path = config["NAME"]

        if db_path != ":memory:":
            # Ensure database directory exists
            abs_path = os.path.abspath(db_path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        connection = await aiosqlite.connect(db_path)
        connection.row_factory = aiosqlite.Row

        await self.apply_sqlite_pragmas(connection, config.get("PRAGMAS", {}))
        return connection

    def apply_config(self, config: Dict):
        """Apply database configuration."""
        for alias, cfg in config.items():
            if alias not in self.CONFIG: # Only add if not already present
                self.CONFIG[alias] = cfg
            # else: Optionally update existing config? Or log a warning?

    async def wait_for_database(self, config: Dict, timeout: int = 30) -> None:
        """Wait for database file to be accessible (for non-memory DBs)."""
        if config["NAME"] == ":memory:":
             return # No waiting needed for in-memory

        start_time = asyncio.get_event_loop().time()
        db_path = config["NAME"]

        while True:
            try:
                # Check if the directory exists and we can connect
                if os.path.exists(os.path.dirname(os.path.abspath(db_path))):
                     conn = await aiosqlite.connect(db_path)
                     await conn.close()
                     return # Success
                else:
                     # Directory doesn't exist yet, wait
                     pass
            except aiosqlite.OperationalError as e:
                 print(f"Database not ready ({db_path}): {e}. Retrying...")
                 pass # Continue waiting
            except Exception as e: # Catch other potential errors during connect attempt
                 print(f"Unexpected error waiting for database ({db_path}): {e}. Retrying...")

            if asyncio.get_event_loop().time() - start_time > timeout:
                raise TimeoutError(
                    f"Database connection timeout after {timeout} seconds for {db_path}"
                )
            await asyncio.sleep(1)

    async def apply_sqlite_pragmas(
        self, connection: aiosqlite.Connection, pragmas: Dict
    ):
        """Apply PRAGMA settings for SQLite."""
        if not pragmas:
            return
        for pragma, value in pragmas.items():
            try:
                # Use parameter binding for PRAGMA values where possible (safer)
                # Note: Not all pragmas support parameter binding for the value
                if isinstance(value, (int, str, bool)):
                    await connection.execute(f"PRAGMA {pragma}={value};") # Direct execution needed for some
                else: # Fallback for complex or unknown types
                     await connection.execute(f"PRAGMA {pragma}={value};")
            except aiosqlite.OperationalError as e:
                print(f"Warning: Error applying PRAGMA {pragma}={value}: {e}")
        await connection.commit() # Commit pragmas

    async def close_connection(self, alias: str):
        """Close a specific database connection."""
        if alias in self.CONNECTIONS:
            try:
                await self.CONNECTIONS[alias].close()
            except Exception as e:
                 print(f"Error closing connection for alias '{alias}': {e}")
            finally:
                del self.CONNECTIONS[alias]
        # else: No connection existed for this alias

    async def shutdown(self, aliases: Optional[List[str]] = None):
        """Shutdown specified or all connections."""
        if aliases is None:
            aliases_to_close = list(self.CONNECTIONS.keys())
        else:
            aliases_to_close = aliases

        for alias in aliases_to_close:
            await self.close_connection(alias)

    def configure_app(
            self,
            app_or_alias: str,  # Can configure an app or a specific alias
            path: Optional[str] = None,
            pragmas: Optional[Dict] = None,
            memory: bool = False,
    ):
        """
        Configure a specific database alias or the alias associated with an app
        via the current router.
        """
        # Determine the target alias
        # If using SingleDatabaseRouter, configuring an 'app' configures the default alias
        target_alias = self.router.db_for_app(app_or_alias)

        if memory:
            config = {
                "ENGINE": "aiosqlite",
                "NAME": ":memory:",
                "PRAGMAS": BASE_PRAGMAS.copy(),
            }
        else:
            # Get default based on the target alias name
            config = self.get_default_config(target_alias)

        if path:
            config["NAME"] = path
            # Ensure directory exists if a path is provided
            if config["NAME"] != ":memory:":
                os.makedirs(os.path.dirname(os.path.abspath(config["NAME"])), exist_ok=True)

        if pragmas:
            # Ensure PRAGMAS dict exists before updating
            if "PRAGMAS" not in config:
                config["PRAGMAS"] = BASE_PRAGMAS.copy()
            config["PRAGMAS"].update(pragmas)

        self.CONFIG[target_alias] = config
        return config


# Global instance



DATABASES = BaseDatabase(router=PerAppRouter())

def use_single_database(alias: str = "default"):
    """Switch to SingleDatabaseRouter globally."""
    from core.db.router import SingleDatabaseRouter
    router = SingleDatabaseRouter(default_alias=alias)
    DATABASES.set_router(router)
    # Configure the database if not already configured
    if alias not in DATABASES.CONFIG:
        DATABASES.configure_app(alias)
    return router

def use_per_app_database():
    """Switch back to PerAppRouter globally (the default)."""
    from core.db.router import PerAppRouter
    router = PerAppRouter()
    DATABASES.set_router(router)
    return router


# Add to the module level functions
def use_postgres_database(alias: str = "default", connection_string: str = None):
    """Set up PostgreSQL database with a single database router."""
    from core.db.postgres import PostgresRouter

    # Create router
    router = PostgresRouter(default_alias=alias)
    DATABASES.set_router(router)

    # Configure the database if not already configured
    if alias not in DATABASES.CONFIG:
        if connection_string:
            DATABASES.CONFIG[alias] = DATABASES.get_default_config(connection_string)
        else:
            # Default localhost configuration
            DATABASES.CONFIG[alias] = {
                "ENGINE": "asyncpg",
                "HOST": "localhost",
                "PORT": 5432,
                "USER": "postgres",
                "PASSWORD": "",
                "NAME": "postgres",
                "SETTINGS": {},
                "MIN_CONNECTIONS": 1,
                "MAX_CONNECTIONS": 10,
            }

    return router