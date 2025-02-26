# core/db/base.py
import asyncio
import os
from enum import Enum
from typing import Dict, Optional, List, Any
import aiosqlite

BASE_PRAGMAS = {
    "journal_mode": "WAL",  # Optimized for concurrency.
    "synchronous": "NORMAL",  # Retains safer disk sync behavior.
    "mmap_size": 17179869184,  # 16GB memory-mapped I/O.
    "cache_size": 5242880,  # ~20GB cache, expressed as pages (4KB each).
    "temp_store": "MEMORY",  # All temporary operations in memory.
    "journal_size_limit": 104857600,  # 100MB WAL journal limit.
}


class DatabaseType(Enum):
    SQLITE = "sqlite"


class DatabaseInfo:
    """Stores database connection and related information."""

    def __init__(
        self,
        name: str,
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
    def __init__(self, database_config: Dict = None):
        self.CONNECTIONS: Dict[str, DatabaseInfo] = {}
        self.CONFIG = {}
        self.base_path = os.environ["DATABASE_PATH"]
        # Ensure base directory exists
        os.makedirs(self.base_path, exist_ok=True)
        if database_config:
            self.apply_config(database_config)

    def get_default_config(self, app_name: str, test_mode: bool = False) -> Dict:
        """Generate default configuration for an app."""
        # Create app-specific directory
        db_path = os.path.join(self.base_path, f"{app_name}.sqlite3")
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        if test_mode:
            return {
                "ENGINE": "aiosqlite",
                "NAME": ":memory:",
                "PRAGMAS": BASE_PRAGMAS.copy(),
            }

        return {"ENGINE": "aiosqlite", "NAME": db_path, "PRAGMAS": BASE_PRAGMAS.copy()}

    async def _init_sqlite_connection(self, config: Dict) -> aiosqlite.Connection:
        """Initialize SQLite connection."""
        db_path = config["NAME"]

        if db_path != ":memory:":
            # Ensure database directory exists
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        connection = await aiosqlite.connect(db_path)
        connection.row_factory = aiosqlite.Row

        await self.apply_sqlite_pragmas(connection, config.get("PRAGMAS", {}))
        return connection

    def apply_config(self, config: Dict):
        """Apply database configuration."""
        for alias, cfg in config.items():
            if alias in self.CONFIG:
                return
            self.CONFIG[alias] = cfg

    async def wait_for_database(self, config: Dict, timeout: int = 30) -> None:
        """Wait for database to become available."""
        start_time = asyncio.get_event_loop().time()

        while True:
            try:
                if config["NAME"] != ":memory:":
                    conn = await aiosqlite.connect(config["NAME"])
                    await conn.close()
                return
            except aiosqlite.OperationalError as e:
                if asyncio.get_event_loop().time() - start_time > timeout:
                    raise TimeoutError(
                        f"Database connection timeout after {timeout} seconds: {str(e)}"
                    )
                await asyncio.sleep(1)

    async def get_connection(
        self, alias: str, retry: bool = True, timeout: int = 30, test_mode: bool = False
    ) -> DatabaseInfo:
        """Initialize a database connection based on configuration with retry logic."""
        if len(alias) == 0:
            raise ValueError("Database alias cannot be empty")

        # Auto-configure database if not explicitly configured
        if alias not in self.CONFIG:
            self.CONFIG[alias] = self.get_default_config(alias, test_mode=test_mode)

        if alias in self.CONNECTIONS:
            return self.CONNECTIONS[alias]

        try:

            config = self.CONFIG[alias]
            engine = config.get("ENGINE", "").lower()

            if engine != "aiosqlite":
                raise ValueError(f"Unsupported database engine: {engine}")

            if retry:
                await self.wait_for_database(config, timeout)

            connection = await self._init_sqlite_connection(config)
            db_info = DatabaseInfo(
                name=alias,
                connection=connection,
                db_type=DatabaseType.SQLITE,
                config=config,
            )
            self.CONNECTIONS[alias] = db_info
            return db_info

        except Exception as e:
            print(f"Error initializing database '{alias}': {e}")
            raise

    async def apply_sqlite_pragmas(
        self, connection: aiosqlite.Connection, pragmas: Dict
    ):
        """Apply PRAGMA settings for SQLite."""
        for pragma, value in pragmas.items():
            try:
                await connection.execute(f"PRAGMA {pragma}={value};")
            except aiosqlite.OperationalError as e:
                print(f"Error applying PRAGMA {pragma}: {e}")
        await connection.commit()

    async def close_connection(self, alias: str):
        """Close a specific database connection."""
        if alias in self.CONNECTIONS:
            try:
                await self.CONNECTIONS[alias].close()
            finally:
                del self.CONNECTIONS[alias]

    async def shutdown(self, aliases: Optional[List[str]] = None):
        """Shutdown all connections."""
        if aliases is None:
            aliases = list(self.CONFIG.keys())

        for alias in aliases:
            await self.close_connection(alias)

    def configure_app(
        self,
        app: str,
        path: Optional[str] = None,
        pragmas: Optional[Dict] = None,
        memory: bool = False,
    ):
        """Configure an app's database with custom settings."""
        if memory:
            config = {
                "ENGINE": "aiosqlite",
                "NAME": ":memory:",
                "PRAGMAS": BASE_PRAGMAS.copy(),
            }
        else:
            config = self.get_default_config(app)

        if path:
            config["NAME"] = path

        if pragmas:
            config["PRAGMAS"].update(pragmas)

        self.CONFIG[app] = config
        return config


# Global instance
DATABASES = BaseDatabase()
