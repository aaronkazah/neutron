import datetime
import enum
import logging
import tempfile
import os
import shutil

from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock

from core.db.editor import get_schema_editor
from core.db.fields import (
    IntegerField,
    CharField,
    TextField,
    DateTimeField,
    JSONField,
    VectorField,
    BooleanField,
    EnumField,
)
from core.db.migrations import (
    Migration,
    CreateModel,
    DeleteModel,
    AddField,
    RemoveField,
    AlterField,
    RenameField,
    RenameModel,
    MigrationManager,
)
from core.db.model import Model
from core.db.base import DATABASES, BASE_PRAGMAS


# --- Helper Functions ---


async def get_table_columns(connection, table_name):
    """Get detailed column information for a table."""
    cursor = await connection.execute(f"PRAGMA table_info({table_name})")
    try:
        columns = await cursor.fetchall()
        # Return a dictionary mapping column names to their types
        return {row["name"]: row["type"] for row in columns}
    finally:
        await cursor.close()


async def table_exists(connection, table_name):
    """Check if a table exists."""
    cursor = await connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
    )
    try:
        result = await cursor.fetchone()
        return result is not None
    finally:
        await cursor.close()


async def get_index_info(connection, table_name):
    """Helper function to fetch index information for a table."""
    # Get list of indexes
    cursor = await connection.execute(f"PRAGMA index_list({table_name})")
    indexes = await cursor.fetchall()
    await cursor.close()

    index_info = []
    for index in indexes:
        # Get details for each index
        cursor = await connection.execute(f"PRAGMA index_info('{index[1]}')")
        index_details = await cursor.fetchall()
        await cursor.close()

        # index[1] is the index name, index[2] is whether it's unique (0 or 1)
        # index_details contains the column information
        index_info.append(
            (
                index[1],  # index name
                bool(index[2]),  # is unique
                [detail[2] for detail in index_details],  # column names
            )
        )

    return index_info


class MigrationTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.app_label = "test_app"
        self.db_alias = "migration_test"
        CONFIG = {
            self.db_alias: {
                "ENGINE": "aiosqlite",
                "NAME": ":memory:",
                "PRAGMAS": BASE_PRAGMAS.copy(),
            }
        }
        DATABASES.apply_config(CONFIG)

    async def asyncSetUp(self):
        logging.basicConfig(level=logging.DEBUG)
        self.connection = await DATABASES.get_connection(self.db_alias)
        self.schema_editor = get_schema_editor(self.connection)

    async def asyncTearDown(self):
        async with await self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ) as cursor:
            tables = await cursor.fetchall()
            for table in tables:
                await self.connection.execute(f"DROP TABLE IF EXISTS {table[0]}")
        await self.connection.commit()
        await DATABASES.shutdown()

    async def apply_migration(self, migration):
        """Helper method to apply a migration."""
        async with get_schema_editor(self.connection) as schema_editor:
            await migration.apply(
                project_state={},
                schema_editor=schema_editor,
                connection=self.connection,
            )
            await self.connection.commit()

    async def get_table_name(self, model_name):
        """Get the full table name including app prefix."""
        return f"{self.app_label}_{model_name.lower()}"

    async def table_exists(self, connection, model_name):
        """Check if table exists with proper app prefix."""
        table_name = await self.get_table_name(model_name)
        cursor = await connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        try:
            result = await cursor.fetchone()
            return result is not None
        finally:
            await cursor.close()

    async def get_table_columns(self, connection, model_name):
        """Get columns with proper table prefix."""
        table_name = await self.get_table_name(model_name)
        cursor = await connection.execute(f"PRAGMA table_info({table_name})")
        try:
            columns = await cursor.fetchall()
            return {col[1]: col[2] for col in columns}
        finally:
            await cursor.close()

    async def test_create_model_basic(self):
        """Test basic model creation migration."""
        migration = Migration(
            self.app_label,
            operations=[
                CreateModel(
                    f"{self.app_label}.TestModel",
                    fields={
                        "id": IntegerField(primary_key=True),
                        "name": CharField(max_length=100),
                        "created_at": DateTimeField(default=datetime.datetime.now),
                    },
                )
            ],
        )

        await self.apply_migration(migration)

        # Verify table was created
        self.assertTrue(await self.table_exists(self.connection, "TestModel"))

        # Verify columns were created correctly
        columns = await self.get_table_columns(self.connection, "TestModel")
        self.assertEqual(columns["id"], "INTEGER")
        self.assertEqual(columns["name"], "VARCHAR(100)")
        self.assertEqual(columns["created_at"], "TEXT")


class StatusEnum(enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


class ComplexMigrationTestCase(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.app_label = "test_app"
        self.db_alias = "complex_migration_test"
        CONFIG = {
            self.db_alias: {
                "ENGINE": "aiosqlite",
                "NAME": ":memory:",
                "PRAGMAS": BASE_PRAGMAS.copy(),
            }
        }
        DATABASES.apply_config(CONFIG)
        logging.basicConfig(level=logging.DEBUG)
        self.connection = await DATABASES.get_connection(self.db_alias)
        self.schema_editor = get_schema_editor(self.connection)

    async def asyncTearDown(self):
        await DATABASES.shutdown()

    async def apply_migration(self, migration):
        """Helper method to apply a migration."""
        async with get_schema_editor(self.connection) as schema_editor:
            await migration.apply(
                project_state={},
                schema_editor=schema_editor,
                connection=self.connection,
            )
            await self.connection.commit()

    async def get_table_name(self, model_name):
        """Get the full table name including app prefix."""
        return f"{self.app_label}_{model_name.lower()}"

    async def table_exists(self, connection, model_name):
        """Check if table exists with proper app prefix."""
        table_name = await self.get_table_name(model_name)
        cursor = await connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        try:
            result = await cursor.fetchone()
            return result is not None
        finally:
            await cursor.close()

    async def get_table_columns(self, connection, model_name):
        """Get columns with proper table prefix."""
        table_name = await self.get_table_name(model_name)
        cursor = await connection.execute(f"PRAGMA table_info({table_name})")
        try:
            columns = await cursor.fetchall()
            return {col[1]: col[2] for col in columns}
        finally:
            await cursor.close()

    async def test_hyper_complex_migrations(self):
        # Migration 1: Create ModelA
        migration_1 = Migration(
            self.app_label,
            operations=[
                CreateModel(
                    f"{self.app_label}.ModelA",
                    fields={
                        "id": IntegerField(primary_key=True),
                        "name": CharField(max_length=100),
                        "created_at": DateTimeField(default=datetime.datetime.now),
                    },
                )
            ],
        )
        await self.apply_migration(migration_1)

        self.assertTrue(await self.table_exists(self.connection, "ModelA"))

        # Migration 2: Add field 'age' to ModelA
        migration_2 = Migration(
            self.app_label,
            dependencies=[(self.app_label, migration_1)],
            operations=[
                AddField(f"{self.app_label}.ModelA", "age", IntegerField(null=True))
            ],
        )
        await self.apply_migration(migration_2)

        columns = await self.get_table_columns(self.connection, "ModelA")
        self.assertEqual(columns["age"], "INTEGER")

        # Migration 3: Rename ModelA to ModelB
        migration_3 = Migration(
            self.app_label,
            dependencies=[(self.app_label, migration_2)],
            operations=[
                RenameModel(f"{self.app_label}.ModelA", f"{self.app_label}.ModelB")
            ],
        )
        await self.apply_migration(migration_3)

        self.assertTrue(await self.table_exists(self.connection, "ModelB"))
        self.assertFalse(await self.table_exists(self.connection, "ModelA"))

    async def test_migration_from_scratch(self):
        # Initial setup - create ModelA
        migration_1 = Migration(
            self.app_label,
            operations=[
                CreateModel(
                    f"{self.app_label}.ModelA",
                    fields={
                        "id": IntegerField(primary_key=True),
                        "name": CharField(max_length=100),
                        "created_at": DateTimeField(default=datetime.datetime.now),
                    },
                )
            ],
        )
        await self.apply_migration(migration_1)

        # Evolution to ModelB
        migration_2 = Migration(
            self.app_label,
            dependencies=[(self.app_label, migration_1)],
            operations=[
                RenameModel(f"{self.app_label}.ModelA", f"{self.app_label}.ModelB"),
                AddField(f"{self.app_label}.ModelB", "age", IntegerField(null=True)),
            ],
        )
        await self.apply_migration(migration_2)

        # Verify final state
        self.assertTrue(await self.table_exists(self.connection, "ModelB"))
        self.assertFalse(await self.table_exists(self.connection, "ModelA"))

        columns = await self.get_table_columns(self.connection, "ModelB")
        self.assertEqual(columns["id"], "INTEGER")
        self.assertEqual(columns["name"], "VARCHAR(100)")
        self.assertEqual(columns["age"], "INTEGER")
        self.assertEqual(columns["created_at"], "TEXT")


# --- MigrationManagerTestCase ---
class TestModel1(Model):
    name = CharField(max_length=100)

    @classmethod
    def get_app_label(cls) -> str:
        return "test_app"


class TestModel2(Model):
    value = IntegerField()

    @classmethod
    def get_app_label(cls) -> str:
        return "test_app"


class MigrationManagerTestCase(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.app_label = "test_app"
        self.manager = MigrationManager(base_dir=tempfile.mkdtemp())  # Use temp dir
        self.db_alias = "test_migration_db"

        CONFIG = {
            self.db_alias: {
                "ENGINE": "aiosqlite",
                "NAME": ":memory:",
                "PRAGMAS": BASE_PRAGMAS.copy(),
            }
        }
        DATABASES.apply_config(CONFIG)
        self.connection = await DATABASES.get_connection(self.db_alias)

    async def asyncTearDown(self):
        await DATABASES.shutdown()
        if hasattr(self, "manager") and hasattr(self.manager, "base_dir"):
            shutil.rmtree(self.manager.base_dir)

    async def test_makemigrations_create_model(self):
        # Create test model
        class TestModel(Model):
            name = CharField(max_length=100)

            @classmethod
            def get_app_label(cls):
                return "test_app"

        models = [TestModel]

        # Generate migrations
        operations = await self.manager.makemigrations(
            self.app_label,
            models=models,
            return_ops=True,  # Just get operations
            clean=True,  # Start fresh
        )

        # Verify operations
        self.assertEqual(len(operations), 1)
        self.assertIsInstance(operations[0], CreateModel)
        self.assertEqual(operations[0].model_name, f"{self.app_label}.TestModel")
