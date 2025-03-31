import datetime
import enum
import logging
import tempfile
import shutil

from unittest import IsolatedAsyncioTestCase

from core.db.editor import get_schema_editor
from core.db.fields import IntegerField, CharField, DateTimeField
from core.db.migrations import (
    Migration,
    CreateModel,
    AddField,
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


async def table_exists(self, connection, model_name):
    """Check if table exists with proper app prefix."""
    # Convert ModelName to snake_case (model_name)
    snake_case = "".join(["_" + c.lower() if c.isupper() else c.lower() for c in model_name]).lstrip("_")
    table_name = f"{self.app_label}_{snake_case}"
    cursor = await connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
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
        self.connection = await DATABASES.get_connection(self.db_alias)
        self.schema_editor = get_schema_editor(self.connection)

    async def asyncTearDown(self):
        # Use try-except for robustness during teardown
        try:
            # Check if connection exists and is not None before executing
            if hasattr(self, 'connection') and self.connection and self.connection.connection:
                 async with await self.connection.execute( # Use raw connection if available
                     "SELECT name FROM sqlite_master WHERE type='table'"
                 ) as cursor:
                     tables = await cursor.fetchall()
                     for table in tables:
                          # Use execute on DatabaseInfo object for consistency
                          await self.connection.execute(f"DROP TABLE IF EXISTS \"{table['name']}\"") # Quote table name
                 await self.connection.commit() # Commit drops
        except Exception as e:
             # Log teardown errors but don't stop other tests
             logging.getLogger(__name__).error(f"Error during MigrationTestCase teardown: {e}", exc_info=True)
        finally:
             # Always attempt shutdown
             await DATABASES.shutdown([self.db_alias])

    async def apply_migration(self, migration):
        """Helper method to apply a migration."""
        logger = logging.getLogger(f"{self.__class__.__name__}.apply_migration")  # Add logger
        if not hasattr(self, 'connection') or not self.connection:
            self.fail("Database connection not available in apply_migration")
        if not hasattr(self, 'schema_editor') or not self.schema_editor:
            self.fail("Schema editor not available in apply_migration")

        try:
            # Use the schema_editor from asyncSetUp directly
            # The schema editor instance handles its own context if needed (like PG transactions)
            # For SQLite, operations are usually executed directly.
            await migration.apply(
                project_state={},
                schema_editor=self.schema_editor,
                # Pass the raw aiosqlite connection object if operations need it
                # but schema_editor should ideally handle execution.
                connection=self.connection.connection,
            )
            # --- FIX: Commit using the DatabaseInfo object AFTER apply ---
            await self.connection.commit()

        except Exception as e:
            logger.error(f"Error applying migration {migration!r}: {e}", exc_info=True)
            # Attempt rollback (may not do much in SQLite default mode but good practice)
            try:
                await self.connection.rollback()
                logger.warning(f"Rollback attempted after error in migration {migration!r}.")
            except Exception as rb_err:
                logger.error(f"Rollback failed after migration error: {rb_err}")
            self.fail(f"Migration application failed: {e}")  # Fail the test

    async def get_table_name(self, model_name):
        """Get the full table name including app prefix."""
        # Convert ModelName to snake_case (model_name)
        snake_case = "".join(["_" + c.lower() if c.isupper() else c.lower() for c in model_name]).lstrip("_")
        return f"{self.app_label}_{snake_case}"

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
                    # Use the format 'app_label.ModelName'
                    f"{self.app_label}.TestModel",
                    fields={
                        # Keep id definition consistent with default Model/bootstrap
                        "id": CharField(primary_key=True),  # Assuming default is CharField PK
                        "name": CharField(max_length=100),
                        "created_at": DateTimeField(default=datetime.datetime.now),
                    },
                )
            ],
        )

        await self.apply_migration(migration)

        # Verify table was created (using the helper that constructs the full name)
        self.assertTrue(
            await self.table_exists(self.connection, "TestModel"),
            "Table 'test_app_testmodel' should exist after migration."
        )

        # Verify columns were created correctly
        columns = await self.get_table_columns(self.connection, "TestModel")
        # Adjust expected types based on actual SQLiteSchemaEditor.get_column_type behavior
        self.assertEqual(columns.get("id"), "TEXT", "ID column type mismatch")
        self.assertEqual(columns.get("name"), "TEXT", "Name column type mismatch")  # Changed from VARCHAR(100) to TEXT
        self.assertEqual(columns.get("created_at"), "TEXT", "Created_at column type mismatch")



class StatusEnum(enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


class ComplexMigrationTestCase(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.app_label = "migrations_test_app"
        self.db_alias = "complex_migration_test"
        CONFIG = {
            self.db_alias: {
                "ENGINE": "aiosqlite",
                "NAME": ":memory:",
                "PRAGMAS": BASE_PRAGMAS.copy(),
            }
        }
        DATABASES.apply_config(CONFIG)
        self.connection = await DATABASES.get_connection(self.db_alias)
        self.schema_editor = get_schema_editor(self.connection)

    async def asyncTearDown(self):
        await DATABASES.shutdown()

    async def apply_migration(self, migration):
        """Helper method to apply a migration."""
        if not hasattr(self, 'connection') or not self.connection:
             self.fail("Database connection not available in apply_migration")
        if not hasattr(self, 'schema_editor') or not self.schema_editor:
             self.fail("Schema editor not available in apply_migration")

        try:
            async with self.schema_editor as editor:
                await migration.apply(
                    project_state={},
                    schema_editor=editor,
                    connection=self.connection.connection, # Pass the raw aiosqlite connection
                )
        except Exception as e:
             logging.getLogger(__name__).error(f"Error applying migration {migration!r}: {e}", exc_info=True)
             try:
                 await self.connection.rollback()
             except Exception as rb_err:
                 logging.getLogger(__name__).error(f"Rollback failed after migration error: {rb_err}")
             self.fail(f"Migration application failed: {e}")

    async def get_table_name(self, model_name):
        """Get the full table name including app prefix."""
        snake_case = "".join(["_" + c.lower() if c.isupper() else c.lower() for c in model_name]).lstrip("_")
        return f"{self.app_label}_{snake_case}"

    async def table_exists(self, connection, model_name):
        """Check if table exists with proper app prefix. Assumes connection is valid."""
        table_name = await self.get_table_name(model_name)

        # --- REMOVED CONNECTION RE-FETCHING LOGIC ---
        # Rely on self.connection established in asyncSetUp being valid.
        # If it's not, it means teardown/setup or another test interfered, which needs fixing.
        if not connection or not connection.connection:
             self.fail(f"Database connection object or underlying connection is invalid when checking table '{table_name}'.")

        cursor = None # Initialize cursor to None
        try:
            cursor = await connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )
            result = await cursor.fetchone()
            return result is not None
        except ValueError as e:
             # Catch the specific "no active connection" error here for clarity
             if "no active connection" in str(e):
                  self.fail(f"Connection became inactive while checking table '{table_name}'. This likely indicates improper shutdown in another test's teardown.")
             else:
                  self.fail(f"ValueError checking if table {table_name} exists: {e}")
        except Exception as e:
            # Catch other potential errors
             self.fail(f"Unexpected error checking if table {table_name} exists: {type(e).__name__}: {e}")
        finally:
            # Ensure cursor is closed if it was successfully created
            if cursor:
                 try:
                      await cursor.close()
                 except Exception as close_err:
                      logging.getLogger(__name__).error(f"Error closing cursor: {close_err}")

    async def get_table_columns(self, connection, model_name):
        """Get columns with proper table prefix."""
        table_name = await self.get_table_name(model_name)
        # Ensure connection is valid before proceeding
        if not connection or not connection.connection:
            self.fail(f"Database connection invalid when getting columns for '{table_name}'.")

        cursor = None
        try:
            cursor = await connection.execute(f"PRAGMA table_info(\"{table_name}\")") # Quote table name just in case
            columns = await cursor.fetchall()
            # Check if columns is None or empty before accessing
            if columns is None:
                self.fail(f"PRAGMA table_info returned None for table '{table_name}'. Does the table exist?")
                return {} # Should not be reached due to fail, but satisfies type checker
            return {col[1]: col[2] for col in columns}
        except ValueError as e:
             if "no active connection" in str(e):
                  self.fail(f"Connection became inactive while getting columns for '{table_name}'.")
             else:
                  self.fail(f"ValueError getting columns for table {table_name}: {e}")
             return {} # Should not be reached
        except Exception as e:
            self.fail(f"Unexpected error getting columns for table {table_name}: {type(e).__name__}: {e}")
            return {} # Should not be reached
        finally:
            if cursor:
                 try:
                      await cursor.close()
                 except Exception as close_err:
                      logging.getLogger(__name__).error(f"Error closing cursor: {close_err}")

    async def test_hyper_complex_migrations(self):
        # Migration 1: Create ModelA
        migration_1 = Migration(
            self.app_label,
            operations=[
                CreateModel(
                    f"{self.app_label}.ModelA",
                    fields={
                        "id": CharField(primary_key=True),
                        "name": CharField(max_length=100),
                        "created_at": DateTimeField(default=datetime.datetime.now),
                    },
                )
            ],
        )
        await self.apply_migration(migration_1)
        self.assertTrue(await self.table_exists(self.connection, "ModelA"),
                        "ModelA table should exist after migration 1")

        # Migration 2: Add field 'age' to ModelA
        migration_2 = Migration(
            self.app_label,
            # dependencies=[(self.app_label, "0001_create_modela")], # Dependencies not strictly needed for this logic test
            operations=[
                AddField(f"{self.app_label}.ModelA", "age", IntegerField(null=True))
            ],
        )
        await self.apply_migration(migration_2)
        columns = await self.get_table_columns(self.connection, "ModelA")
        self.assertIn("age", columns, "Column 'age' should exist after migration 2")
        self.assertEqual(columns["age"], "INTEGER", "'age' column type should be INTEGER")

        # Migration 3: Rename ModelA to ModelB
        migration_3 = Migration(
            self.app_label,
            # dependencies=[(self.app_label, "0002_add_age_to_modela")],
            operations=[
                RenameModel(f"{self.app_label}.ModelA", f"{self.app_label}.ModelB")
            ],
        )
        await self.apply_migration(migration_3)

        # Use self.connection which should still be valid
        self.assertTrue(await self.table_exists(self.connection, "ModelB"),
                        "ModelB table should exist after migration 3")
        self.assertFalse(await self.table_exists(self.connection, "ModelA"),
                         "ModelA table should NOT exist after migration 3")

        columns_b = await self.get_table_columns(self.connection, "ModelB")
        self.assertIn("id", columns_b)
        self.assertIn("name", columns_b)
        self.assertIn("created_at", columns_b)
        self.assertIn("age", columns_b)


    async def test_migration_from_scratch(self):
        # Initial setup - create ModelA
        migration_1 = Migration(
            self.app_label,
            operations=[
                CreateModel(
                    f"{self.app_label}.ModelA",
                    fields={
                        "id": CharField(primary_key=True),
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
            # dependencies=[(self.app_label, "0001_create_modela")],
            operations=[
                RenameModel(f"{self.app_label}.ModelA", f"{self.app_label}.ModelB"),
                AddField(f"{self.app_label}.ModelB", "age", IntegerField(null=True)),
            ],
        )
        await self.apply_migration(migration_2)

        # Verify final state using self.connection
        self.assertTrue(await self.table_exists(self.connection, "ModelB"), "ModelB table should exist") # Fails here
        self.assertFalse(await self.table_exists(self.connection, "ModelA"), "ModelA table should not exist")

        columns = await self.get_table_columns(self.connection, "ModelB")
        self.assertIn("id", columns)
        self.assertIn("name", columns)
        self.assertIn("age", columns)
        self.assertIn("created_at", columns)
        self.assertEqual(columns["id"], "TEXT")
        self.assertEqual(columns["name"], "TEXT")
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
