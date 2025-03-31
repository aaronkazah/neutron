# core/tests/db/test_routers.py (renamed from test_strategies.py)

import asyncio
from unittest import IsolatedAsyncioTestCase

from core.db.base import DATABASES, BaseDatabase
from core.db.router import PerAppRouter, SingleDatabaseRouter
from core.db.model import Model
from core.db.fields import CharField
from core.db.migrations import MigrationManager


class TestModel(Model):
    name = CharField(max_length=100)

    @classmethod
    def get_app_label(cls) -> str:
        return "test_router_app"

    @classmethod
    def get_table_name(cls) -> str:
        return "test_router_app_testmodel"


class DatabaseRouterTestCase(IsolatedAsyncioTestCase):  # Renamed from DatabaseStrategyTestCase

    async def asyncSetUp(self):
        # Create a fresh database instance for each test
        self.test_db = BaseDatabase()

    async def asyncTearDown(self):
        await self.test_db.shutdown()

    async def test_per_app_router(self):  # Renamed from test_per_app_strategy
        """Test that PerAppRouter routes each app to its own database."""
        # Set the router
        self.test_db.set_router(PerAppRouter())

        # Configure a test app
        app_label = "test_app_1"
        self.test_db.configure_app(app_label, memory=True)

        # Get connection using the app label
        connection = await self.test_db.get_connection(app_label)

        # Verify the connection is stored under the app label
        self.assertEqual(connection.name, app_label)
        self.assertIn(app_label, self.test_db.CONNECTIONS)

        # Verify another app gets a different connection
        app_label2 = "test_app_2"
        self.test_db.configure_app(app_label2, memory=True)
        connection2 = await self.test_db.get_connection(app_label2)

        self.assertEqual(connection2.name, app_label2)
        self.assertNotEqual(connection.name, connection2.name)

    async def test_single_database_router(self):  # Renamed from test_single_database_strategy
        """Test that SingleDatabaseRouter routes all apps to a single database."""
        # Set up the SingleDatabaseRouter with default alias
        default_alias = "test_single_db"
        self.test_db.set_router(SingleDatabaseRouter(default_alias))

        # Configure the default database
        self.test_db.configure_app(default_alias, memory=True)

        # Get connections for multiple apps
        app_label1 = "test_app_1"
        app_label2 = "test_app_2"

        connection1 = await self.test_db.get_connection(app_label1)
        connection2 = await self.test_db.get_connection(app_label2)

        # Verify both connections point to the same database
        self.assertEqual(connection1.name, default_alias)
        self.assertEqual(connection2.name, default_alias)
        self.assertEqual(connection1, connection2)

    async def test_model_with_different_routers(self):  # Renamed from test_model_with_different_strategies
        """Test model operations with different database routers."""
        # Test with PerAppRouter
        self.test_db.set_router(PerAppRouter())
        app_label = TestModel.get_app_label()

        # Make sure to configure the test_router_app database
        self.test_db.configure_app(app_label, memory=True)

        # Get connection for the database operations
        conn = await self.test_db.get_connection(app_label)

        # Create the table manually before using it
        table_name = TestModel.get_table_name()
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL
        )
        """
        await conn.connection.execute(create_table_sql)
        await conn.connection.commit()

        # Create a model instance and save it
        instance1 = TestModel(name="Test with PerAppRouter")
        instance1.db_alias = app_label

        # Save the instance to our test database
        await self._save_instance(instance1, conn)

        # Verify the model was saved in the app-specific database
        cursor = await conn.connection.execute(
            f"SELECT * FROM {table_name} WHERE name = ?",
            ["Test with PerAppRouter"]
        )
        row = await cursor.fetchone()
        self.assertIsNotNone(row)

        # Test with SingleDatabaseRouter
        await self.test_db.shutdown()  # Close previous connections
        self.test_db = BaseDatabase()  # Fresh start

        default_alias = "test_single_db"
        self.test_db.set_router(SingleDatabaseRouter(default_alias))
        self.test_db.configure_app(default_alias, memory=True)

        # Get connection for the database operations
        conn = await self.test_db.get_connection(default_alias)

        # Create the table manually
        await conn.connection.execute(create_table_sql)
        await conn.connection.commit()

        # Create a model instance and save it
        instance2 = TestModel(name="Test with SingleDatabaseRouter")
        instance2.db_alias = default_alias

        # Save using our custom method
        await self._save_instance(instance2, conn)

        # Verify the model was saved in the default database
        cursor = await conn.connection.execute(
            f"SELECT * FROM {table_name} WHERE name = ?",
            ["Test with SingleDatabaseRouter"]
        )
        row = await cursor.fetchone()
        self.assertIsNotNone(row)

    async def _save_instance(self, instance, connection):
        """Custom save method that directly inserts into the database."""
        table_name = instance.get_table_name()

        # Generate an ID if necessary
        if not instance.id:
            instance.id = f"test-{asyncio.get_event_loop().time()}"

        # Create a simple insert statement
        query = f"INSERT INTO {table_name} (id, name) VALUES (?, ?)"
        await connection.connection.execute(query, (instance.id, instance.name))
        await connection.connection.commit()