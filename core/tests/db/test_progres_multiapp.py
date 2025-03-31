# core/tests/db/test_progres_multiapp.py
from unittest import skipIf
import logging

from core.db.base import DATABASES, use_postgres_database
from core.db.fields import CharField, IntegerField, BooleanField
from core.db.model import Model
from core.tests.db.postgres_testcase import SKIP_POSTGRES_TESTS, PostgresTestCase, POSTGRES_CONFIG
from core.db.postgres import PostgreSQLDatabaseInfo


# Define models for different apps
class App1Model(Model):
    name = CharField(max_length=100)
    value = IntegerField()

    @classmethod
    def get_app_label(cls) -> str:
        return "app1"

    @classmethod
    def get_table_name(cls) -> str:
        """Return the table name without app prefix for PostgreSQL."""
        return "app1model"  # Simple table name without app prefix


class App2Model(Model):
    title = CharField(max_length=100)
    active = BooleanField(default=True)

    @classmethod
    def get_app_label(cls) -> str:
        return "app2"

    @classmethod
    def get_table_name(cls) -> str:
        """Return the table name without app prefix for PostgreSQL."""
        return "app2model"  # Simple table name without app prefix


@skipIf(SKIP_POSTGRES_TESTS or not POSTGRES_CONFIG.get("HOST"), "PostgreSQL tests disabled or not configured")
class PostgresMultiAppTestCase(PostgresTestCase):
    """Test multiple apps with PostgreSQL using schemas."""

    # Test case configuration
    db_alias = "postgres_multiapp_test"  # Use a specific alias for this test DB

    # No models or app_label here, handled in custom setup

    async def asyncSetUp(self):
        """Custom setup for multi-app test using schemas."""
        self.logger = logging.getLogger(f"{self.__module__}.{self.__class__.__name__}")
        self.logger.setLevel(logging.ERROR)

        # 1. Reset Global State
        await DATABASES.shutdown()
        DATABASES.CONFIG = {}
        DATABASES.CONNECTIONS = {}

        # 2. Configure the single PostgreSQL DB connection
        self.CONFIG = {
            self.db_alias: {
                "ENGINE": "asyncpg",
                "HOST": POSTGRES_CONFIG["HOST"],
                "PORT": POSTGRES_CONFIG["PORT"],
                "USER": POSTGRES_CONFIG["USER"],
                "PASSWORD": POSTGRES_CONFIG["PASSWORD"],
                "NAME": POSTGRES_CONFIG["NAME"],
                "SETTINGS": {},
                "MIN_CONNECTIONS": 1, "MAX_CONNECTIONS": 5,
            }
        }
        DATABASES.apply_config(self.CONFIG)

        # Configure app1 and app2 to use the same database
        DATABASES.CONFIG["app1"] = DATABASES.CONFIG[self.db_alias]
        DATABASES.CONFIG["app2"] = DATABASES.CONFIG[self.db_alias]

        # 3. Set Router to route all apps to the single test DB alias
        self.router = use_postgres_database(self.db_alias)

        # 4. Get Connection & Ensure it's PostgreSQL
        try:
            self.connection_info = await DATABASES.get_connection(self.db_alias)
            if not isinstance(self.connection_info, PostgreSQLDatabaseInfo):
                raise TypeError("Expected PostgreSQL connection")
        except Exception as e:
            self.logger.error(f"Failed to get/verify PostgreSQL connection for '{self.db_alias}': {e}", exc_info=True)
            self.skipTest(f"Could not connect/verify PostgreSQL DB '{self.db_alias}': {e}")
            return

        # 5. Manually create schemas
        for app_label in ["app1", "app2"]:
            try:
                schema_query = f'CREATE SCHEMA IF NOT EXISTS "{app_label}"'
                await self.connection_info.execute(schema_query)
            except Exception as schema_err:
                self.logger.error(f"Failed to ensure schema '{app_label}': {schema_err}", exc_info=True)
                # Attempt cleanup before skipping
                await self.asyncTearDown()
                self.skipTest(f"Could not create schema '{app_label}': {schema_err}")
                return

        # 6. Manually setup app1 and app2 connections
        DATABASES.CONNECTIONS["app1"] = self.connection_info
        DATABASES.CONNECTIONS["app2"] = self.connection_info

        # 7. Manually create tables for App1Model and App2Model
        await self.create_app_tables()

        # 8. Verify tables exist
        tables_exist = await self.verify_tables_exist()
        if not tables_exist:
            self.skipTest("Tables not created properly. Skipping tests.")

    async def create_app_tables(self):
        """Manually create tables for each app model."""
        # App1Model table
        app1_create_table = """
        CREATE TABLE IF NOT EXISTS "app1"."app1model" (
            "id" TEXT PRIMARY KEY,
            "name" TEXT NOT NULL,
            "value" INTEGER NOT NULL
        )
        """

        # App2Model table
        app2_create_table = """
        CREATE TABLE IF NOT EXISTS "app2"."app2model" (
            "id" TEXT PRIMARY KEY,
            "title" TEXT NOT NULL,
            "active" BOOLEAN NOT NULL DEFAULT TRUE
        )
        """

        try:
            await self.connection_info.execute(app1_create_table)
            await self.connection_info.execute(app2_create_table)
        except Exception as e:
            self.logger.error(f"Error creating tables: {e}", exc_info=True)
            raise

    async def verify_tables_exist(self):
        """Verify that tables exist in the appropriate schemas."""
        # Check app1 schema and table
        app1_result = await self.connection_info.fetch_one("""
            SELECT EXISTS (SELECT 1 FROM information_schema.tables 
                          WHERE table_schema = 'app1' AND table_name = 'app1model')
        """)
        app1_exists = app1_result and app1_result["exists"]

        # Check app2 schema and table
        app2_result = await self.connection_info.fetch_one("""
            SELECT EXISTS (SELECT 1 FROM information_schema.tables 
                          WHERE table_schema = 'app2' AND table_name = 'app2model')
        """)
        app2_exists = app2_result and app2_result["exists"]

        if not app1_exists:
            self.logger.error("Table app1.app1model does not exist!")
        if not app2_exists:
            self.logger.error("Table app2.app2model does not exist!")

        return app1_exists and app2_exists

    async def asyncTearDown(self):
        """Clean up schemas created during the test."""
        db_info = DATABASES.CONNECTIONS.get(self.db_alias)
        if db_info and isinstance(db_info, PostgreSQLDatabaseInfo):
            apps_to_cleanup = ["app1", "app2"]
            for app_label in apps_to_cleanup:
                try:
                    drop_schema_sql = f'DROP SCHEMA IF EXISTS "{app_label}" CASCADE'
                    # self.logger.debug(f"Executing teardown: {drop_schema_sql}")
                    await db_info.execute(drop_schema_sql)
                except Exception as drop_err:
                    self.logger.error(f"Error dropping schema '{app_label}': {drop_err}", exc_info=True)
        else:
            self.logger.warning(
                f"No PostgreSQL connection info found for alias '{self.db_alias}' during teardown drop.")

        await DATABASES.shutdown([self.db_alias])

    async def test_multi_app_tables(self):
        """Test that tables from different apps can coexist and be queried."""
        # ORM operations use the router, directing them to the correct schema within db_alias
        app1_obj = await App1Model.objects.create(name="App1Object", value=42)
        app2_obj = await App2Model.objects.create(title="App2Object", active=True)

        # Verify App1 instance saved correctly
        saved_app1 = await App1Model.objects.get(id=app1_obj.id)
        self.assertEqual(saved_app1.name, "App1Object")
        self.assertEqual(saved_app1.value, 42)

        # Verify App2 instance saved correctly
        saved_app2 = await App2Model.objects.get(id=app2_obj.id)
        self.assertEqual(saved_app2.title, "App2Object")
        self.assertTrue(saved_app2.active)

        # Verify we can query each table separately
        app1_count = await App1Model.objects.count()
        app2_count = await App2Model.objects.count()

        self.assertEqual(app1_count, 1)
        self.assertEqual(app2_count, 1)

    async def test_multi_app_schema_separation(self):
        """Test schemas are used correctly for different apps."""
        conn = self.connection_info  # Use connection from setup

        # Get the expected table parts using the model's parser
        app1_schema, app1_table_part = "app1", "app1model"
        app2_schema, app2_table_part = "app2", "app2model"

        # Check if tables exist in the correct schemas using information_schema
        table_query = """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE (table_schema = $1 AND table_name = $2) OR (table_schema = $3 AND table_name = $4)
            ORDER BY table_schema, table_name
        """
        tables_found = await conn.fetch_all(table_query, [
            app1_schema, app1_table_part,
            app2_schema, app2_table_part
        ])

        self.assertEqual(len(tables_found), 2, f"Expected 2 tables in specific schemas, found {len(tables_found)}")

        # Verify schemas and names match expectations
        self.assertEqual(tables_found[0]['table_schema'], app1_schema)
        self.assertEqual(tables_found[0]['table_name'], app1_table_part)
        self.assertEqual(tables_found[1]['table_schema'], app2_schema)
        self.assertEqual(tables_found[1]['table_name'], app2_table_part)

        # Create multiple objects (will use the router correctly)
        for i in range(5):
            await App1Model.objects.create(name=f"MultiTest1_{i}", value=i * 10)
            await App2Model.objects.create(title=f"MultiTest2_{i}", active=i % 2 == 0)

        # Query via ORM to verify counts
        app1_count = await App1Model.objects.filter(name__startswith="MultiTest1").count()
        app2_count = await App2Model.objects.filter(title__startswith="MultiTest2").count()

        self.assertEqual(app1_count, 5)
        self.assertEqual(app2_count, 5)