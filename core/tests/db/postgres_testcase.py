# core/tests/db/postgres_testcase.py
import os
import logging
import uuid
import asyncio
import socket
from typing import Tuple
from unittest import IsolatedAsyncioTestCase, skipIf

import asyncpg # Keep import

from core.db.base import DATABASES, use_postgres_database, DatabaseType # Added DatabaseType
from core.db.migrations import MigrationManager
from core.db.postgres import PostgreSQLDatabaseInfo # Keep import

# Default PostgreSQL configuration - Keep as is
POSTGRES_CONFIG = {
    "HOST": os.environ.get("POSTGRES_HOST", "localhost"),
    "PORT": int(os.environ.get("POSTGRES_PORT", "5432")),
    "USER": os.environ.get("POSTGRES_USER", "postgres"),
    "PASSWORD": os.environ.get("POSTGRES_PASSWORD", "postgres"),
    "NAME": os.environ.get("POSTGRES_DB", "postgres"),  # Default system database
}

# Check if PostgreSQL tests should be skipped (keep this logic)
SKIP_POSTGRES_TESTS = os.environ.get("SKIP_POSTGRES_TESTS", "").lower() in ("true", "1", "yes")

class PostgresTestCase(IsolatedAsyncioTestCase):
    """Base test case for PostgreSQL tests (Simplified Setup/Teardown)."""

    # Override these in subclasses if needed
    models = []
    app_label = 'postgres_test' # Default, can be overridden
    db_alias = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.logger = logging.getLogger(f"{cls.__module__}.{cls.__name__}")
        # Set db_alias based on app_label if not explicitly set
        if cls.db_alias is None:
             cls.db_alias = cls.app_label
        # Handle skipping tests early
        if SKIP_POSTGRES_TESTS:
             cls.skipTest = lambda self, reason: super(PostgresTestCase, self).skipTest(reason) # Correct skip usage


    async def asyncSetUp(self):
        """Set up the test environment for PostgreSQL (Simplified)."""
        try:
            # 1. Reset Global State
            await DATABASES.shutdown()  # Ensure complete shutdown first
            DATABASES.CONFIG = {}
            DATABASES.CONNECTIONS = {}

            # 2. Configure Database
            db_alias_to_use = self.db_alias
            self.CONFIG = {
                db_alias_to_use: {
                    "ENGINE": "asyncpg", "HOST": POSTGRES_CONFIG["HOST"], "PORT": POSTGRES_CONFIG["PORT"],
                    "USER": POSTGRES_CONFIG["USER"], "PASSWORD": POSTGRES_CONFIG["PASSWORD"],
                    "NAME": POSTGRES_CONFIG["NAME"], "SETTINGS": {},
                    "MIN_CONNECTIONS": 1, "MAX_CONNECTIONS": 5,
                }
            }
            DATABASES.apply_config(self.CONFIG)

            # 3. Set Router
            self.router = use_postgres_database(db_alias_to_use)  # Route to the specific alias used

            test_app_label = self.app_label

            db_info = await DATABASES.get_connection(db_alias_to_use)
            if not isinstance(db_info, PostgreSQLDatabaseInfo):
                raise TypeError(f"Expected PostgreSQLDatabaseInfo for alias '{db_alias_to_use}', got {type(db_info)}")

            schema_query = f'CREATE SCHEMA IF NOT EXISTS "{test_app_label}"'
            await db_info.execute(schema_query)

            # 5. Bootstrap (creates tables)
            self.migration_manager = MigrationManager()
            # Bootstrap now returns alias_used, connection_info tuple, or None, None
            _, connection_info = await self.migration_manager.bootstrap(
                app_label=test_app_label,
                models=self.models,
                db=db_alias_to_use,  # Ensure bootstrap uses the correct target DB alias
                test_mode=True,
            )
            if connection_info is None:
                # If bootstrap didn't run (e.g., no models), get connection info manually
                connection_info = await DATABASES.get_connection(db_alias_to_use)
                # Still raise error if it's not the right type
                if not isinstance(connection_info, PostgreSQLDatabaseInfo):
                    raise TypeError(f"Could not get PostgreSQLDatabaseInfo for alias '{db_alias_to_use}'")
            elif not isinstance(connection_info, PostgreSQLDatabaseInfo):
                # If bootstrap ran but returned wrong type
                raise TypeError(f"Bootstrap returned unexpected connection type: {type(connection_info)}")


            # 6. Verification (USING MODEL'S PARSING METHOD)
            all_verified = True
            # Use the connection_info obtained above
            for model_cls in self.models:
                # *** Use the Model's internal parsing logic ***
                try:
                    # Note: _get_parsed_table_name is implicitly called by model methods,
                    # but we call it here EXPLICITLY for verification.
                    # It directly uses get_app_label and get_table_name from the model.
                    schema, table_part = model_cls._get_parsed_table_name()
                except Exception as parse_err:
                    self.logger.error(f"Error calling {model_cls.__name__}._get_parsed_table_name: {parse_err}",
                                      exc_info=True)
                    all_verified = False
                    continue  # Skip check for this model if parsing fails

                check_query = "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = $1 AND table_name = $2)"
                try:
                    # Use connection_info from bootstrap/get_connection
                    result = await connection_info.fetch_one(check_query, [schema, table_part])
                    exists = result["exists"] if result else False
                    if not exists:
                        all_verified = False
                except Exception as verify_err:
                    self.logger.error(f"Error verifying table '{schema}.{table_part}': {verify_err}", exc_info=True)
                    all_verified = False

            if not all_verified:
                try:
                    tables_in_schema = await connection_info.fetch_all(
                        "SELECT table_name FROM information_schema.tables WHERE table_schema = $1", [test_app_label])
                    self.logger.error(
                        f"Existing tables found in schema '{test_app_label}': {[t['table_name'] for t in tables_in_schema]}")
                except Exception as list_err:
                    self.logger.error(f"Could not list tables in schema '{test_app_label}': {list_err}")
                raise RuntimeError("Table verification failed after bootstrap.")

            # 7. Setup Test Data (Optional)
            if hasattr(self, 'setup_test_data') and callable(self.setup_test_data):
                await self.setup_test_data()

        except Exception as setup_err:
            self.logger.error(f"asyncSetUp Error: {type(setup_err).__name__}: {setup_err}", exc_info=True)
            # Use skipTest method defined in setUpClass
            self.skipTest(f"PostgreSQL setup failed: {setup_err}")


    async def asyncTearDown(self):
        """Clean up test resources (Simplified)."""
        db_alias_to_use = getattr(self, 'db_alias', self.app_label)
        test_app_label = self.app_label


        db_info = DATABASES.CONNECTIONS.get(db_alias_to_use)

        if db_info and isinstance(db_info, PostgreSQLDatabaseInfo):
             try:
                 if test_app_label != 'public':
                      schema_to_drop = test_app_label
                      drop_schema_sql = f'DROP SCHEMA IF EXISTS "{schema_to_drop}" CASCADE'
                      await db_info.execute(drop_schema_sql)
                 else:
                      self.logger.warning("Schema is 'public', attempting to drop individual tables.")
                      tables_to_drop_info = []
                      if hasattr(self, 'models') and self.models:
                           for model_cls in self.models:
                                schema, table_part = self._parse_table_name_for_verification(model_cls)
                                if schema == 'public':
                                     tables_to_drop_info.append({'schema': schema, 'table': table_part})

                      for info in tables_to_drop_info:
                           table_part = info['table']
                           full_quoted_name = f'"public"."{table_part}"'
                           drop_sql = f'DROP TABLE IF EXISTS {full_quoted_name} CASCADE'
                           await db_info.execute(drop_sql)

             except Exception as drop_err:
                  self.logger.error(f"Error during explicit drop in teardown: {drop_err}", exc_info=True)
        else:
            self.logger.warning(f"No active PostgreSQL connection info found for alias '{db_alias_to_use}' during teardown drop phase.")

        try:
            await DATABASES.shutdown([db_alias_to_use])
        except Exception as shutdown_err:
            self.logger.error(f"Error during DATABASES.shutdown for '{db_alias_to_use}': {shutdown_err}", exc_info=True)
            if 'Event loop is closed' in str(shutdown_err):
                 self.logger.warning("Shutdown error likely due to event loop already closing.")

    async def setup_test_data(self):
        """Placeholder for test data setup."""
        pass