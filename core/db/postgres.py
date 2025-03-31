# core/db/postgres.py

import asyncio
import re

import numpy as np
import datetime
import json
import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
import asyncpg

# Assuming these are correctly defined elsewhere
from core.db.base import DatabaseType, DatabaseInfo
from core.db.fields import (
    BaseField, CharField, TextField, IntegerField, DateTimeField,
    JSONField, VectorField, BooleanField, EnumField, FloatField, BinaryField
)
from core.db.router import SingleDatabaseRouter

# --- End logging config ---


class PostgreSQLDatabaseInfo(DatabaseInfo):
    """Stores PostgreSQL database connection POOL and related information."""

    def __init__(
            self,
            name: str,
            pool: asyncpg.Pool,  # Only accept pool
            config: Dict,
    ):
        # Initialize base class - connection is None for pool-based management
        super().__init__(name=name, connection=None, db_type=DatabaseType.POSTGRES, config=config)
        self.pool = pool
        self.logger = logging.getLogger(f"PostgreSQLDatabaseInfo.{name}")

    def _convert_query_params(self, query: str, values: Optional[List] = None) -> str:
        """Convert ? style parameters to PostgreSQL $1, $2, etc. style,
           handling the JSONB '?' operator specifically."""
        if not values or '?' not in query:
            return query

        param_count = 0
        converted_query = ''
        i = 0
        n = len(query)
        while i < n:
            char = query[i]
            if char == '?':
                # Look behind for '::jsonb ?' pattern, allowing for spaces
                prefix_idx = max(0, i - 8)  # Approx index start
                substring = query[prefix_idx:i + 1].strip()
                is_operator = substring.endswith('::jsonb ?')

                if is_operator:
                    # It's the operator, keep the '?'
                    converted_query += '?'
                else:
                    # It's a value placeholder, convert it
                    param_count += 1
                    converted_query += f'${param_count}'
            else:
                converted_query += char
            i += 1

        # --- Verification ---
        final_placeholder_count = len(re.findall(r'\$\d+', converted_query))
        if final_placeholder_count != len(values):
            self.logger.error(f"Placeholder/Value mismatch ERROR!")
            self.logger.error(f"  Original Query: {query}")
            self.logger.error(f"  Converted Query: {converted_query}")
            self.logger.error(f"  Values: {values} (Count: {len(values)})")
            self.logger.error(f"  Placeholders generated: {final_placeholder_count}")
            raise ValueError(
                f"Internal Error: Query placeholder count ({final_placeholder_count}) != value count ({len(values)}) after conversion.")
        # --- End Verification ---

        return converted_query

    def __repr__(self):
        cfg_summary = {k: v for k, v in self.config.items() if k not in ['PASSWORD']}
        pool_status = "Active" if self.pool and not self.pool._closed else "Closed/None"
        return f"PostgreSQLDatabaseInfo(name={self.name}, pool_status='{pool_status}', config={cfg_summary})"

    def __str__(self):
        return f"PostgreSQLDatabaseInfo(name={self.name})"

    async def executemany(self, query: str, values: List[List[Any]]) -> None:
        """Execute many queries using prepared statement."""
        if not self.pool: raise RuntimeError(f"Connection pool for '{self.name}' is closed or not initialized.")
        if not values: return  # Nothing to execute

        # Assume QuerySet provides PG-style ($1, $2) query for bulk operations
        # If '?' is present, log warning but proceed - rely on upstream formatting.
        converted_query = query
        if '?' in query:
            self.logger.warning(
                "Executemany received '?' placeholders for PG. Assuming upstream formatted correctly. Query may fail.")
            # If conversion is strictly needed here based on first row:
            # converted_query = self._convert_query_params(query, values[0])

        # Convert values in each sublist
        pg_values = []
        for row_values in values:
            pg_values.append([self._convert_field_value(v) for v in row_values])

        try:
            async with self.pool.acquire() as conn:
                await conn.executemany(converted_query, pg_values)
        except Exception as e:
            self.logger.error(f"Error in executemany: {e}", exc_info=True)
            self.logger.error(f"Query: {converted_query}")
            raise

    def _convert_field_value(self, value, field_name=None):
        """Convert Python value to DB type for asyncpg binding."""
        if value is None: return None
        # Let asyncpg handle basic types
        if isinstance(value, (bool, int, float, str, bytes, bytearray, datetime.datetime)):
            return value
        # Explicit conversions
        elif isinstance(value, (dict, list)):
            try:
                return json.dumps(value)
            except TypeError as e:
                self.logger.warning(f"JSON serialization failed: {e}. Value: {value}", exc_info=True); raise
        elif isinstance(value, Enum):
            return value.value
        elif isinstance(value, np.ndarray):
            return value.astype(np.float32).tobytes()
        # Fallback
        try:
            s_val = str(value)
            self.logger.warning(
                f"Converting unknown type {type(value)} to string '{s_val}' for DB bind (field: {field_name}).")
            return s_val
        except Exception as str_e:
            self.logger.error(f"Failed to convert type {type(value)} to string: {str_e}", exc_info=True)
            raise TypeError(f"Unsupported type for DB binding: {type(value)}") from str_e

    async def execute(self, query: str, values: Optional[List] = None) -> Any:
        """Execute a query (INSERT, UPDATE, DELETE, DDL)."""
        if not self.pool: raise RuntimeError(f"Connection pool for '{self.name}' is closed or not initialized.")
        final_query = self._convert_query_params(query, values)
        pg_values = [self._convert_field_value(v) for v in values] if values else []

        try:
            async with self.pool.acquire() as conn:
                # Returns status string (e.g., 'INSERT 1')
                return await conn.execute(final_query, *pg_values)
        except Exception as e:
            self.logger.error(f"Error executing query: {query} with values {values}")
            self.logger.error(f"Converted query: {final_query}; Converted values: {pg_values}")
            self.logger.error(f"Error details: {type(e).__name__}: {str(e)}", exc_info=True)
            raise

    async def fetch_one(self, query: str, values: Optional[List] = None) -> Optional[Dict]:
        """Fetch one row."""
        if not self.pool: raise RuntimeError(f"Connection pool for '{self.name}' is closed or not initialized.")
        final_query = self._convert_query_params(query, values)
        pg_values = [self._convert_field_value(v) for v in values] if values else []

        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchrow(final_query, *pg_values)
                return dict(result) if result else None
        except Exception as e:
            self.logger.error(f"Error fetching one: {query} with values {values}")
            self.logger.error(f"Converted query: {final_query}; Converted values: {pg_values}")
            self.logger.error(f"Error details: {type(e).__name__}: {str(e)}", exc_info=True)
            raise

    async def fetch_all(self, query: str, values: Optional[List] = None) -> List[Dict]:
        """Fetch all rows."""
        if not self.pool: raise RuntimeError(f"Connection pool for '{self.name}' is closed or not initialized.")
        final_query = self._convert_query_params(query, values)
        pg_values = [self._convert_field_value(v) for v in values] if values else []

        try:
            async with self.pool.acquire() as conn:
                results = await conn.fetch(final_query, *pg_values)
                return [dict(row) for row in results]
        except Exception as e:
            self.logger.error(f"Error fetching all: {query} with values {values}")
            self.logger.error(f"Converted query: {final_query}; Converted values: {pg_values}")
            self.logger.error(f"Error details: {type(e).__name__}: {str(e)}", exc_info=True)
            raise

    async def commit(self):
        self.logger.debug("Commit called (implicit)")

    async def rollback(self):
        self.logger.debug("Rollback called (implicit on error)")

    async def close(self):
        """Gracefully close the connection pool."""
        # No initial connection to release anymore
        if self.pool and not self.pool._closed:
            try:
                await self.pool.close()
            except Exception as close_err:
                self.logger.error(f"Error closing connection pool for {self.name}: {close_err}", exc_info=True)
            finally:
                self.pool = None


class PostgresConnectionManager:
    """Manages PostgreSQL connections."""

    @staticmethod
    async def create_connection(config: Dict) -> asyncpg.Pool:  # Return only the pool
        """Create PostgreSQL connection pool."""
        host = config.get("HOST", "localhost");
        port = config.get("PORT", 5432)
        user = config.get("USER", "postgres");
        password = config.get("PASSWORD", "")
        database = config.get("NAME", "postgres")
        min_size = config.get("MIN_CONNECTIONS", 1);
        max_size = config.get("MAX_CONNECTIONS", 10)
        logger = logging.getLogger("PostgresConnectionManager")
        pool = None
        try:
            pool = await asyncpg.create_pool(user=user, password=password, database=database, host=host, port=port,
                                             min_size=min_size, max_size=max_size)
            # Test connection and apply settings within the context
            async with pool.acquire() as conn_test:
                if "SETTINGS" in config: await PostgresConnectionManager.apply_settings(conn_test, config["SETTINGS"])
            return pool  # Return the created pool
        except Exception as e:
            logger.error(f"Failed pool creation/connection: {e}", exc_info=True)
            if pool and not pool._closed: await pool.close()  # Cleanup pool if created
            raise

    @staticmethod
    async def apply_settings(connection: asyncpg.Connection, settings: Dict):
        logger = logging.getLogger("PostgresConnectionManager")
        if settings:
            for setting, value in settings.items():
                try:
                    await connection.execute(f"SET {setting} = $1", value)
                except Exception as set_err:
                    logger.warning(f"Could not apply setting {setting}={value}: {set_err}")


class PostgresSchemaEditor:
    """Schema editor for PostgreSQL databases."""

    def __init__(self, database: PostgreSQLDatabaseInfo):
        self.database = database
        db_alias = getattr(self.database, 'name', 'unknown_pg_db')
        self.logger = logging.getLogger(f"PostgresSchemaEditor.{db_alias}")
        if not getattr(self.database, 'logger', None):
            self.database.logger = logging.getLogger(f"PostgreSQLDatabaseInfo.{db_alias}")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

    def _quote_name(self, name: str) -> str:
        if not isinstance(name, str): self.logger.warning(f"_quote_name received non-string: {type(name)}"); return name
        if name.startswith('"') and name.endswith('"'): return name
        return f'"{name.replace("\"", "\"\"")}"'

    def map_field_type(self, field: BaseField) -> str:
        if isinstance(field, CharField): return f"VARCHAR({field.max_length})" if field.max_length else "TEXT"
        if isinstance(field, TextField): return "TEXT"
        if isinstance(field, IntegerField): return "INTEGER"
        if isinstance(field, BooleanField): return "BOOLEAN"
        if isinstance(field, DateTimeField): return "TIMESTAMP WITH TIME ZONE"
        if isinstance(field, JSONField): return "JSONB"
        if isinstance(field, VectorField): return "BYTEA"
        if isinstance(field, BinaryField): return "BYTEA"
        if isinstance(field, EnumField): return "TEXT"
        if isinstance(field, FloatField): return "REAL"
        raise ValueError(f"Unsupported field type for PostgreSQL: {type(field)}")

    def _process_default_value(self, field, default):
        if default is None: return None
        is_callable = callable(default)
        original_callable = default if is_callable else None
        if is_callable:
            field_default = getattr(field, 'default', None)
            if isinstance(field, DateTimeField) and field_default == datetime.datetime.now: return "CURRENT_TIMESTAMP"
            try:
                default = default(); self.logger.warning(f"Callable default {original_callable} evaluated.")
            except Exception as e:
                self.logger.error(f"Could not eval default {original_callable}: {e}"); return None
        if isinstance(default, bool): return str(default).lower()
        if isinstance(default, (int, float)): return str(default)
        if isinstance(default, str): return f"'{default.replace("'", "''")}'"
        if isinstance(default, datetime.datetime): return f"'{default.isoformat()}'::timestamptz"
        if isinstance(default, (dict, list)) and isinstance(field,
                                                            JSONField): return f"'{json.dumps(default).replace("'", "''")}'::jsonb"
        if isinstance(default, Enum) and isinstance(field,
                                                    EnumField): return f"'{str(default.value).replace("'", "''")}'"
        self.logger.warning(f"Default value type {type(default)} format unknown.")
        return f"'{str(default).replace("'", "''")}'"

    async def create_table(self, app_label: str, table_base_name: str, fields: List[Tuple[str, BaseField]]):
        """Create table using app_label as schema and table_base_name as table name."""
        schema, table = app_label, table_base_name  # Use directly
        quoted_schema, quoted_table = self._quote_name(schema), self._quote_name(table)
        full_quoted_name = f"{quoted_schema}.{quoted_table}"

        try:
            exists = await self.table_exists(schema, table)  # Use unquoted for check
            if exists: self.logger.warning(f"Table {full_quoted_name} already exists, skipping."); return
        except Exception as check_err:
            raise RuntimeError(f"Failed check table existence {schema}.{table}") from check_err

        field_defs, pks, constraints = [], [], []
        try:
            for name, field in fields:
                if not isinstance(field, BaseField): raise TypeError(f"Field '{name}' must be BaseField")
                col_type = self.map_field_type(field);
                q_name = self._quote_name(name)
                defn = [f"{q_name} {col_type}"]
                if not field.null: defn.append("NOT NULL")
                if field.default is not None:
                    def_val = self._process_default_value(field, field.default)
                    if def_val is not None: defn.append(f"DEFAULT {def_val}")
                field_defs.append(" ".join(defn))
                if field.primary_key: pks.append(q_name)
                if field.unique: constraints.append(f"UNIQUE ({q_name})")
            if pks: field_defs.append(f"PRIMARY KEY ({', '.join(pks)})")
            field_defs.extend(constraints)
        except Exception as build_err:
            raise ValueError("Failed build field defs") from build_err

        schema_sql = f'CREATE SCHEMA IF NOT EXISTS {quoted_schema}'
        table_sql = f"CREATE TABLE {full_quoted_name} ({', '.join(field_defs)})"
        try:
            await self.database.execute(schema_sql);
            await self.database.execute(table_sql);
        except Exception as e:
            self.logger.error(f"Error creating table {full_quoted_name}: {e}", exc_info=True)
            self.logger.error(f"Schema SQL: {schema_sql}");
            self.logger.error(f"Table SQL: {table_sql}")
            raise

    async def table_exists(self, schema: str, table: str) -> bool:
        """Check if table exists in a specific schema."""
        q = "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = $1 AND table_name = $2)"
        q_schema, q_table = schema.replace('"', ''), table.replace('"', '')
        try:
            r = await self.database.fetch_one(q, [q_schema, q_table]); exists = r[
                "exists"] if r else False; return exists
        except Exception as e:
            self.logger.error(f"Error check table exists {q_schema}.{q_table}: {e}"); raise

    async def column_exists(self, schema: str, table: str, column_name: str) -> bool:
        """Check if column exists in a specific schema and table."""
        q = "SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = $1 AND table_name = $2 AND column_name = $3)"
        q_schema, q_table, q_col = schema.replace('"', ''), table.replace('"', ''), column_name.replace('"', '')
        try:
            r = await self.database.fetch_one(q, [q_schema, q_table, q_col]); exists = r[
                "exists"] if r else False; return exists
        except Exception as e:
            self.logger.error(f"Error check column exists {q_col} in {q_schema}.{q_table}: {e}"); raise

    async def drop_table(self, app_label: str, table_base_name: str):
        """Drop table using app_label as schema."""
        schema, table = app_label, table_base_name;
        name = f"{self._quote_name(schema)}.{self._quote_name(table)}"
        sql = f'DROP TABLE IF EXISTS {name} CASCADE';
        try:
            await self.database.execute(sql)
        except Exception as e:
            self.logger.error(f"Error drop table {name}: {e}"); raise

    async def rename_table(self, old_app_label: str, old_table_base_name: str, new_app_label: str,
                           new_table_base_name: str):
        """Rename table, potentially changing schema."""
        o_schema, o_table = old_app_label, old_table_base_name;
        n_schema, n_table = new_app_label, new_table_base_name
        qo_schema, qo_table = self._quote_name(o_schema), self._quote_name(o_table)
        qn_schema, qn_table = self._quote_name(n_schema), self._quote_name(n_table)
        full_old, full_new = f"{qo_schema}.{qo_table}", f"{qn_schema}.{qn_table}"
        try:
            if o_schema != n_schema:
                sql1 = f'ALTER TABLE {full_old} SET SCHEMA {qn_schema}';
                await self.database.execute(sql1)
                interim = f"{qn_schema}.{qo_table}"
                sql2 = f'ALTER TABLE {interim} RENAME TO {qn_table}';
                await self.database.execute(sql2)
            else:
                sql = f'ALTER TABLE {full_old} RENAME TO {qn_table}';
                await self.database.execute(sql)
        except Exception as e:
            self.logger.error(f"Error rename table {full_old} to {full_new}: {e}"); raise

    async def add_column(self, app_label: str, table_base_name: str, field_name: str, field: BaseField):
        schema, table = app_label, table_base_name;
        name = f"{self._quote_name(schema)}.{self._quote_name(table)}";
        q_field = self._quote_name(field_name)
        if await self.column_exists(schema, table, field_name): self.logger.warning(
            f"Column {q_field} exists in {name}, skip add."); return
        col_type = self.map_field_type(field);
        defn = [f"ADD COLUMN {q_field} {col_type}"]
        if not field.null: defn.append("NOT NULL")
        if field.unique: defn.append("UNIQUE")
        def_val = self._process_default_value(field, field.default)
        if def_val is not None: defn.append(f"DEFAULT {def_val}")
        sql = f'ALTER TABLE {name} {" ".join(defn)}';
        try:
            await self.database.execute(sql)
        except Exception as e:
            self.logger.error(f"Error add column {q_field} to {name}: {e}"); raise

    async def remove_column(self, app_label: str, table_base_name: str, field_name: str):
        schema, table = app_label, table_base_name;
        name = f"{self._quote_name(schema)}.{self._quote_name(table)}";
        q_field = self._quote_name(field_name)
        if not await self.column_exists(schema, table, field_name): self.logger.warning(
            f"Column {q_field} not in {name}, skip remove."); return
        sql = f'ALTER TABLE {name} DROP COLUMN {q_field} CASCADE';
        try:
            await self.database.execute(sql)
        except Exception as e:
            self.logger.error(f"Error remove column {q_field} from {name}: {e}"); raise

    async def rename_column(self, app_label: str, table_base_name: str, old_name: str, new_name: str):
        schema, table = app_label, table_base_name;
        name = f"{self._quote_name(schema)}.{self._quote_name(table)}";
        q_old = self._quote_name(old_name);
        q_new = self._quote_name(new_name)
        if not await self.column_exists(schema, table, old_name): self.logger.warning(
            f"Column {q_old} not in {name}, skip rename."); return
        if await self.column_exists(schema, table, new_name): raise ValueError(f"Target column '{new_name}' exists.")
        sql = f'ALTER TABLE {name} RENAME COLUMN {q_old} TO {q_new}';
        try:
            await self.database.execute(sql)
        except Exception as e:
            self.logger.error(f"Error rename column {q_old} to {q_new}: {e}"); raise

    async def alter_column(self, app_label: str, table_base_name: str, field_name: str, field: BaseField):
        schema, table = app_label, table_base_name;
        name = f"{self._quote_name(schema)}.{self._quote_name(table)}";
        q_field = self._quote_name(field_name)
        if not await self.column_exists(schema, table, field_name): raise ValueError(
            f"Column {field_name} not in {schema}.{table}")
        actions = [];
        col_type = self.map_field_type(field);
        actions.append(f'ALTER COLUMN {q_field} TYPE {col_type}')  # Add USING if needed
        def_val = self._process_default_value(field, field.default)
        if def_val is not None:
            actions.append(f'ALTER COLUMN {q_field} SET DEFAULT {def_val}')
        else:
            actions.append(f'ALTER COLUMN {q_field} DROP DEFAULT')
        if field.null:
            actions.append(f'ALTER COLUMN {q_field} DROP NOT NULL')
        else:
            actions.append(f'ALTER COLUMN {q_field} SET NOT NULL')
        sql = f'ALTER TABLE {name} {", ".join(actions)}';
        try:
            await self.database.execute(sql)
        except Exception as e:
            self.logger.error(f"Error alter column {q_field}: {e}"); self.logger.error(f"Actions: {actions}"); raise

    async def get_column_info(self, schema: str, table: str) -> List[Dict]:
        """Get column information for a specific schema and table."""
        query = "SELECT column_name AS name, data_type AS type FROM information_schema.columns WHERE table_schema = $1 AND table_name = $2 ORDER BY ordinal_position"
        q_schema, q_table = schema.replace('"', ''), table.replace('"', '')
        try:
            result = await self.database.fetch_all(query, [q_schema, q_table]); return result
        except Exception as e:
            self.logger.error(f"Error get column info {q_schema}.{q_table}: {e}"); raise


class PostgresRouter(SingleDatabaseRouter):
    """Router for PostgreSQL that uses separate schemas for apps."""

    def __init__(self, default_alias: str = "default"):
        super().__init__(default_alias)
