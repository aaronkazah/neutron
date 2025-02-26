# core/db/editor.py
import json
import logging
import datetime
from enum import Enum
from typing import Optional, List, Tuple, Any

from core.db.fields import (
    BooleanField,
    VectorField,
    JSONField,
    CharField,
    TextField,
    IntegerField,
    DateTimeField,
    EnumField,
    BaseField,
    FloatField,
    BinaryField,
)
from core.db.base import DatabaseInfo, DatabaseType


class BaseSchemaEditor:
    """Base schema editor with common functionality"""

    def __init__(self, database: DatabaseInfo):
        self.connection = database
        self.database_type = database.db_type
        self.logger = logging.getLogger("SchemaEditor")

    async def execute(self, sql: str, params: Optional[dict] = None):
        self.logger.info("Executing: %s with params %s", sql, params)
        try:
            await self.connection.execute(query=sql, values=params)
        except Exception as e:
            self.logger.error(f"Error executing SQL: {sql} - {e}")
            raise

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self

    def _process_default_value(self, default) -> str:
        """Process default value for SQL."""
        if callable(default):
            default = default()

        if isinstance(default, datetime.datetime):
            return f"'{default.strftime('%Y-%m-%d %H:%M:%S.%f')}'"

        elif isinstance(default, Enum):
            return f"'{default.value}'"

        elif isinstance(default, (dict, list)):
            json_str = json.dumps(default).replace("'", "''")
            return f"'{json_str}'"

        elif isinstance(default, bool):
            return "1" if default else "0"

        elif isinstance(default, str):
            escaped_str = str(default).replace("'", "''")
            return f"'{escaped_str}'"

        return str(default)


class SQLiteSchemaEditor(BaseSchemaEditor):
    async def table_exists(self, model_name: str) -> bool:
        query = """
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        """
        result = await self.connection.fetch_one(query=query, values=[model_name])
        return bool(result)

    async def rename_column(self, model_name: str, old_name: str, new_name: str):
        """Rename column in SQLite (requires table rebuild)"""
        # Get current table info
        columns = await self.get_column_info(model_name)

        # Create new table with renamed column
        new_table = f"{model_name}_new"
        new_fields = []

        for col in columns:
            # Create the field without await since _create_field_from_column is not async
            field = self._create_field_from_column(col)
            field_name = new_name if col["name"] == old_name else col["name"]
            new_fields.append((field_name, field))

        await self.create_table(new_table, new_fields)

        # Copy data
        columns_str_old = ", ".join(
            f"{col['name']}" if col["name"] != old_name else f"{old_name}"
            for col in columns
        )
        columns_str_new = ", ".join(
            f"{col['name']}" if col["name"] != old_name else f"{new_name}"
            for col in columns
        )

        await self.execute(
            f"INSERT INTO {new_table} ({columns_str_new}) SELECT {columns_str_old} FROM {model_name}"
        )

        # Replace old table
        await self.drop_table(model_name)
        await self.rename_table(new_table, model_name)

    async def all_tables(self):
        """Get all tables in the database."""
        cursor = await self.connection.execute("SELECT name FROM sqlite_master")
        tables = await cursor.fetchall()
        return [t[0] for t in tables]

    def _create_field_from_column(self, column_info: dict) -> BaseField:
        """Create a field instance from column information"""
        field_type = column_info["type"].upper()

        field_kwargs = {
            "null": not column_info["notnull"],
            "default": column_info["default"],
            "primary_key": column_info["is_primary"],
        }

        # Handle VARCHAR with max_length
        if "VARCHAR" in field_type:
            import re

            match = re.match(r"VARCHAR\((\d+)\)", field_type)
            if match:
                field_kwargs["max_length"] = int(match.group(1))
            return CharField(**field_kwargs)

        # Map other field types
        type_mapping = {
            "TEXT": TextField,
            "INTEGER": IntegerField,
            "BLOB": VectorField,
            "BOOLEAN": BooleanField,
        }

        # Get the field class or default to CharField
        field_class = next(
            (
                field_cls
                for type_prefix, field_cls in type_mapping.items()
                if type_prefix in field_type
            ),
            CharField,
        )

        return field_class(**field_kwargs)

    async def alter_column(self, model_name: str, field_name: str, field: Any):
        """Alter column in SQLite (requires table rebuild)"""
        all_tables = await self.all_tables()
        # Get current table info
        columns = await self.get_column_info(model_name)

        # Create new table
        new_table = f"{model_name}_new"
        new_fields = []

        for col in columns:
            if col["name"] == field_name:
                new_fields.append((field_name, field))
            else:
                # Create field without await
                old_field = self._create_field_from_column(col)
                new_fields.append((col["name"], old_field))

        await self.create_table(new_table, new_fields)

        # Copy data
        columns_str = ", ".join(col["name"] for col in columns)
        await self.execute(
            f"INSERT INTO {new_table} ({columns_str}) SELECT {columns_str} FROM {model_name}"
        )

        # Replace old table
        await self.drop_table(model_name)
        await self.rename_table(new_table, model_name)

    async def get_column_info(self, model_name: str) -> List[dict]:
        query = f"PRAGMA table_info({model_name})"
        result = await self.connection.fetch_all(query=query)

        return [
            {
                "name": col[1],
                "type": col[2],
                "notnull": bool(col[3]),
                "default": col[4],
                "is_primary": bool(col[5]),
            }
            for col in result
        ]

    def get_column_type(self, field) -> str:
        type_mapping = {
            CharField: lambda f: (
                f"VARCHAR({f.max_length})"
                if hasattr(f, "max_length") and f.max_length
                else "VARCHAR"
            ),
            TextField: "TEXT",
            IntegerField: "INTEGER",
            BooleanField: "INTEGER",
            DateTimeField: "TEXT",
            JSONField: "TEXT",
            VectorField: "BLOB",
            BinaryField: "BLOB",
            EnumField: "TEXT",
            FloatField: "REAL",
        }

        field_type = type(field)
        column_type = type_mapping.get(field_type)

        if not column_type:
            raise ValueError(f"Unsupported field type: {field_type}")

        if callable(column_type):
            return column_type(field)

        return column_type

    async def create_table(self, table_name: str, fields: List[Tuple[str, Any]]):
        """Create a new table in SQLite"""
        if await self.table_exists(table_name):
            return

        field_definitions = []
        for field_name, field in fields:
            column_type = self.get_column_type(field)

            # Build column definition
            parts = [f"{field_name} {column_type}"]

            if field.primary_key:
                parts.append("PRIMARY KEY")
            if not field.null:
                parts.append("NOT NULL")
            if field.unique:
                parts.append("UNIQUE")
            if field.default is not None:
                parts.append(f"DEFAULT {self._process_default_value(field.default)}")

            field_definitions.append(" ".join(parts))

        create_table_sql = f"""
            CREATE TABLE {table_name} (
                {", ".join(field_definitions)}
            )
        """
        await self.execute(create_table_sql)

    async def drop_table(self, model_name: str):
        """Drop table in SQLite"""
        await self.execute(f"DROP TABLE IF EXISTS {model_name}")

    async def column_exists(self, model_name: str, column_name: str) -> bool:
        """Check if a column exists in the table"""
        columns = await self.get_column_info(model_name)
        return any(col["name"] == column_name for col in columns)

    async def remove_column(self, model_name: str, field_name: str):
        """Remove column in SQLite (requires table rebuild)"""
        # Get current table info
        columns = await self.get_column_info(model_name)

        # Create new table without the column
        new_table = f"{model_name}_new"
        new_fields = [
            (col["name"], self._create_field_from_column(col))
            for col in columns
            if col["name"] != field_name
        ]

        await self.create_table(new_table, new_fields)

        # Copy data - only include columns that exist in both tables
        remaining_columns = [col["name"] for col in columns if col["name"] != field_name]
        columns_str = ", ".join(remaining_columns)

        try:
            await self.execute(
                f"INSERT INTO {new_table} ({columns_str}) SELECT {columns_str} FROM {model_name}"
            )
        except Exception as e:
            # If something goes wrong, clean up the new table
            await self.drop_table(new_table)
            raise e

        # Replace old table
        await self.drop_table(model_name)
        await self.rename_table(new_table, model_name)

    async def add_column(self, model_name: str, field_name: str, field: Any):
        """Add column in SQLite"""
        # First check if column exists
        if await self.column_exists(model_name, field_name):
            return  # Skip if column already exists

        # Get the table's current columns
        columns = await self.get_column_info(model_name)

        # Create new table with all columns including the new one
        new_table = f"{model_name}_new"
        new_fields = [
            (col["name"], self._create_field_from_column(col))
            for col in columns
        ]
        # Add the new field
        new_fields.append((field_name, field))

        await self.create_table(new_table, new_fields)

        # Copy existing data
        existing_columns = ", ".join(col["name"] for col in columns)
        try:
            await self.execute(
                f"INSERT INTO {new_table} ({existing_columns}) SELECT {existing_columns} FROM {model_name}"
            )

            # If the new column has a default value, update it
            if field.default is not None:
                default_value = self._process_default_value(field.default)
                await self.execute(
                    f"UPDATE {new_table} SET {field_name} = {default_value}"
                )
        except Exception as e:
            # If something goes wrong, clean up the new table
            await self.drop_table(new_table)
            raise e

        # Replace old table
        await self.drop_table(model_name)
        await self.rename_table(new_table, model_name)

    async def rename_table(self, old_name: str, new_name: str):
        """Rename table in SQLite"""
        await self.execute(f"ALTER TABLE {old_name} RENAME TO {new_name}")


def get_schema_editor(database: DatabaseInfo) -> BaseSchemaEditor:
    """Get appropriate schema editor based on database type"""
    return SQLiteSchemaEditor(database)
