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
from core.db.base import DatabaseInfo
from core.db.postgres import PostgresSchemaEditor


class BaseSchemaEditor:
    """Base schema editor with common functionality"""

    def __init__(self, database: DatabaseInfo):
        self.connection = database
        self.database_type = database.db_type
        self.logger = logging.getLogger("SchemaEditor")
        self.logger.setLevel(logging.ERROR)

    async def execute(self, sql: str, params: Optional[dict] = None):
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
    async def table_exists(self, app_label_or_table_name: str, table_base_name: Optional[str] = None) -> bool:
        """Check if a table exists in SQLite."""
        # SQLite doesn't use schemas, combine names if both provided
        if table_base_name:
            table_name_to_check = f"{app_label_or_table_name}_{table_base_name}"
        else:
            table_name_to_check = app_label_or_table_name

        # The SQL query uses a placeholder '?'
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"

        try:
            # Use fetch_one which returns None if not found
            # --- FIX: Pass the UNQUOTED table name as the parameter value ---
            result = await self.connection.fetch_one(query=query, values=[table_name_to_check])
            return result is not None
        except Exception as e:
            self.logger.error(f"Error checking table existence for '{table_name_to_check}': {e}", exc_info=True)
            # If an error occurs (e.g., connection closed), assume it doesn't exist or re-raise
            return False  # Or re-raise depending on desired strictness

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

    async def alter_column(self, app_label: str, table_base_name: str, field_name: str, field: Any):
        """Alter column in SQLite (requires table rebuild)"""
        # Construct the full table name from app_label and table_base_name
        table_name = f"{app_label}_{table_base_name}"
        quoted_table_name = f'"{table_name}"' # Quote the name

        self.logger.debug(f"Altering column '{field_name}' in table '{table_name}'")
        # await self.all_tables() # Usually not needed here

        # Get current table info using the constructed table_name
        try:
            columns = await self.get_column_info(table_name)
            if not columns:
                self.logger.error(f"Cannot alter column: Table '{table_name}' not found or has no columns.")
                raise ValueError(f"Table '{table_name}' not found for altering column.")
        except Exception as e:
            self.logger.error(f"Error getting column info for '{table_name}' during alter_column: {e}", exc_info=True)
            raise

        # Create new table definition
        new_table_name_temp = f"{table_name}_alter_temp"
        quoted_new_table_name_temp = f'"{new_table_name_temp}"'
        new_fields = []
        found_column_to_alter = False

        for col in columns:
            current_col_name = col["name"]
            if current_col_name == field_name:
                # Use the new field definition for the altered column
                new_fields.append((field_name, field))
                found_column_to_alter = True
            else:
                # Keep the old field definition for other columns
                # Create field instance from existing column info
                old_field = self._create_field_from_column(col) # Ensure this helper works correctly
                new_fields.append((current_col_name, old_field))

        if not found_column_to_alter:
            self.logger.error(f"Column '{field_name}' not found in table '{table_name}' during alter operation.")
            # Depending on desired behavior, you might raise an error or just log and return
            raise ValueError(f"Column '{field_name}' to alter not found in table '{table_name}'.")


        # --- Rebuild Table Logic ---
        try:
            # 1. Create the new table with the altered definition
            # Pass app_label and the temporary base name to create_table
            temp_base_name = f"{table_base_name}_alter_temp"
            await self.create_table(app_label, temp_base_name, new_fields)

            # 2. Copy data from old table to new table
            # Ensure column names are quoted correctly
            columns_str = ", ".join(f'"{col["name"]}"' for col in columns)
            insert_sql = f"INSERT INTO {quoted_new_table_name_temp} ({columns_str}) SELECT {columns_str} FROM {quoted_table_name}"
            await self.execute(insert_sql)

            # 3. Drop the old table
            await self.drop_table(table_name) # Pass the original full table name

            # 4. Rename the new table to the original name
            # Use the rename_table method which expects app labels and base names
            await self.rename_table(app_label, temp_base_name, app_label, table_base_name)

            self.logger.debug(f"Successfully altered column '{field_name}' in table '{table_name}'.")

        except Exception as e:
            self.logger.error(f"Error during table rebuild for alter_column on '{table_name}': {e}", exc_info=True)
            # Attempt cleanup: drop the temporary table if it exists
            try:
                # Construct temp table name again for cleanup
                temp_full_name = f"{app_label}_{table_base_name}_alter_temp"
                await self.drop_table(temp_full_name)
            except Exception as cleanup_err:
                self.logger.error(f"Failed to drop temporary table '{temp_full_name}' after alter error: {cleanup_err}")
            raise # Re-raise the original error

    # --- FIX THE remove_column METHOD SIGNATURE AND USAGE ---
    async def remove_column(self, app_label: str, table_base_name: str, field_name: str):
        """Remove column in SQLite (requires table rebuild)"""
        # Construct the full table name
        table_name = f"{app_label}_{table_base_name}"
        quoted_table_name = f'"{table_name}"'

        self.logger.debug(f"Removing column '{field_name}' from table '{table_name}'")

        # Get current table info
        try:
            columns = await self.get_column_info(table_name)
            if not columns:
                 self.logger.error(f"Cannot remove column: Table '{table_name}' not found or has no columns.")
                 raise ValueError(f"Table '{table_name}' not found for removing column.")
        except Exception as e:
            self.logger.error(f"Error getting column info for '{table_name}' during remove_column: {e}", exc_info=True)
            raise

        # Check if the column to remove actually exists
        if not any(col["name"] == field_name for col in columns):
            self.logger.warning(f"Column '{field_name}' not found in table '{table_name}'. Skipping removal.")
            return # Nothing to do if column doesn't exist

        # --- Rebuild Table Logic ---
        new_table_name_temp = f"{table_name}_remove_temp"
        quoted_new_table_name_temp = f'"{new_table_name_temp}"'

        try:
            # 1. Create new table definition without the column
            new_fields = []
            remaining_columns_names = [] # Keep track of names for INSERT/SELECT
            for col in columns:
                if col["name"] != field_name:
                    field = self._create_field_from_column(col)
                    new_fields.append((col["name"], field))
                    remaining_columns_names.append(f'"{col["name"]}"') # Quote for safety

            if not new_fields:
                self.logger.error(f"Cannot remove column '{field_name}': Removing it would leave table '{table_name}' with no columns.")
                raise ValueError(f"Cannot remove last column '{field_name}' from table '{table_name}'.")

            # Create the new table
            temp_base_name = f"{table_base_name}_remove_temp"
            await self.create_table(app_label, temp_base_name, new_fields)

            # 2. Copy data - only include columns that exist in both tables
            columns_to_select_str = ", ".join(remaining_columns_names)
            insert_sql = f"INSERT INTO {quoted_new_table_name_temp} ({columns_to_select_str}) SELECT {columns_to_select_str} FROM {quoted_table_name}"
            await self.execute(insert_sql)

            # 3. Drop the old table
            await self.drop_table(table_name) # Pass original full name

            # 4. Rename the new table to the original name
            await self.rename_table(app_label, temp_base_name, app_label, table_base_name)

            self.logger.debug(f"Successfully removed column '{field_name}' from table '{table_name}'.")

        except Exception as e:
            self.logger.error(f"Error during table rebuild for remove_column on '{table_name}': {e}", exc_info=True)
            # Attempt cleanup
            try:
                temp_full_name = f"{app_label}_{table_base_name}_remove_temp"
                await self.drop_table(temp_full_name)
            except Exception as cleanup_err:
                self.logger.error(f"Failed to drop temporary table '{temp_full_name}' after remove error: {cleanup_err}")
            raise # Re-raise original error

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
        """Map Django field types to SQLite column types."""
        type_mapping = {
            CharField: "TEXT",  # Always use TEXT for CharField regardless of max_length
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

        # Handle callable mappings if needed in the future
        if callable(column_type):
            return column_type(field)

        return column_type

    # Inside class SQLiteSchemaEditor in core/db/editor.py

    async def create_table(self, app_label: str, table_base_name: str,
                           fields: List[Tuple[str, BaseField]]):  # Use BaseField hint
        """Create a new table in SQLite using the combined app_label_table_base_name format."""
        table_name = f"{app_label}_{table_base_name}"
        quoted_table_name = f'"{table_name}"'  # Quote combined name

        # Check existence using the combined name
        exists = await self.table_exists(table_name)
        if exists:
            self.logger.warning(f"SQLite table '{table_name}' already exists, skipping.")
            return

        field_definitions = []
        primary_keys = []
        has_explicit_pk = False

        for field_name, field in fields:
            if not isinstance(field, BaseField):
                # Log error and skip this field or raise error? Raising is safer.
                self.logger.error(f"Invalid field type for '{field_name}' in table '{table_name}': {type(field)}")
                raise TypeError(f"Invalid field type provided for '{field_name}': {type(field)}")

            column_type = self.get_column_type(field)
            quoted_field_name = f'"{field_name}"'  # Quote field name
            parts = [f"{quoted_field_name} {column_type}"]

            # Handle PK constraints separately
            if field.primary_key:
                primary_keys.append(quoted_field_name)
                has_explicit_pk = True
                # SQLite implicitly adds NOT NULL to PRIMARY KEY columns

            if not field.null and not field.primary_key:  # Don't add NOT NULL if PK
                parts.append("NOT NULL")
            if field.unique:
                # SQLite needs UNIQUE constraint defined separately or inline for single column
                parts.append("UNIQUE")  # Inline UNIQUE is fine for single column
            if field.default is not None:
                default_sql = self._process_default_value(field.default)
                if default_sql is not None:  # Check if processing yielded a value
                    parts.append(f"DEFAULT {default_sql}")

            field_definitions.append(" ".join(parts))

        # Add PRIMARY KEY constraint at the end if defined
        if primary_keys:
            # For SQLite, if it's a single integer PK, defining it inline is common for auto-increment behavior.
            # However, defining it at the end works for single or composite keys.
            field_definitions.append(f"PRIMARY KEY ({', '.join(primary_keys)})")
        elif not has_explicit_pk:
            # Let SQLite handle its implicit ROWID if no PK is specified.
            pass

        # Construct the final SQL
        create_table_sql = f"CREATE TABLE {quoted_table_name} ({', '.join(field_definitions)})"

        # --- Add Logging ---
        # --- End Logging ---

        try:
            await self.execute(create_table_sql)
        except Exception as e:
            self.logger.error(f"Failed to execute CREATE TABLE for '{table_name}'. SQL: {create_table_sql}",
                              exc_info=True)
            raise  # Re-raise the exception

    async def drop_table(self, model_name: str):
        """Drop table in SQLite"""
        await self.execute(f"DROP TABLE IF EXISTS {model_name}")

    async def column_exists(self, model_name: str, column_name: str) -> bool:
        """Check if a column exists in the table"""
        columns = await self.get_column_info(model_name)
        return any(col["name"] == column_name for col in columns)

    async def add_column(self, app_label: str, table_base_name: str, field_name: str, field: Any):
        """Add column in SQLite - simplified version that directly alters the table"""
        table_name = f"{app_label}_{table_base_name}"

        # Check if column already exists
        columns = await self.get_column_info(table_name)
        if any(col["name"] == field_name for col in columns):
            self.logger.info(f"Column '{field_name}' already exists in table '{table_name}', skipping.")
            return

        # Get column definition
        column_type = self.get_column_type(field)
        column_def = f"{field_name} {column_type}"

        # Add NOT NULL constraint if field is not nullable
        if not field.null:
            column_def += " NOT NULL"

        # Add DEFAULT clause if field has a default value
        if field.default is not None:
            default_value = self._process_default_value(field.default)
            column_def += f" DEFAULT {default_value}"

        # Execute ALTER TABLE directly
        alter_stmt = f'ALTER TABLE "{table_name}" ADD COLUMN {column_def}'
        try:
            await self.execute(alter_stmt)
        except Exception as e:
            self.logger.error(f"Error adding column '{field_name}' to '{table_name}': {e}")
            raise

    async def rename_table(self, old_app_label: str, old_table_base_name: str, new_app_label: str,
                           new_table_base_name: str):
        """Rename table with proper handling of app prefixes"""
        # Construct full table names with app prefixes
        old_table_name = f"{old_app_label}_{old_table_base_name}"
        new_table_name = f"{new_app_label}_{new_table_base_name}"


        # Verify old table exists
        old_exists = await self.table_exists(old_table_name)
        if not old_exists:
            self.logger.error(f"Cannot rename table '{old_table_name}' because it doesn't exist")
            raise ValueError(f"Table '{old_table_name}' does not exist")

        # Check if new table already exists
        new_exists = await self.table_exists(new_table_name)
        if new_exists:
            self.logger.error(f"Cannot rename to '{new_table_name}' because it already exists")
            raise ValueError(f"Target table '{new_table_name}' already exists")

        # Execute the rename
        try:
            await self.execute(f'ALTER TABLE "{old_table_name}" RENAME TO "{new_table_name}"')
        except Exception as e:
            self.logger.error(f"Error renaming table '{old_table_name}' to '{new_table_name}': {e}")
            raise


def get_schema_editor(database: Any) -> PostgresSchemaEditor | SQLiteSchemaEditor:
    """Get appropriate schema editor based on database type"""
    from core.db.base import DatabaseType

    if hasattr(database, 'db_type'):
        if database.db_type == DatabaseType.SQLITE:
            return SQLiteSchemaEditor(database)
        elif database.db_type == DatabaseType.POSTGRES:
            from core.db.postgres import PostgresSchemaEditor
            return PostgresSchemaEditor(database)

    # Default to SQLite for backward compatibility
    return SQLiteSchemaEditor(database)