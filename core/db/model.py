# core/db/model.py
import inspect
import json
import logging
import uuid
from enum import Enum
from typing import Type, Optional, TypeVar, Any, Dict, List, Union, Tuple

import aiosqlite
import asyncpg
import numpy as np

from core import exceptions
from core.db.fields import (
    DateTimeField,
    BaseField,
    JSONField,
    VectorField,
    CharField,
)
from core.db.queryset import QuerySet
from core.db.base import DATABASES, DatabaseType

T = TypeVar("T", bound="Model")  # Type variable for models


class classproperty(property):
    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


class Model:
    id: str = CharField(primary_key=True)
    db_alias: str = None
    PREFIX: str = None

    _fields: Dict[str, BaseField] = {}
    _json_fields: List[str] = []
    _data: Dict[str, Any] = {}

    DoesNotExist = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        cls.DoesNotExist = type("DoesNotExist", (Exception,), {})

        cls._fields = {
            **{
                k: v
                for k, v in cls.__dict__.items()
                if isinstance(v, BaseField) and k != "id"
            }
        }
        cls.db_alias = cls.get_app_label()
        cls.logger = logging.getLogger(f"Model.{cls.__name__}")

        # Add the id field first - make sure to get the class attribute, not the instance
        id_field = getattr(cls, "id")
        if isinstance(id_field, BaseField):
            cls._fields["id"] = id_field

        # Set field names
        for field_name, field in cls._fields.items():
            field.contribute_to_class(cls, field_name)

        # Collect JSON fields
        cls._json_fields = [
            k for k, v in cls._fields.items() if isinstance(v, JSONField)
        ]

    def __init__(self, **kwargs) -> None:
        # Determine the db_alias for this instance
        explicit_db_alias = kwargs.pop("db_alias", None)
        if explicit_db_alias:
            self.db_alias = explicit_db_alias
        else:
            app_label = self.__class__.get_app_label()

            # For backward compatibility, try using the app_label directly first
            if app_label in DATABASES.CONFIG:
                self.db_alias = app_label
            # Otherwise use router if available
            elif hasattr(DATABASES, 'router') and DATABASES.router:
                self.db_alias = DATABASES.router.db_for_app(app_label)
            # Final fallback to app_label
            else:
                self.db_alias = app_label
        # Handle original initialization logic without automatic ID generation
        for field_name, field in self._fields.items():
            value = kwargs.get(field_name)

            # Only use default if key wasn't in kwargs at all
            if value is None and field_name not in kwargs:
                default = field.default
                if callable(default):
                    value = default()
                else:
                    value = default

            # Set the attribute directly
            setattr(self, field_name, value)

    def __str__(self):
        return f"{self.__class__.__name__}(id={self.id})"

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id})"

    @classmethod
    def get_app_label(cls) -> str:
        """Auto-detect app label from module path."""
        module = inspect.getmodule(cls)
        if not module:
            raise ValueError(f"Could not determine module for {cls.__name__}")

        # Get full module path and file path
        module_path = module.__name__
        file_path = module.__file__ if module.__file__ else ""

        # If the path contains 'apps/', extract the app name
        if "apps/" in file_path:
            # Split on 'apps/' and take the next directory name
            parts = file_path.split("apps/")
            if len(parts) > 1:
                app_name = parts[1].split("/")[0]
                return app_name

        # Fallback to module path analysis
        parts = module_path.split(".")
        if "apps" in parts:
            apps_index = parts.index("apps")
            if len(parts) > apps_index + 1:
                app_name = parts[apps_index + 1]
                return app_name

        raise ValueError(
            f"Model {cls.__name__} must be under the apps/ directory. "
            f"Current path: {module_path}"
        )

    @classmethod
    def get_table_name(cls) -> str:
        """Get the table name for the model using {app}_{table} format."""
        table_name = cls.__name__
        snake_case = "".join(
            [f"_{c.lower()}" if c.isupper() else c.lower() for c in table_name]
        ).lstrip("_")

        # Get app label dynamically
        app_label = cls.get_app_label()
        name = f"{app_label}_{snake_case}"
        return name

    def clean(self):
        """Validate field values."""
        for field_name, field in self._fields.items():
            value = getattr(self, field_name, None)
            field.validate(value)

    # Inside class Model in core/db/model.py

    # Inside class Model in core/db/model.py

    @classmethod
    def _get_parsed_table_name(cls) -> Tuple[str, str]:
        """
        Parse table name into schema and table parts.
        Schema is cls.get_app_label().
        Table part is extracted from cls.get_table_name()
        by removing the "schema_" prefix.
        """
        schema = cls.get_app_label()
        full_table_name = cls.get_table_name()
        prefix = f"{schema}_"

        # Ensure we handle potential quoting in the full name just in case
        cleaned_full_name = full_table_name.replace('"', '')

        if cleaned_full_name.startswith(prefix):
            table_part = cleaned_full_name[len(prefix):]
            return schema, table_part
        else:
            return schema, full_table_name  # Fallback if convention is broken

    def _quote_orm(self, name: str) -> str:
        """Quotes identifiers for ORM queries if needed."""
        # Simple quoting, adjust if complex identifiers are used
        if any(c in name for c in ['"', '.', ' ']) or not name.islower():
            # Check if already quoted
            if name.startswith('"') and name.endswith('"'):
                return name
            return f'"{name}"'
        return name

    async def _save(self) -> None:
        """Internal method to save a new instance."""
        if not self.db_alias:
            raise ValueError(f"Cannot save model {self.__class__.__name__} without a db_alias.")
        # Ensure DATABASES is available and alias is configured/connected
        if DATABASES is None: raise RuntimeError("DATABASES global instance not initialized.")
        if self.db_alias not in DATABASES.CONFIG and self.db_alias not in DATABASES.CONNECTIONS:
            raise ValueError(
                f"Database alias '{self.db_alias}' not found in CONFIG or CONNECTIONS."
            )

        db = await DATABASES.get_connection(self.db_alias)
        is_postgres = getattr(db, 'db_type', None) == DatabaseType.POSTGRES

        # Determine table identifier (schema-qualified for PG)
        schema, table = self._get_parsed_table_name()
        if is_postgres:
            table_identifier = f"{self._quote_orm(schema)}.{self._quote_orm(table)}"
        else:
            # SQLite uses the plain name (schema isn't really separate)
            table_identifier = self._quote_orm(f"{schema}_{table}")  # Reconstruct original name and quote

        # Generate ID if needed (only if PK field is 'id' and value is missing)
        pk_field_name = None
        for fname, field in self._fields.items():
            if field.primary_key:
                pk_field_name = fname
                break

        if pk_field_name == 'id' and not self.id:
            self.id = self._generate_id()

        # Prepare field values for insert
        field_values_for_db = {}
        values_list = []  # List to hold values in order

        for field_name, field in self._fields.items():
            db_column_name = field.db_column or field_name
            value = getattr(self, field_name, None)

            if value is None and field.default is not None:
                value = field.default() if callable(field.default) else field.default

            if value is None and not field.null and not field.primary_key:
                raise exceptions.ValidationError(f"Field '{field_name}' cannot be null.")

            db_value = field.to_db(value)  # Convert to DB format
            field_values_for_db[db_column_name] = db_value

        # Ensure order matches columns for placeholder binding
        quoted_columns_list = []
        for db_col_name in field_values_for_db.keys():
            quoted_columns_list.append(self._quote_orm(db_col_name))
            values_list.append(field_values_for_db[db_col_name])  # Add value in correct order

        quoted_columns = ", ".join(quoted_columns_list)

        # Prepare placeholders
        if is_postgres:
            placeholders = ", ".join(f"${i + 1}" for i in range(len(values_list)))
            # Use the list directly for asyncpg
        else:
            placeholders = ", ".join("?" for _ in values_list)
            values_list = tuple(values_list)  # Use tuple for aiosqlite

        try:
            self.clean()  # Validate the model

            query = f'INSERT INTO {table_identifier} ({quoted_columns}) VALUES ({placeholders})'

            await db.execute(query, values_list)
            await db.commit()

        except Exception as e:
            await db.rollback()
            pk_val = getattr(self, pk_field_name, 'N/A')
            # Check for specific UndefinedTableError for better message
            if is_postgres and isinstance(e, asyncpg.exceptions.UndefinedTableError):
                error_msg = f"Table '{table_identifier}' does not exist. Target Schema: '{schema}', Table: '{table}'. Alias: '{self.db_alias}'. Ensure migrations ran correctly."
                self.logger.error(error_msg, exc_info=True)
                raise exceptions.DatabaseError(error_msg) from e
            # General error re-raise
            error_msg = f"Error saving {self.__class__.__name__} (id={pk_val}) to table '{table_identifier}' using alias '{self.db_alias}': {type(e).__name__}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise exceptions.DatabaseError(error_msg) from e

    async def _update(self):
        """Internal method to update an existing instance."""
        if not self.db_alias: raise ValueError(f"Cannot update model {self.__class__.__name__} without a db_alias.")
        if DATABASES is None: raise RuntimeError("DATABASES global instance not initialized.")
        db = await DATABASES.get_connection(self.db_alias)
        is_postgres = getattr(db, 'db_type', None) == DatabaseType.POSTGRES

        schema, table = self._get_parsed_table_name()
        if is_postgres:
            table_identifier = f"{self._quote_orm(schema)}.{self._quote_orm(table)}"
        else:
            table_identifier = self._quote_orm(f"{schema}_{table}")

        field_values_for_db = {}
        pk_field_name = None
        pk_db_column = None
        pk_value_db = None

        # Identify PK and prepare SET clause data
        for field_name, field in self._fields.items():
            if field.primary_key:
                pk_field_name = field_name
                pk_db_column = self._quote_orm(field.db_column or field_name)
                pk_value_db = field.to_db(getattr(self, field_name))
                continue  # Skip PK in SET clause

            # Prepare other fields for SET clause
            db_column_name = field.db_column or field_name
            value = getattr(self, field_name, None)
            if value is None and field.default is not None:
                value = field.default() if callable(field.default) else field.default
            if value is None and not field.null:
                raise exceptions.ValidationError(f"Field '{field_name}' cannot be null during update.")

            field_values_for_db[db_column_name] = field.to_db(value)

        if not field_values_for_db:
            self.logger.warning(
                f"Update called for {self.__class__.__name__} (id={getattr(self, pk_field_name, 'N/A')}) but no fields changed.")
            return  # Nothing to update
        if not pk_field_name: raise ValueError("Cannot update model without primary key.")
        if pk_value_db is None: raise ValueError("Primary key value is missing for update.")

        # Build SET clause and WHERE clause with appropriate placeholders
        set_clause_parts = []
        update_values = []  # Values for SET part
        param_index = 1

        if is_postgres:
            for db_col_name, db_val in field_values_for_db.items():
                set_clause_parts.append(f"{self._quote_orm(db_col_name)} = ${param_index}")
                update_values.append(db_val)
                param_index += 1
            # Add PK value for WHERE clause
            where_clause = f"{pk_db_column} = ${param_index}"
            update_values.append(pk_value_db)
            final_values = update_values  # Use list for PG
        else:
            for db_col_name, db_val in field_values_for_db.items():
                set_clause_parts.append(f"{self._quote_orm(db_col_name)} = ?")
                update_values.append(db_val)
            # Add PK value for WHERE clause
            where_clause = f"{pk_db_column} = ?"
            update_values.append(pk_value_db)
            final_values = tuple(update_values)  # Use tuple for SQLite

        set_clause = ", ".join(set_clause_parts)
        query = f"UPDATE {table_identifier} SET {set_clause} WHERE {where_clause}"

        try:
            self.clean()  # Validate before update

            cursor = await db.execute(query, final_values)
            await db.commit()
        except Exception as e:
            await db.rollback()
            pk_val = getattr(self, pk_field_name, 'N/A')
            error_msg = f"Error updating {self.__class__.__name__} (id={pk_val}) in table '{table_identifier}' using alias '{self.db_alias}': {type(e).__name__}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            # Distinguish between DB operational errors and integrity errors
            if isinstance(e, (aiosqlite.OperationalError, asyncpg.PostgresError)):
                raise exceptions.DatabaseError(error_msg) from e
            elif isinstance(e, (aiosqlite.IntegrityError, asyncpg.IntegrityConstraintViolationError)):
                raise exceptions.IntegrityError(error_msg) from e
            else:
                raise  # Re-raise unknown errors

    async def refresh_from_db(self, using: Optional[str] = None):
        """Reloads the instance's fields from the database, optionally specifying an alias."""
        db_alias_to_use = using or getattr(self, 'db_alias', None)
        if not db_alias_to_use:
            raise RuntimeError(f"Instance of {self.__class__.__name__} has no db_alias set for refresh_from_db.")

        # Ensure DATABASES is available
        if DATABASES is None: raise RuntimeError("DATABASES global instance not initialized.")
        db = await DATABASES.get_connection(db_alias_to_use)  # Get connection for the target alias
        table_name = self.get_table_name()

        # Find primary key
        pk_field_name = None
        pk_db_column = None
        pk_value = None
        for field_name, field in self._fields.items():
            if field.primary_key:
                pk_field_name = field_name
                pk_db_column = field.db_column or field_name
                pk_value = getattr(self, pk_field_name, None)
                break

        if pk_field_name is None or pk_value is None:
            raise exceptions.ObjectNotPersistedError(
                f"Cannot refresh {self.__class__.__name__} instance without a primary key value.")

        pk_db_value = self._fields[pk_field_name].to_db(pk_value)
        select_columns = ", ".join(
            f'"{field.db_column or name}"' for name, field in self._fields.items())  # Quote names
        query = f'SELECT {select_columns} FROM "{table_name}" WHERE "{pk_db_column}" = ?'  # Quote names

        row = await db.fetch_one(query, [pk_db_value])

        if row is None:
            raise self.DoesNotExist(
                f"{self.__class__.__name__} with primary key '{pk_value}' does not exist in database '{db_alias_to_use}'.")

        # Update instance attributes from the row
        for field_name, field in self._fields.items():
            db_column = field.db_column or field_name
            try:
                if db_column in row.keys():
                    value_from_db = row[db_column]
                    python_value = field.from_db(value_from_db)
                    setattr(self, field_name, python_value)
            except Exception as e:
                print(
                    f"Warning: Error processing field '{field_name}' (column: {db_column}) during refresh_from_db: {e}. Field may be incorrect.")

        # Crucially, update the instance's db_alias to the one used for refresh
        self.db_alias = db_alias_to_use

    async def delete(self, using: Optional[str] = None):
        """Delete the current instance from the database, optionally specifying an alias."""
        db_alias_to_use = using or getattr(self, 'db_alias', None)
        if not db_alias_to_use:
            raise RuntimeError(f"Instance of {self.__class__.__name__} has no db_alias set for delete.")

        if DATABASES is None: raise RuntimeError("DATABASES global instance not initialized.")
        db = await DATABASES.get_connection(db_alias_to_use)

        # --- CORRECTED: Use schema-qualified name ---
        is_postgres = getattr(db, 'db_type', None) == DatabaseType.POSTGRES
        schema, table = self._get_parsed_table_name()
        table_identifier = f"{self._quote_orm(schema)}.{self._quote_orm(table)}" if is_postgres \
                           else self._quote_orm(f"{schema}_{table}") # Reconstruct original name and quote for SQLite
        # --- END CORRECTION ---

        # Find primary key
        pk_field_name = None
        pk_db_column = None
        pk_value = None
        for field_name, field in self._fields.items():
            if field.primary_key:
                pk_field_name = field_name
                pk_db_column = field.db_column or field_name # Use db_column if available
                pk_value = getattr(self, pk_field_name, None)
                break

        if pk_field_name is None or pk_value is None:
            raise exceptions.ObjectNotPersistedError(
                f"Cannot delete {self.__class__.__name__} instance without a primary key value."
            )

        pk_db_value = self._fields[pk_field_name].to_db(pk_value)

        # Use correct placeholder and table identifier
        placeholder = '$1' if is_postgres else '?'
        # --- CORRECTED: Quote PK column name and use correct placeholder ---
        sql = f'DELETE FROM {table_identifier} WHERE {self._quote_orm(pk_db_column)} = {placeholder}'
        # --- END CORRECTION ---

        try:
            cursor_or_status = await db.execute(sql, [pk_db_value])

            # Determine deleted count based on DB type
            deleted_count = 0
            if is_postgres and isinstance(cursor_or_status, str) and 'DELETE ' in cursor_or_status:
                 deleted_count = int(cursor_or_status.split(' ')[1])
            elif not is_postgres and hasattr(cursor_or_status, 'rowcount'): # Check rowcount for SQLite
                 deleted_count = cursor_or_status.rowcount
                 await cursor_or_status.close() # Close cursor for SQLite

            if deleted_count == 0:
                raise self.DoesNotExist(
                    f"{self.__class__.__name__} with primary key '{pk_value}' does not exist in database '{db_alias_to_use}'.")

            await db.commit()
        except self.DoesNotExist:
             raise # Re-raise if caught
        except Exception as e:
            await db.rollback()
            # More specific error wrapping
            error_msg = f"Error deleting {self.__class__.__name__} (id={pk_value}) from {table_identifier}: {type(e).__name__}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            if isinstance(e, (aiosqlite.OperationalError, asyncpg.PostgresError)):
                raise exceptions.DatabaseError(error_msg) from e
            else:
                raise exceptions.DatabaseError(f"Unexpected error during delete: {e}") from e

    @classmethod
    def describe(cls):
        """Get a description of the model's fields."""
        fields = {}

        # Add all fields using their describe() method
        for name, field in cls._fields.items():
            fields[name] = field.describe()

        return {"fields": fields}

    @classmethod
    def _prepare_filter_value(cls, value: Any) -> Any:
        """Prepare filter values based on their type."""
        if isinstance(value, np.ndarray):
            return value.astype(np.float32).tobytes()
        if isinstance(value, list) and all(isinstance(v, float) for v in value):
            return np.array(value, dtype=np.float32).tobytes()
        elif isinstance(value, (dict, list)):
            return json.dumps(value)
        elif isinstance(value, bool):
            return int(value)
        elif isinstance(value, float):
            return value
        elif isinstance(value, Enum):
            return value.value
        else:
            return value

    # Define 'objects' as a class property
    @classproperty
    def objects(cls) -> QuerySet:
        """Return a QuerySet bound to the model's db_alias."""
        return QuerySet(cls)

    async def update(self, **kwargs):
        """Updates the model instance."""
        ignore_fields = ["id"]
        for field, value in kwargs.items():
            if field in ignore_fields:
                continue
            setattr(self, field, value)

    def using(self, db_alias: str) -> "Model":
        """Set the database alias to be used for this query."""
        if not db_alias:
            raise ValueError("db_alias cannot be empty.")
        self.db_alias = db_alias
        return self

    @classmethod
    async def get(cls: Type[T], id: int) -> Optional[T]:
        """Retrieve a single instance by ID."""
        return await cls.objects.get(id=id)

    @classmethod
    def _generate_id(cls):
        """Generates a UUID-based ID with a prefix."""
        prefix = cls.PREFIX or cls.__name__.lower()[:3]
        return f"{prefix}-{uuid.uuid4().hex[:24]}"

    # Also, update from_row to handle dicts from PostgreSQL correctly
    @classmethod
    def from_row(cls, row: Union[aiosqlite.Row, Dict], db_alias: str) -> "Model":
        """Instantiate a Model object from a database row (aiosqlite.Row or dict)."""
        kwargs = {}
        is_dict = isinstance(row, dict) # Check if it's a dictionary (from asyncpg)

        for field_name, field in cls._fields.items():
            db_column = field.db_column or field_name
            try:
                # Access value by key whether it's a Row or dict
                value = row[db_column]
            except (KeyError, IndexError):
                 # Handle case where column might not exist in the result set
                 value = None

            if value is not None:
                value = field.from_db(value)
            else:
                # If DB returned None, respect it unless there's a non-callable default
                # and the field is NOT nullable (this case shouldn't really happen with proper schema)
                # Keep value as None if field allows null or DB returned null.
                # Default is primarily for object creation, not hydration from DB.
                pass # Keep value as None

            kwargs[field_name] = value

        obj = cls(**kwargs)

        # Set the primary key ('id') directly from the row data
        # asyncpg.Record acts like a dict, aiosqlite.Row allows key access
        pk_field_name = 'id' # Assuming 'id' is the standard PK name
        try:
            obj.id = row[pk_field_name]
        except (KeyError, IndexError):
            obj.id = None # Or potentially raise error if ID is expected

        obj.db_alias = db_alias
        return obj

    async def save(self, create: bool = False, using: Optional[str] = None):
        """Save or update the instance, optionally specifying a database alias."""
        # Use the explicitly provided alias, or the one already set on the instance
        db_alias_to_use = using or getattr(self, 'db_alias', None)

        # Only if no alias is specified at all, use the router to determine it
        if not db_alias_to_use:
            app_label = self.__class__.get_app_label()
            if hasattr(DATABASES, 'router'):
                db_alias_to_use = DATABASES.router.db_for_app(app_label)
            else:
                db_alias_to_use = app_label

        if not db_alias_to_use:
            raise RuntimeError(f"Cannot determine database alias for saving {self.__class__.__name__}.")

        # Set the working alias for this operation
        original_alias = self.db_alias
        self.db_alias = db_alias_to_use

        try:
            # Only generate ID if not already set and in create mode
            if create and not self.id:
                self.id = self._generate_id()

            # Choose save mode based on ID and create flag
            if self.id and not create:
                await self._update()
            else:
                await self._save()

        finally:
            # Restore original alias if it was different
            if original_alias != db_alias_to_use:
                self.db_alias = original_alias

    def serialize(self) -> Dict:
        """Serialize the Model instance to a dictionary."""
        data = {}
        ignored = ["vector", "metadata"]

        for field_name, field in self._fields.items():
            if field_name in ignored:
                continue
            value = getattr(self, field_name, None)

            # Handle JSON fields specifically
            if field_name in self._json_fields:
                if value is not None:
                    data[field_name] = value
                else:
                    data[field_name] = None

            # Handle VectorField specifically
            elif isinstance(field, VectorField):
                if value is not None and isinstance(value, np.ndarray):
                    data[field_name] = value.tolist()
                else:
                    data[field_name] = value
            # Handle DateTimeField specifically
            elif isinstance(field, DateTimeField):
                if value is not None:
                    data[field_name] = value.isoformat()
                else:
                    data[field_name] = None
            # Handle other field types
            else:
                data[field_name] = value

        if self.id is not None:
            data["id"] = self.id

        return data
