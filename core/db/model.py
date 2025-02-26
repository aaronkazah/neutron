# core/db/model.py
import inspect
import json
import uuid
from enum import Enum
from typing import Type, Optional, TypeVar, Any, Dict, List

import aiosqlite
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
from core.db.base import DATABASES

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
        # Handle ID first to ensure it's available before other fields
        id_value = kwargs.get("id")
        if id_value is not None:
            self.id = id_value
        else:
            self.id = None

        # Handle remaining fields
        for field_name, field in self._fields.items():
            if field_name == "id":
                continue

            default = field.default
            if callable(default):
                default = default()

            value = kwargs.get(field_name, default)

            # Let each field type handle its own validation and conversion
            try:
                if value is not None:
                    field.validate(value)
                    # Convert the value using the field's to_db method
                    # This ensures proper type conversion for all field types
                    converted_value = field.from_db(field.to_db(value))
                    value = converted_value
            except Exception as e:
                raise ValueError(f"Invalid value for field {field_name}: {str(e)}")

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

    async def _save(self) -> None:
        """Internal method to save a new instance."""

        if self.db_alias not in DATABASES.CONFIG:
            raise ValueError(
                f"Database alias '{self.db_alias}' not found in CONFIG: {DATABASES.CONFIG.keys()}"
            )

        db = await DATABASES.get_connection(self.db_alias)

        table_name = self.get_table_name()
        field_values = {}

        # Only generate ID if not already set
        if not self.id:
            self.id = self._generate_id()

        # Include id in field values
        field_values["id"] = self.id

        for field_name, field in self._fields.items():
            if field_name == "id":  # Skip id as we've already handled it
                continue
            field.db_column = field_name
            value = getattr(self, field_name, field.default)
            if value is None and not field.null:
                raise exceptions.ValidationError(
                    f"Field '{field_name}' cannot be null."
                )
            db_value = field.to_db(value)
            field_values[field.db_column or field_name] = db_value

        columns = ", ".join(field_values.keys())
        placeholders = ", ".join("?" for _ in field_values)
        values = tuple(field_values.values())

        try:
            self.clean()
            await db.execute(
                f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})", values
            )
            await db.commit()
        except aiosqlite.OperationalError as e:
            raise e
        except aiosqlite.IntegrityError as e:
            raise exceptions.IntegrityError(e)
        except ValueError as e:
            raise exceptions.ValidationError(e)
        except Exception as e:
            raise e

    async def _update(self):
        """Internal method to update an existing instance."""
        if self.db_alias not in DATABASES.CONFIG:
            raise ValueError(
                f"Database alias '{self.db_alias}' not found in DATABASES."
            )

        db = await DATABASES.get_connection(self.db_alias)
        table_name = self.get_table_name()

        # Build update data
        field_values = {}
        for field_name, field in self._fields.items():
            if field_name == "id":  # Skip id as it's used in WHERE clause
                continue
            value = getattr(self, field_name, field.default)
            if value is None and not field.null:
                raise ValueError(f"Field '{field_name}' cannot be null.")
            db_value = field.to_db(value)
            field_values[field.db_column or field_name] = db_value

        # Build UPDATE query
        set_clause = ", ".join(f"{key} = ?" for key in field_values.keys())
        values = tuple(field_values.values()) + (self.id,)
        query = f"UPDATE {table_name} SET {set_clause} WHERE id = ?"

        try:
            await db.execute(query, values)
            await db.commit()
        except aiosqlite.OperationalError as e:
            raise e
        except aiosqlite.IntegrityError as e:
            raise exceptions.IntegrityError(e)

    async def refresh_from_db(self):
        """Reloads the instance's fields from the database."""
        db_alias = self.db_alias
        db = await DATABASES.get_connection(db_alias)
        table_name = self.get_table_name()

        query = f"SELECT * FROM {table_name} WHERE id = ?"
        cursor = await db.execute(query, (self.id,))
        row = await cursor.fetchone()

        if row is None:
            raise self.DoesNotExist(
                f"{self.__class__.__name__} with id '{self.id}' does not exist."
            )

        for field_name, field in self._fields.items():
            db_column = field.db_column or field_name
            field.db_column = db_column
            value = row[db_column]
            if value is not None:
                setattr(self, field_name, field.from_db(value))
            else:
                setattr(self, field_name, field.default)

    async def delete(self):
        """Delete the current instance from the database."""
        db = await DATABASES.get_connection(self.db_alias)
        table_name = self.get_table_name()

        await db.execute(f"DELETE FROM {table_name} WHERE id = ?", (self.id,))
        await db.commit()

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

    async def update(self, fields: List[str] = None, data: Dict[str, Any] = None):
        """Updates the model instance."""
        ignore_fields = ["id", "workspace_id", "created_at", "modified_at"]
        for field, value in data.items():
            if field in ignore_fields:
                continue
            setattr(self, field, value)

    def using(self, db_alias: str) -> "Model":
        """Set the database alias to be used for this query."""
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

    @classmethod
    def from_row(cls, row: aiosqlite.Row, db_alias: str) -> "Model":
        """Instantiate a Model object from a database row."""
        kwargs = {}
        for field_name, field in cls._fields.items():
            db_column = field.db_column or field_name
            value = row[db_column]
            if value is not None:
                value = field.from_db(value)
            else:
                value = field.default
            kwargs[field_name] = value
        obj = cls(**kwargs)
        obj.id = row["id"]
        obj.db_alias = db_alias
        return obj

    async def save(self, *args, **kwargs):
        """Save or update the instance."""
        if self.id is None or kwargs.get("create", False):
            await self._save()
        else:
            await self._update()

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
