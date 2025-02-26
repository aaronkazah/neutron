# core/migrations.py
import datetime
import importlib
import inspect
import re
import sys
import textwrap
import traceback
import uuid
from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import List, Type, Dict, Set

from core.db.editor import get_schema_editor
from core.db.fields import *
from core.db.model import Model
from core.db.base import DATABASES


class Migration:
    def __init__(self, app_label, operations, dependencies=None):
        self.app_label = app_label
        self.dependencies = dependencies or []
        self.operations = operations

    async def apply(self, project_state, schema_editor, connection):
        """Apply migration operations"""
        for operation in self.operations:
            await operation.database_forwards(
                app_label=self.app_label,
                schema_editor=schema_editor,
                from_state=None,
                to_state=project_state,
                connection=connection,
            )

    def __repr__(self):
        return f"<Migration {self.app_label}>"


from abc import ABC, abstractmethod


class Operation(ABC):
    def _get_table_name(self, app_label, model_name):
        """Convert model_name to proper table name format."""
        if "." in model_name:
            prefix, name = model_name.split(".")
            return f"{prefix}_{name.lower()}"
        return f"{app_label}_{model_name.lower()}"

    @abstractmethod
    async def database_forwards(
        self, app_label, schema_editor, from_state, to_state, connection
    ):
        """Performs the database operation forwards."""
        pass

    @abstractmethod
    async def database_backwards(
        self, app_label, schema_editor, from_state, to_state, connection
    ):
        """Performs the database operation in reverse."""
        pass

    @abstractmethod
    def describe(self):
        """Returns a human-readable description of the operation."""
        pass


class CreateModel(Operation):
    def __init__(self, model_name, fields):
        self.model_name = model_name
        self.fields = fields

    async def database_forwards(
        self, app_label, schema_editor, from_state, to_state, connection
    ):
        field_items = [(name, field) for name, field in self.fields.items()]
        table_name = self._get_table_name(app_label, self.model_name)
        await schema_editor.create_table(table_name, field_items)

    async def database_backwards(
        self, app_label, schema_editor, from_state, to_state, connection
    ):
        table_name = self._get_table_name(app_label, self.model_name)
        await schema_editor.drop_table(table_name)

    def describe(self):
        return f"CreateModel('{self.model_name}', fields={{"


class DeleteModel(Operation):
    def __init__(self, model_name):
        self.model_name = model_name

    async def database_forwards(
        self, app_label, schema_editor, from_state, to_state, connection
    ):
        table_name = self._get_table_name(app_label, self.model_name)
        await schema_editor.drop_table(table_name)

    async def database_backwards(
        self, app_label, schema_editor, from_state, to_state, connection
    ):
        pass

    def describe(self):
        return f"DeleteModel('{self.model_name}')"


class AddField(Operation):
    def __init__(self, model_name, field_name, field):
        self.model_name = model_name
        self.field_name = field_name
        self.field = field

    async def database_forwards(
        self, app_label, schema_editor, from_state, to_state, connection
    ):
        table_name = self._get_table_name(app_label, self.model_name)
        # The schema_editor.add_column will now skip if column exists
        await schema_editor.add_column(table_name, self.field_name, self.field)

    async def database_backwards(
        self, app_label, schema_editor, from_state, to_state, connection
    ):
        table_name = self._get_table_name(app_label, self.model_name)
        # Only remove if column exists
        if await schema_editor.column_exists(table_name, self.field_name):
            await schema_editor.remove_column(table_name, self.field_name)

    def describe(self):
        return f"AddField('{self.model_name}', '{self.field_name}', {self.field.describe()})"

class RemoveField(Operation):
    def __init__(self, model_name, field_name):
        self.model_name = model_name
        self.field_name = field_name

    async def database_forwards(
        self, app_label, schema_editor, from_state, to_state, connection
    ):
        table_name = self._get_table_name(app_label, self.model_name)
        await schema_editor.remove_column(table_name, self.field_name)

    async def database_backwards(
        self, app_label, schema_editor, from_state, to_state, connection
    ):
        print(
            f"  Re-adding field '{self.field_name}' to model '{self.model_name}' in app '{app_label}' - field details unknown"
        )

    def describe(self):
        return f"RemoveField('{self.model_name}', '{self.field_name}')"


class AlterField(Operation):
    def __init__(self, model_name, field_name, field):
        self.model_name = model_name
        self.field_name = field_name
        self.field = field

    async def database_forwards(
        self, app_label, schema_editor, from_state, to_state, connection
    ):
        table_name = self._get_table_name(app_label, self.model_name)
        await schema_editor.alter_column(table_name, self.field_name, self.field)

    async def database_backwards(
        self, app_label, schema_editor, from_state, to_state, connection
    ):
        print(
            f"  Reverting field '{self.field_name}' on model '{self.model_name}' in app '{app_label}' - old field details unknown"
        )

    def describe(self):
        return f"AlterField('{self.model_name}', '{self.field_name}', {self.field.describe()})"


class RenameField(Operation):
    def __init__(self, model_name, old_field_name, new_field_name):
        self.model_name = model_name
        self.old_field_name = old_field_name
        self.new_field_name = new_field_name

    async def database_forwards(
        self, app_label, schema_editor, from_state, to_state, connection
    ):
        table_name = self._get_table_name(app_label, self.model_name)
        await schema_editor.rename_column(
            table_name, self.old_field_name, self.new_field_name
        )

    async def database_backwards(
        self, app_label, schema_editor, from_state, to_state, connection
    ):
        table_name = self._get_table_name(app_label, self.model_name)
        await schema_editor.rename_column(
            table_name, self.new_field_name, self.old_field_name
        )

    def describe(self):
        return f"RenameField('{self.model_name}', '{self.old_field_name}', '{self.new_field_name}')"


class RenameModel(Operation):
    def __init__(self, old_model_name, new_model_name):
        self.old_model_name = old_model_name
        self.new_model_name = new_model_name

    async def database_forwards(
        self, app_label, schema_editor, from_state, to_state, connection
    ):
        old_table = self._get_table_name(app_label, self.old_model_name)
        new_table = self._get_table_name(app_label, self.new_model_name)
        await schema_editor.rename_table(old_table, new_table)

    async def database_backwards(
        self, app_label, schema_editor, from_state, to_state, connection
    ):
        old_table = self._get_table_name(app_label, self.old_model_name)
        new_table = self._get_table_name(app_label, self.new_model_name)
        await schema_editor.rename_table(new_table, old_table)

    def describe(self):
        return f"RenameModel('{self.old_model_name}', '{self.new_model_name}')"


class MigrationManager:
    def __init__(self, apps=None, base_dir="apps"):
        """
        Initialize MigrationManager with base apps directory

        Args:
            apps (list): Optional list of specific apps to manage
            base_dir (str): Base directory containing all apps (defaults to 'apps')
        """
        self.apps = apps if apps is not None else self._discover_apps(base_dir)
        self.base_dir = base_dir
        self.project_state = {}
        self._models_cache: Dict[str, List[Type]] = {}

    def _discover_apps(self, base_dir: str) -> List[str]:
        """
        Discover all apps in the base directory

        Args:
            base_dir: Base directory containing all apps

        Returns:
            List of app names that contain a models directory
        """
        apps = []
        for item in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, item)):
                models_dir = os.path.join(base_dir, item, "models")
                if os.path.isdir(models_dir):
                    apps.append(item)
        return apps

    def _discover_models(self, app_label: str) -> List[Type]:
        """
        Discover all models in an app's models directory
        """
        if app_label in self._models_cache:
            return self._models_cache[app_label]

        models = []
        models_dir = os.path.join(self.base_dir, app_label, "models")

        if not os.path.exists(models_dir):
            return models

        import sys

        sys.path.insert(0, self.base_dir)

        try:
            for filename in os.listdir(models_dir):
                if filename.endswith(".py") and not filename.startswith("__"):
                    module_name = f"{app_label}.models.{filename[:-3]}"
                    try:
                        module = importlib.import_module(module_name)

                        for name, obj in inspect.getmembers(module):
                            if (
                                inspect.isclass(obj)
                                and hasattr(obj, "_fields")
                                and not name.startswith("_")
                                and obj.__module__.startswith(f"{app_label}.")
                            ):  # <- Add this check
                                # Don't include the base Model class
                                if obj.__name__ != "Model":  # <- And this check
                                    models.append(obj)
                    except Exception as e:
                        print(f"Error importing {module_name}: {str(e)}")
                        # Print the traceback for debugging
                        traceback.print_exc()
                        # File information
                        print(f"File: {module_name}")
                        continue
        finally:
            sys.path.remove(self.base_dir)

        self._models_cache[app_label] = models
        return models

    async def bootstrap_all(self, test_mode: bool = True):
        """
        Bootstrap all apps. If db is None, each app uses its own database.
        """

        for app_label in self.apps:
            models = self._discover_models(app_label)
            if models:
                await self.bootstrap(
                    app_label=app_label,
                    models=models,
                    db=None,
                    test_mode=test_mode,
                )

    async def bootstrap(
        self,
        app_label: str,
        models: Optional[list] = None,
        db: Optional[str] = None,
        test_mode: bool = True,
        **kwargs,
    ):
        """
        Bootstrap the database with models using provided config.
        If models is None, autodiscover models for the app.
        If db is None, use app_label as the database alias.
        """

        if models is None:
            models = self._discover_models(app_label)

        if not models:
            print(f"No models found for app {app_label}")
            return

        db = db or app_label

        # Database will be auto-configured if not exists
        connection = await DATABASES.get_connection(db, test_mode=test_mode)

        try:

            # Generate and apply migrations
            operations = await self.makemigrations(
                app_label=app_label,
                models=models,
                return_ops=test_mode,
                clean=test_mode,
            )

            await self.migrate(
                app_label=app_label,
                connection=connection,
                operations=operations if test_mode else None,
            )

            return db, connection

        except Exception as e:
            await connection.rollback()
            raise e

    def _find_enum_location(self, enum_class: Type) -> str:
        """Find the actual module location of an enum by using its __module__ attribute"""
        return enum_class.__module__

    def _get_required_enum_imports(self, operations: List[Operation]) -> List[str]:
        """Determine which enum imports are needed based on the operations"""
        required_enums = set()

        for op in operations:
            if isinstance(op, CreateModel):
                for field in op.fields.values():
                    if isinstance(field, EnumField):
                        enum_class = field.enum_class
                        required_enums.add(
                            f"from {enum_class.__module__} import {enum_class.__name__}"
                        )

        return sorted(list(required_enums))

    def _generate_migration_file_content(
        self,
        app_label: str,
        operations: List[Operation],
        models: List[Type],
    ) -> str:
        """Generate the content of a migration file"""
        operations_str = self._format_operations(operations)
        migration_id = int(datetime.datetime.now(datetime.UTC).timestamp())

        # Get required enum imports based on the operations
        enum_imports = self._get_required_enum_imports(operations)

        # Generate state hash
        state = {}
        for model_class in models:
            state[model_class.__name__] = {
                "fields": dict(
                    (name, field.describe())
                    for name, field in model_class._fields.items()
                )
            }
        state_json = json.dumps(state, indent=4)

        # Generate the file content with HASH at the bottom
        return textwrap.dedent(
            f"""\
# Generated by MigrationManager

import datetime
import json
from core.db.migrations import Migration
from core.db.migrations import (
    CreateModel,
    DeleteModel,
    AddField,
    RemoveField,
    AlterField,
    RenameField,
    RenameModel,
)
from core.db.fields import (
    IntegerField,
    CharField,
    TextField,
    DateTimeField,
    JSONField,
    VectorField,
    BooleanField,
    EnumField,
    FloatField,
    BinaryField
)
from core.db.model import Model
{os.linesep.join(enum_imports) if enum_imports else ''}

class Migration{migration_id}(Migration):
    \"\"\"
    Auto-generated migration
    \"\"\"

    dependencies = []

    operations = [
{operations_str}
    ]

# DO NOT CHANGE
HASH = {state_json}
    """
        )

    def _get_model_dependencies(self, model: Type) -> Dict[str, Set[Type]]:
        """Recursively get all model and enum dependencies from fields"""
        model_deps = set()
        enum_deps = set()

        # If it's an Enum class, add to enum deps
        if isinstance(model, type) and issubclass(model, Enum):
            enum_deps.add(model)
            return {"models": model_deps, "enums": enum_deps}

        # Process Model fields
        if hasattr(model, "_fields"):
            for field in model._fields.values():
                # Check if field is a reference to another model
                if hasattr(field, "model_class"):
                    ref_model = field.model_class
                    if ref_model and ref_model != model:
                        model_deps.add(ref_model)
                        # Recursively get dependencies
                        nested_deps = self._get_model_dependencies(ref_model)
                        model_deps.update(nested_deps["models"])
                        enum_deps.update(nested_deps["enums"])

                # Check if field is an enum
                if hasattr(field, "enum_class"):
                    enum_class = field.enum_class
                    if enum_class:
                        enum_deps.add(enum_class)

        return {"models": model_deps, "enums": enum_deps}

    def get_migrations_dir(self, app_label: str) -> str:
        """Get the migrations directory path for an app"""
        return os.path.join(self.base_dir, app_label, "migrations")

    def _prefix_model_name(self, app_label: str, model_name: str) -> str:
        """Prefix model name with app label to prevent clashes"""
        return f"{app_label}.{model_name}"

    def _detect_changes(self, previous_state, current_state, models, app_label):
        """Detects changes between the previous and current model states."""
        operations = []

        # Create model lookup dict
        model_lookup = {model.__name__: model for model in models}

        # Detect new models
        for model_name in current_state:
            if model_name not in previous_state:
                model_class = model_lookup[model_name]
                prefixed_name = self._prefix_model_name(app_label, model_name)
                operations.append(
                    CreateModel(prefixed_name, fields=model_class._fields)
                )

        # Detect deleted models
        for model_name in previous_state:
            if model_name not in current_state:
                prefixed_name = self._prefix_model_name(app_label, model_name)
                operations.append(DeleteModel(prefixed_name))

        # Detect field changes
        for model_name in current_state:
            if model_name in previous_state:
                model_class = model_lookup[model_name]
                prefixed_name = self._prefix_model_name(app_label, model_name)

                # Add type checking and conversion if needed
                if isinstance(previous_state[model_name], str):
                    try:
                        previous_model_state = json.loads(previous_state[model_name])
                    except json.JSONDecodeError:
                        print(f"Error: Invalid JSON in previous state for {model_name}")
                        continue
                else:
                    previous_model_state = previous_state[model_name]

                current_fields = current_state[model_name]["fields"]
                previous_fields = previous_model_state["fields"]

                # Field operations...
                for field_name in current_fields:
                    if field_name not in previous_fields:
                        operations.append(
                            AddField(
                                prefixed_name,
                                field_name,
                                model_class._fields[field_name],
                            )
                        )

                for field_name in previous_fields:
                    if field_name not in current_fields:
                        operations.append(RemoveField(prefixed_name, field_name))

                for field_name in current_fields:
                    if field_name in previous_fields:
                        if current_fields[field_name] != previous_fields[field_name]:
                            operations.append(
                                AlterField(
                                    prefixed_name,
                                    field_name,
                                    model_class._fields[field_name],
                                )
                            )

        return operations

    def _generate_migration_filename(self, migrations_dir):
        """Generate a unique migration filename"""
        existing_migrations = [
            f
            for f in os.listdir(migrations_dir)
            if f.endswith(".py") and f != "__init__.py"
        ]
        existing_numbers = [
            int(f.split("_")[0])
            for f in existing_migrations
            if f.split("_")[0].isdigit()
        ]
        next_number = max(existing_numbers, default=0) + 1
        return f"{str(next_number).zfill(4)}_auto.py"

    def _format_operations(self, operations: List[Operation]) -> str:
        """Format migration operations as string"""
        formatted = []
        for op in operations:
            if isinstance(op, CreateModel):
                fields_str = ",\n            ".join(
                    f"'{name}': {field.describe()}" for name, field in op.fields.items()
                )
                formatted.append(
                    f"""        CreateModel(
            '{op.model_name}',
            fields={{
                {fields_str}
            }}
        )"""
                )
            elif isinstance(op, DeleteModel):
                formatted.append(f"        DeleteModel('{op.model_name}')")
            elif isinstance(op, AddField):
                formatted.append(
                    f"        AddField('{op.model_name}', '{op.field_name}', {op.field.describe()})"
                )
            elif isinstance(op, RemoveField):
                formatted.append(
                    f"        RemoveField('{op.model_name}', '{op.field_name}')"
                )
            elif isinstance(op, AlterField):
                formatted.append(
                    f"        AlterField('{op.model_name}', '{op.field_name}', {op.field.describe()})"
                )
            elif isinstance(op, RenameField):
                formatted.append(
                    f"        RenameField('{op.model_name}', '{op.old_field_name}', '{op.new_field_name}')"
                )
            elif isinstance(op, RenameModel):
                formatted.append(
                    f"        RenameModel('{op.old_model_name}', '{op.new_model_name}')"
                )

        return ",\n".join(formatted)

    def _build_state_from_migrations(self, app_label):
        """Build current state from existing migrations by loading the HASH from the latest migration"""
        state = {}
        migrations_dir = self.get_migrations_dir(app_label)

        if not os.path.exists(migrations_dir):
            return state

        migration_files = sorted(
            [
                f
                for f in os.listdir(migrations_dir)
                if f.endswith(".py") and f != "__init__.py"
            ],
            key=lambda x: int(x.split("_")[0]),
        )

        if migration_files:
            last_migration_file = migration_files[-1]
            module_name = (
                f"{app_label}.migrations.{os.path.splitext(last_migration_file)[0]}"
            )

            try:
                # Import the migration module
                migration_module = importlib.import_module(module_name)
                # Get the HASH directly from the module - it's already a dict
                if hasattr(migration_module, "HASH"):
                    state = migration_module.HASH
                else:
                    print(f"Warning: No HASH variable found in {module_name}")

            except Exception as e:
                print(f"Error loading migration state from {module_name}: {str(e)}")
                raise  # Re-raise the exception to prevent creating new migrations

        return state

    async def _load_migration(
        self, app_label: str, migrations_dir: str, migration_file: str
    ) -> Migration:
        """Load a migration from a file."""
        migration_path = Path(migrations_dir) / migration_file
        module_name = f"{app_label}.migrations.{migration_file[:-3]}"

        try:
            # Remove the module from sys.modules if it exists to ensure fresh load
            if module_name in sys.modules:
                del sys.modules[module_name]

            # Load the module using importlib
            spec = importlib.util.spec_from_file_location(module_name, migration_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load migration file: {migration_path}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Find migration class - looking specifically for classes that start with Migration and aren't the base Migration class
            migration_class = None
            for name, obj in vars(module).items():
                if (
                    isinstance(obj, type)
                    and issubclass(obj, Migration)
                    and name.startswith("Migration")
                    and name != "Migration"
                ):  # This is the key change - exclude the base Migration class
                    migration_class = obj
                    break

            if not migration_class:
                raise ValueError(f"No migration class found in {migration_file}")

            # Create instance and get operations from the class
            operations = getattr(migration_class, "operations", [])

            migration = migration_class(app_label, operations=operations)

            return migration

        except Exception as e:
            print(f"Error loading migration {migration_file}: {str(e)}")
            import traceback

            traceback.print_exc()
            raise

    async def get_migrations(self, app_label=None):
        """
        Gets all migrations for an app or all apps.

        Args:
            app_label (str, optional): Specific app to get migrations for

        Returns:
            dict: Dictionary of app labels to list of migrations
        """
        migrations = {}

        if app_label:
            app_labels = [app_label]
        else:
            app_labels = self.apps

        for app in app_labels:
            migrations_dir = self.get_migrations_dir(app)
            if not os.path.exists(migrations_dir):
                continue

            migration_files = sorted(
                f
                for f in os.listdir(migrations_dir)
                if f.endswith(".py") and f != "__init__.py"
            )

            app_migrations = []
            for migration_file in migration_files:
                try:
                    migration = await self._load_migration(
                        app, migrations_dir, migration_file
                    )
                    app_migrations.append(migration)
                except Exception as e:
                    print(f"Error loading migration {migration_file}: {str(e)}")
                    continue

            migrations[app] = app_migrations

        return migrations

    async def makemigrations(
        self,
        app_label: str,
        models: list,
        return_ops: bool = False,
        clean: bool = False,
    ) -> Optional[List[Operation]]:
        """
        Generate migrations for models.
        """
        # In makemigrations method:
        state = {}
        for model_class in models:
            state[model_class.__name__] = {
                "fields": dict(
                    (name, field.describe())
                    for name, field in model_class._fields.items()
                )
            }

        if clean:
            previous_state = {}
        else:
            migrations_dir = self.get_migrations_dir(app_label)
            os.makedirs(migrations_dir, exist_ok=True)

            init_path = os.path.join(migrations_dir, "__init__.py")
            if not os.path.exists(init_path):
                with open(init_path, "w") as f:
                    f.write("")

            previous_state = self._build_state_from_migrations(app_label)

        # Detect changes and generate operations
        operations = self._detect_changes(previous_state, state, models, app_label)

        if not operations:
            return [] if return_ops else None

        if return_ops:
            return operations

        if not clean:
            migrations_dir = self.get_migrations_dir(app_label)
            migration_filename = self._generate_migration_filename(migrations_dir)
            migration_content = self._generate_migration_file_content(
                app_label=app_label, operations=operations, models=models
            )

            filepath = os.path.join(migrations_dir, migration_filename)
            with open(filepath, "w") as f:
                f.write(migration_content)

        return operations

    async def migrate(self, app_label, connection, operations=None):
        """Applies migrations to the database."""

        async with get_schema_editor(connection) as schema_editor:
            if operations is not None:
                migration = Migration(app_label, operations)
                await migration.apply(
                    project_state=self.project_state,
                    schema_editor=schema_editor,
                    connection=connection,
                )
                await connection.commit()
                return

            migrations_dir = self.get_migrations_dir(app_label)

            if not os.path.exists(migrations_dir):
                print(f"No migrations directory found for {app_label}")
                return

            # File-based migrations
            migration_files = sorted(
                f
                for f in os.listdir(migrations_dir)
                if f.endswith(".py") and f != "__init__.py"
            )

            try:
                for migration_file in migration_files:
                    # Load and apply migration
                    migration = await self._load_migration(
                        app_label, migrations_dir, migration_file
                    )

                    # Apply the migration with project_state
                    await migration.apply(
                        project_state=self.project_state,
                        schema_editor=schema_editor,
                        connection=connection,
                    )

                await connection.commit()

            except Exception as e:
                print(f"ERROR in migrate: {str(e)}")
                await connection.rollback()
                raise

        print(f"Migration {app_label} completed successfully.")
