# core/db/queryset.py
import datetime
import json
from enum import Enum
from typing import Type, TypeVar, List, Optional, Tuple, Any, Union, Dict

import aiosqlite
import numpy as np
from core.db.fields import DateTimeField
from core.db.base import DATABASES, DatabaseType

T = TypeVar("T", bound="Model")  # Type variable for models


class LazyQuerySet:
    """Base class for lazy query evaluation."""

    def __init__(self):
        self._result_cache = None
        self._fetch_attempted = False

    async def _fetch_all(self) -> List[Any]:
        raise NotImplementedError

    def _clone(self):
        raise NotImplementedError

    def __await__(self):
        """Allow using await on the queryset to get all results."""

        async def _get_all():
            if self._result_cache is None:
                self._result_cache = await self._fetch_all()
            return self

        return _get_all().__await__()

    async def __aiter__(self):
        """Make the queryset async iterable."""
        if self._result_cache is None:
            self._result_cache = await self._fetch_all()
        for item in self._result_cache:
            yield item

    def __len__(self):
        """Return the length of the results."""
        return len(self._result_cache)

    def __iter__(self):
        """Return an iterator over the results."""
        return iter(self._result_cache)

    def __getitem__(self, k):
        """Retrieve an item or slice from the set of results."""
        if isinstance(k, int):
            return self._result_cache[k]


class QuerySet(LazyQuerySet):
    LOOKUP_OPERATORS = {
        "exact": "= ?",
        "iexact": "LIKE ? COLLATE NOCASE",
        "contains": "LIKE ?",
        "icontains": "LIKE ? COLLATE NOCASE",
        "in": "IN ({})",
        "gt": "> ?",
        "gte": ">= ?",
        "lt": "< ?",
        "lte": "<= ?",
        "startswith": "LIKE ?",
        "istartswith": "LIKE ? COLLATE NOCASE",
        "endswith": "LIKE ?",
        "iendswith": "LIKE ? COLLATE NOCASE",
        "range": "BETWEEN ? AND ?",
        "isnull": "IS NULL",
    }

    def __init__(self, model_cls: Type["Model"]):
        super().__init__()
        if not hasattr(model_cls, '_fields'):
            raise TypeError("QuerySet must be initialized with a Model class.")
        self.model_cls = model_cls

        # For backward compatibility, use the model's db_alias if set as a class attribute
        model_db_alias = getattr(model_cls, "db_alias", None)
        if model_db_alias:
            self.db_alias = model_db_alias
        else:
            # Determine default alias using the router based on the model's app label
            app_label = self.model_cls.get_app_label()
            self.db_alias = DATABASES.router.db_for_app(app_label)

        self.filters = []
        self.excludes = []
        self.ordering: Optional[List[str]] = None
        self.limit_value: Optional[int] = None
        self.offset_value: Optional[int] = None
        self._last_method = None  # Initialize _last_method attribute

    def _clone(self):
        """Create a copy of the current queryset."""
        clone = self.__class__(self.model_cls)
        clone.db_alias = self.db_alias
        clone.filters = self.filters.copy()
        clone.ordering = self.ordering
        clone.limit_value = self.limit_value
        clone._last_method = self._last_method
        return clone

    def _convert_enum_values(self, data: dict) -> dict:
        """Convert any Enum values in a dict to their string values."""
        converted = {}
        for key, value in data.items():
            if isinstance(value, Enum):
                converted[key] = value.value
            else:
                converted[key] = value
        return converted

    def using(self, db_alias: str) -> "QuerySet":
        """Set the database alias to be used for this query chain."""
        if not db_alias: raise ValueError("Database alias cannot be empty.")

        new_qs = self._clone()
        new_qs.db_alias = db_alias
        return new_qs

    def filter(self, **kwargs):
        new_qs = self._clone()
        converted_kwargs = self._convert_enum_values(kwargs)
        new_qs.filters.append(converted_kwargs)
        new_qs._last_method = "filter"
        return new_qs

    def exclude(self, **kwargs):
        """Exclude records matching the given filters."""
        self.filters.append({"__exclude__": kwargs})
        self._last_method = "exclude"
        return self

    def order_by(self, field: str) -> "QuerySet":
        """Handle ordering, including JSON field ordering."""
        if field.startswith("-"):
            direction = "DESC"
            field = field[1:]
        else:
            direction = "ASC"

        # Handle JSON field ordering
        if "__" in field and field.split("__")[0] in getattr(
            self.model_cls, "_json_fields", []
        ):
            field_parts = field.split("__")
            json_field = field_parts[0]
            json_path = field_parts[1:]

            # Build the JSON path for SQLite
            json_path_str = f"$.{'.'.join(json_path)}"
            self.ordering = f"COALESCE(CAST(json_extract({json_field}, '{json_path_str}') AS TEXT), '') {direction}"
        else:
            self.ordering = f"{field} {direction}"

        self._last_method = "order_by"
        return self

    def limit(self, value: int):
        self.limit_value = value
        self._last_method = "limit"
        return self

    async def get(self, **kwargs):
        """Retrieve a single record matching the conditions."""
        if kwargs:
            self.filters.append(kwargs)
        results = await self._fetch_all()
        if len(results) > 1:
            self.filters = [f for f in self.filters if f != kwargs]
            raise ValueError(
                "get() returned more than one record. Expected exactly one."
            )
        elif len(results) == 0:
            self.filters = [f for f in self.filters if f != kwargs]
            raise self.model_cls.DoesNotExist(
                f"No {self.model_cls.__name__} found matching the given criteria."
            )
        return results[0]

    async def count(self) -> int:
        """Count the number of records matching the conditions."""
        db = await DATABASES.get_connection(self.db_alias)

        # Check if this is PostgreSQL database
        is_postgres = getattr(db, 'db_type', None) == DatabaseType.POSTGRES

        if is_postgres:
            # Get schema and table from _get_parsed_table_name
            schema, table = self.model_cls._get_parsed_table_name()
            # Use fully schema-qualified table name
            table_identifier = f'"{schema}"."{table}"'
        else:
            # For SQLite, use the regular table name
            table_name = self.model_cls.get_table_name()
            table_identifier = f'"{table_name}"'

        conditions, values, _ = self._build_conditions()

        # Create proper condition string based on DB type
        if conditions:
            condition_str = " AND ".join(conditions)
        else:
            # Use TRUE for PostgreSQL, 1 for SQLite
            condition_str = "TRUE" if is_postgres else "1"

        query = f"SELECT COUNT(*) as count FROM {table_identifier} WHERE {condition_str}"

        result = await db.fetch_one(query, tuple(values))
        return result["count"] if result else 0

    async def create(self, **kwargs):
        """Create a new record in the database by instantiating and saving the model."""
        if 'db_alias' not in kwargs:
            kwargs['db_alias'] = self.db_alias
        converted_kwargs = self._convert_enum_values(kwargs)
        instance = self.model_cls(**converted_kwargs)
        instance.db_alias = self.db_alias
        await instance.save(create=True)
        return instance

    async def bulk_create(self, objs: List[T]) -> List[T]:
        """Bulk create model instances."""
        db = await DATABASES.get_connection(self.db_alias)
        table_name = self.model_cls.get_table_name()

        if not objs:
            return []

        # Generate IDs for objects that don't have one
        for obj in objs:
            if not obj.id:
                obj.id = (
                    f"{self.model_cls.get_app_label()}-{self.model_cls._generate_id()}"
                )

        # Get all fields including id
        fields = list(self.model_cls._fields.keys())
        columns_str = ", ".join(fields)
        placeholders = ", ".join("?" for _ in fields)
        query = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"

        values = []
        for obj in objs:
            row_values = []
            for field_name in fields:
                field = self.model_cls._fields[field_name]
                value = getattr(obj, field_name, field.default)
                if callable(value):
                    value = value()
                db_value = field.to_db(value)
                row_values.append(db_value)
            values.append(tuple(row_values))

        try:
            await db.executemany(query, values)
            await db.commit()
            return objs
        except aiosqlite.OperationalError as e:
            raise e

    async def update(self, **kwargs) -> int:
        if not kwargs:
            return 0  # Nothing to update

        # Convert any Enum values
        kwargs = self._convert_enum_values(kwargs)

        db = await DATABASES.get_connection(self.db_alias)
        table_name = self.model_cls.get_table_name()

        # Build SET clause
        set_items = []
        update_values = []

        for field_name, value in kwargs.items():
            # Get the field instance if it exists in the model
            field = self.model_cls._fields.get(field_name)
            if field:
                # Convert value to database format if we have a field
                db_value = field.to_db(value)
            else:
                # Use value as-is if no field definition exists
                db_value = self._prepare_insert_value(value)

            set_items.append(f"{field_name} = ?")
            update_values.append(db_value)

        set_clause = ", ".join(set_items)

        # Build WHERE clause
        conditions, condition_values, _ = self._build_conditions()
        where_clause = " AND ".join(conditions) if conditions else "1"

        # Combine all values for the query
        all_values = tuple(update_values + condition_values)

        # Build and execute the update query
        query = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"

        try:
            cursor = await db.execute(query, all_values)
            await db.commit()
            return cursor.rowcount
        except Exception as e:
            raise Exception(f"Update failed: {str(e)}") from e

    async def bulk_update(self, objs: List[T], fields: List[str]) -> None:
        db = await DATABASES.get_connection(self.db_alias)
        is_postgres = getattr(db, 'db_type', None) == DatabaseType.POSTGRES

        # --- CORRECTED: Get schema-qualified table name for PG ---
        schema, table = self.model_cls._get_parsed_table_name()
        table_identifier = f'"{schema}"."{table}"' if is_postgres else f'"{self.model_cls.get_table_name()}"'
        # --- END CORRECTION ---

        if not objs or not fields:
            return

        # --- CORRECTED: Build SET clause with correct placeholders ---
        set_parts = []
        param_index = 1
        for field in fields:
            placeholder = f'${param_index}' if is_postgres else '?'
            set_parts.append(f'"{field}" = {placeholder}') # Quote field name
            param_index += 1
        set_clause = ", ".join(set_parts)
        pk_placeholder = f'${param_index}' if is_postgres else '?'
        # --- END CORRECTION ---

        # --- CORRECTED: Quote ID column ---
        query = f'UPDATE {table_identifier} SET {set_clause} WHERE "id" = {pk_placeholder}'
        # --- END CORRECTION ---

        values = []
        for obj in objs:
            row_values = [
                self.model_cls._fields[field].to_db(getattr(obj, field))
                for field in fields
            ]
            # Add PK value at the end
            row_values.append(obj.id)
            # --- CORRECTED: Use list for asyncpg, tuple for sqlite ---
            values.append(row_values if is_postgres else tuple(row_values))
            # --- END CORRECTION ---

        try:
            await db.executemany(query, values)
            # Commit is often handled by the DatabaseInfo/ConnectionManager now,
            # but explicitly committing here might be needed depending on transaction control.
            # If tests fail again due to transaction issues, uncomment below.
            # await db.commit()
        except Exception as e:
            self.model_cls.logger.error(f"Bulk update failed: {e}. Query: {query}", exc_info=True)
            raise e # Re-raise original error for clarity

    async def bulk_delete(self, objs: List[T]) -> None:
        db = await DATABASES.get_connection(self.db_alias)
        is_postgres = getattr(db, 'db_type', None) == DatabaseType.POSTGRES

        # --- CORRECTED: Get schema-qualified table name for PG ---
        schema, table = self.model_cls._get_parsed_table_name()
        table_identifier = f'"{schema}"."{table}"' if is_postgres else f'"{self.model_cls.get_table_name()}"'
        # --- END CORRECTION ---


        if not objs:
            return

        ids = [obj.id for obj in objs if obj.id is not None]
        if not ids:
            return

        # --- CORRECTED: Use correct placeholders and table identifier ---
        if is_postgres:
            placeholders = ", ".join(f"${i+1}" for i in range(len(ids)))
            query = f'DELETE FROM {table_identifier} WHERE "id" IN ({placeholders})' # Quote id column
            values_tuple = ids # asyncpg expects a list
        else:
            placeholders = ", ".join("?" for _ in ids)
            query = f'DELETE FROM {table_identifier} WHERE "id" IN ({placeholders})' # Quote id column
            values_tuple = tuple(ids) # aiosqlite expects a tuple
        # --- END CORRECTION ---

        try:
            # Use execute, not executemany for IN clause
            await db.execute(query, values_tuple)
            # Commit might be implicit depending on db_info implementation
            # await db.commit() # Uncomment if necessary
        except Exception as e:
             # Log error with details
             self.model_cls.logger.error(f"Bulk delete failed: {e}. Query: {query}, Values: {values_tuple}", exc_info=True)
             raise e

    def all(self) -> "QuerySet":
        """Return a new QuerySet that is a copy of the current one."""
        return self._clone()

    async def _fetch_all(self) -> List[T]:
        """Get all records that match the current filters."""
        db = await DATABASES.get_connection(self.db_alias)

        # Check if this is PostgreSQL database
        is_postgres = getattr(db, 'db_type', None) == DatabaseType.POSTGRES

        if is_postgres:
            # Get schema and table from _get_parsed_table_name
            schema, table = self.model_cls._get_parsed_table_name()
            # Use fully schema-qualified table name
            table_identifier = f'"{schema}"."{table}"'
        else:
            # For SQLite, use the regular table name
            table_name = self.model_cls.get_table_name()
            table_identifier = f'"{table_name}"'

        # Build query with proper table identifier
        sql_conditions, sql_values, json_filters = self._build_conditions()
        condition_str = " AND ".join(sql_conditions) if sql_conditions else "1"
        query = f'SELECT * FROM {table_identifier} WHERE {condition_str}'

        if self.ordering:
            query += f" ORDER BY {self.ordering}"

        if self.limit_value:
            query += f" LIMIT {self.limit_value}"

        try:
            rows = await db.fetch_all(query, tuple(sql_values))
        except Exception as e:
            print(f"Error in _fetch_all executing query: {query}")
            print(f"Values: {sql_values}")
            print(f"Database Alias: {self.db_alias}")
            print(f"Database Object Type: {type(db)}")
            raise e

        # Instantiate model objects from the rows
        records = [self.model_cls.from_row(row, db_alias=self.db_alias) for row in rows]

        # Process JSON filters (remains the same)
        for field_name, lookup_path, value, negate in json_filters:
            filtered_records = []
            for obj in records:
                if lookup_path == ["isnull"]:
                    # Special handling for isnull
                    json_value = getattr(obj, field_name, None)
                    if value:  # isnull=True
                        if json_value is None:
                            filtered_records.append(obj)
                    else:  # isnull=False
                        if json_value is not None:
                            filtered_records.append(obj)
                else:
                    # Apply regular json filter
                    if self._apply_json_filter(
                        obj, field_name, lookup_path, value, negate
                    ):
                        filtered_records.append(obj)

            records = filtered_records  # Update records with the filtered list

        return records

    async def first(self):
        await self.limit(1)
        results = await self._fetch_all()
        return results[0] if results else None

    async def last(self):
        qs = self.order_by("-id")
        await qs.limit(1)
        results = await qs._fetch_all()
        return results[0] if results else None

    async def vectors(self) -> List[Tuple[int, np.ndarray]]:
        """Retrieve vectors and corresponding IDs."""
        db = await DATABASES.get_connection(self.db_alias)
        table_name = self.model_cls.get_table_name()
        conditions, values, _ = self._build_conditions()
        condition_str = " AND ".join(conditions) if conditions else "1"
        query = f"SELECT id, vector FROM {table_name} WHERE {condition_str}"
        cursor = await db.execute(query, tuple(values))
        rows = await cursor.fetchall()
        return [
            (row["id"], np.frombuffer(row["vector"], dtype=np.float32)) for row in rows
        ]

    async def search(
        self,
        vector: np.ndarray,
        vectors: Optional[List[Tuple[int, np.ndarray]]] = None,
        limit: int = 10,
    ):
        if vectors is None:
            vectors = await self.vectors()
        if not vectors:
            return []

        ids, vector_list = zip(*vectors)
        vectors_array = np.array(vector_list)
        dot_product = np.dot(vectors_array, vector)
        norm_product = np.linalg.norm(vectors_array, axis=1) * np.linalg.norm(vector)
        cosine_similarities = dot_product / np.clip(norm_product, 1e-7, None)

        size = cosine_similarities.size
        actual_limit = min(limit, size)

        if size <= actual_limit:
            res = sorted(zip(ids, cosine_similarities), key=lambda x: -x[1])[
                :actual_limit
            ]

        else:
            best_idx = np.argpartition(-cosine_similarities, actual_limit)[
                :actual_limit
            ]
            best_idx = best_idx[np.argsort(-cosine_similarities[best_idx])]
            res = [(ids[i], cosine_similarities[i]) for i in best_idx]

        # Create a dictionary to store results with slugs as keys
        results_dict = {}
        for idx, score in res:
            obj = await self.model_cls.objects.using(self.db_alias).get(
                id=idx
            )  # Use slug here
            obj.p = score
            results_dict[idx] = obj  # Use slug as the key

        # Return a list of objects in the order of similarity
        return [results_dict[idx] for idx, _ in res]

    @staticmethod
    def cosine_distance(vectors: np.ndarray, query_vector: np.ndarray) -> np.ndarray:

        # Ensure query_vector is 1D and matches the number of features in vectors
        if query_vector.ndim != 1:
            query_vector = query_vector.ravel()

        if vectors.shape[1] != query_vector.shape[0]:
            raise ValueError(
                f"Dimension mismatch: vectors has shape {vectors.shape} but query_vector has length {query_vector.shape[0]}"
            )

        # Calculate dot product
        dot_product = np.dot(vectors, query_vector)

        # Calculate norms
        norm_vectors = np.linalg.norm(vectors, axis=1)
        norm_query = np.linalg.norm(query_vector)

        # Compute norm product with epsilon to avoid division by zero
        eps = np.finfo(np.float32).eps
        norm_product = norm_vectors * norm_query
        norm_product = np.where(norm_product < eps, eps, norm_product)

        # Calculate cosine distance
        cosine_similarity = dot_product / norm_product

        # Clip to handle floating point errors that might push similarity slightly over 1
        cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)

        return 1 - cosine_similarity

    def _process_filters(self, filters: dict, negate: bool = False):
        sql_conditions = []
        sql_values = []
        json_filters = []

        # Determine if the connection for the current db_alias is PostgreSQL
        is_postgres = False
        # Check DATABASES existence and router attribute first
        if hasattr(DATABASES, 'CONNECTIONS') and hasattr(DATABASES, 'router'):
            # Use the router to find the target alias for the model's app label
            target_alias = DATABASES.router.db_for_app(self.model_cls.get_app_label())
            db_info = DATABASES.CONNECTIONS.get(target_alias)  # Check connection using the target alias
            if db_info and getattr(db_info, 'db_type', None) == DatabaseType.POSTGRES:
                is_postgres = True
        # Note: This check assumes a connection might already exist.
        # A more robust check might involve looking at the CONFIG for the target_alias
        # if no connection is present yet, but that adds complexity.

        # Convert any Enum values in filters first
        processed_filters = {}
        for key, value in filters.items():
            if isinstance(value, Enum):
                processed_filters[key] = value.value
            else:
                processed_filters[key] = value

        # Now process the converted filters
        for key, value in processed_filters.items():
            parts = key.split("__")
            field_name = parts[0]
            # Ensure field name is quoted for safety, especially with reserved words
            quoted_field_name = f'"{field_name}"'

            lookup_type = parts[-1] if len(parts) > 1 else "exact"

            # Special handling for datetime fields
            field = self.model_cls._fields.get(field_name)
            if isinstance(field, DateTimeField):
                if isinstance(value, datetime.datetime):
                    value = value.isoformat()  # Convert datetime to ISO string for DB

            if field_name in getattr(self.model_cls, "_json_fields", []) and len(parts) > 1:
                if is_postgres:
                    # PostgreSQL JSON handling
                    if lookup_type == "contains":
                        # Remove quotes if string is quoted (PostgreSQL JSONB contains works better without outer quotes for simple strings)
                        if isinstance(value, str) and len(value) >= 2 and value.startswith('"') and value.endswith('"'):
                            value_to_check = value[1:-1]
                        else:
                            value_to_check = value
                        # Use PostgreSQL containment operator '@>' with a single placeholder
                        # We need to wrap the value in JSON structure for the operator
                        sql_conditions.append(f"{quoted_field_name}::jsonb @> ?::jsonb")
                        # Handle different value types correctly for JSON contains
                        if isinstance(value_to_check, (dict, list)):
                            sql_values.append(json.dumps(value_to_check))
                        elif isinstance(value_to_check, str):
                            # For string containment, check if the string exists as a value or key
                            # A simple string value check needs to be wrapped in JSON array/object
                            # This example assumes checking if the string exists as a top-level value in an array or dict value
                            # More complex logic might be needed for nested checks via SQL only
                            sql_values.append(json.dumps([value_to_check]))  # Check if string is in top-level array
                            # Or potentially: sql_values.append(f'"{value_to_check}"') # If checking for exact JSON string value
                        else:
                            # For numbers or booleans, wrap them appropriately
                            sql_values.append(json.dumps(value_to_check))

                        continue  # Skip adding to Python-based json_filters

                    elif lookup_type == "has":
                        # Use PostgreSQL has key operator '?' with a single placeholder
                        sql_conditions.append(f"{quoted_field_name}::jsonb ? ?")  # Generate SQL with '?'
                        sql_values.append(str(value))  # The key must be a string
                        continue  # Skip adding to Python-based json_filters

                # Default behavior for SQLite or other unhandled PostgreSQL cases (Python-based filtering)
                json_filters.append((field_name, parts[1:], value, negate))
            else:
                # Handle regular field lookups
                if lookup_type in [
                    "contains",
                    "icontains",
                    "startswith",
                    "istartswith",
                    "endswith",
                    "iendswith",
                ]:
                    # Prepare value for LIKE operator
                    # Note: icontains/istartswith/iendswith might need specific handling for PostgreSQL (ILIKE)
                    # vs SQLite (COLLATE NOCASE)
                    pg_like_operator = "LIKE"
                    sqlite_like_operator = "LIKE"
                    collate_nocase = ""  # For SQLite

                    if lookup_type in ["icontains", "istartswith", "iendswith"]:
                        pg_like_operator = "ILIKE"  # Use ILIKE for case-insensitive in PG
                        collate_nocase = " COLLATE NOCASE"  # Use COLLATE for SQLite

                    if lookup_type in ["contains", "icontains"]:
                        value = f"%{value}%"
                    elif lookup_type in ["startswith", "istartswith"]:
                        value = f"{value}%"
                    elif lookup_type in ["endswith", "iendswith"]:
                        value = f"%{value}"

                    # Choose operator based on DB type
                    chosen_like_operator = pg_like_operator if is_postgres else sqlite_like_operator
                    chosen_collate = "" if is_postgres else collate_nocase
                    operator_sql = f"{chosen_like_operator} ?{chosen_collate}"

                    if negate:
                        operator_sql = f"NOT {operator_sql}"

                    sql_conditions.append(f"{quoted_field_name} {operator_sql}")
                    sql_values.append(value)

                elif lookup_type == "in":
                    # Handle Enum values in lists
                    processed_value = [
                        v.value if isinstance(v, Enum) else v for v in value
                    ]
                    if not processed_value:  # Handle empty list case
                        # WHERE id IN () is invalid SQL, effectively means "match nothing"
                        # Add a condition that is always false
                        sql_conditions.append("1 = 0")
                    else:
                        placeholders = ", ".join(["?"] * len(processed_value))
                        operator = self.LOOKUP_OPERATORS["in"].format(placeholders)
                        if negate:
                            operator = f"NOT {operator}"
                        sql_conditions.append(f"{quoted_field_name} {operator}")  # Use quoted field name
                        sql_values.extend(processed_value)

                elif lookup_type == "isnull":
                    operator = "IS NULL" if value else "IS NOT NULL"
                    if negate:
                        operator = "IS NOT NULL" if value else "IS NULL"
                    sql_conditions.append(f"{quoted_field_name} {operator}")  # Use quoted field name
                    # No value needed for IS NULL / IS NOT NULL

                else:
                    # Handle exact, gt, gte, lt, lte etc.
                    base_operator = self.LOOKUP_OPERATORS.get(lookup_type)
                    if not base_operator:
                        # If lookup_type isn't in operators, assume it's part of the field name (e.g., json lookup fallback)
                        # This path shouldn't be hit if JSON fields are handled above, but as a safety measure:
                        print(
                            f"Warning: Unrecognized lookup type '{lookup_type}' for non-JSON field '{field_name}'. Treating as exact match.")
                        base_operator = "= ?"  # Default to exact match

                    # Apply negation if needed
                    if negate:
                        if base_operator == "= ?":
                            base_operator = "!= ?"
                        elif base_operator == "IS NULL":
                            base_operator = "IS NOT NULL"
                        # Add more negation logic if necessary (e.g., NOT LIKE, NOT BETWEEN)
                        else:
                            # Generic negation (might not work for all operators, e.g. LIKE)
                            base_operator = f"NOT ({base_operator})"

                    sql_conditions.append(f"{quoted_field_name} {base_operator}")  # Use quoted field name
                    sql_values.append(value)

        return sql_conditions, sql_values, json_filters

    def _build_conditions(self):
        sql_conditions = []
        sql_values = []
        json_filters = []

        for filter_set in self.filters:
            if "__exclude__" in filter_set:
                excl_conditions, excl_values, excl_json_filters = self._process_filters(
                    filter_set["__exclude__"], negate=True
                )
                sql_conditions.extend(excl_conditions)
                sql_values.extend(excl_values)
                json_filters.extend(excl_json_filters)
            else:
                incl_conditions, incl_values, incl_json_filters = self._process_filters(
                    filter_set
                )
                sql_conditions.extend(incl_conditions)
                sql_values.extend(incl_values)
                json_filters.extend(incl_json_filters)

        return sql_conditions, sql_values, json_filters

    def _apply_lookup(self, data, lookup_type, value):
        if lookup_type == "exact":
            return data == value
        elif lookup_type == "iexact":
            return str(data).lower() == str(value).lower()
        elif lookup_type == "contains":
            return value in data
        elif lookup_type == "icontains":
            return str(value).lower() in str(data).lower()
        elif lookup_type == "in":
            return data in value
        elif lookup_type == "gt":
            return data > value
        elif lookup_type == "gte":
            return data >= value
        elif lookup_type == "lt":
            return data < value
        elif lookup_type == "lte":
            return data <= value
        elif lookup_type == "startswith":
            return str(data).startswith(str(value))
        elif lookup_type == "istartswith":
            return str(data).lower().startswith(str(value).lower())
        elif lookup_type == "endswith":
            return str(data).endswith(str(value))
        elif lookup_type == "iendswith":
            return str(data).lower().endswith(str(value).lower())
        elif lookup_type == "range":
            return value[0] <= data <= value[1]
        elif lookup_type == "isnull":
            return data is None
        # Add other lookup types as needed
        else:
            raise ValueError(f"Unsupported lookup type: {lookup_type}")

    def _apply_json_filter(self, obj, field_name, lookup_path, value, negate=False):

        json_value = getattr(obj, field_name, None)
        if json_value is None:
            return negate

        current_value = json_value
        # Traverse the lookup path
        for i, key in enumerate(lookup_path[:-1]):
            if isinstance(current_value, dict) and key in current_value:
                current_value = current_value[key]
            elif isinstance(current_value, list) and key.isdigit():
                try:
                    current_value = current_value[int(key)]
                except IndexError:
                    return False
            else:
                return False

        # Check if the last part of the lookup path is a valid operator
        lookup_type = lookup_path[-1]

        if lookup_type == "has":
            result = value in current_value
        elif lookup_type in self.LOOKUP_OPERATORS:
            if lookup_type == "exact":
                result = current_value == value
            elif lookup_type == "iexact":
                result = str(current_value).lower() == str(value).lower()
            elif lookup_type == "contains":
                if isinstance(current_value, (str, list, dict)):
                    if isinstance(current_value, str):
                        result = value in current_value
                    else:
                        json_str = json.dumps(current_value)
                        result = value in json_str
                else:
                    result = False
            elif lookup_type == "icontains":
                if isinstance(current_value, (str, list, dict)):
                    if isinstance(current_value, str):
                        result = str(value).lower() in str(current_value).lower()
                    else:
                        json_str = json.dumps(current_value)
                        result = str(value).lower() in json_str.lower()
                else:
                    result = False
            elif lookup_type == "in":
                result = current_value in value
            elif lookup_type == "gt":
                result = current_value > value
            elif lookup_type == "gte":
                result = current_value >= value
            elif lookup_type == "lt":
                result = current_value < value
            elif lookup_type == "lte":
                result = current_value <= value
            elif lookup_type == "startswith":
                result = str(current_value).startswith(str(value))
            elif lookup_type == "istartswith":
                result = str(current_value).lower().startswith(str(value).lower())
            elif lookup_type == "endswith":
                result = str(current_value).endswith(str(value))
            elif lookup_type == "iendswith":
                result = str(current_value).lower().endswith(str(value).lower())
            elif lookup_type == "range":
                result = value[0] <= current_value <= value[1]
            elif lookup_type == "isnull":
                result = current_value is None
            else:
                raise ValueError(f"Unsupported lookup type: {lookup_type}")
        else:
            if isinstance(current_value, dict) and lookup_path[-1] in current_value:
                result = current_value[lookup_path[-1]] == value
            elif isinstance(current_value, list) and lookup_path[-1].isdigit():
                try:
                    result = current_value[int(lookup_path[-1])] == value
                except IndexError:
                    return False
            else:
                return False

        if negate:
            return not result
        return result

    def _build_insert_fields(self, kwargs):
        fields = []
        values = []
        placeholders = []
        for key, value in kwargs.items():
            fields.append(key)
            placeholders.append("?")
            values.append(self._prepare_insert_value(value))
        return fields, values, placeholders

    def _prepare_insert_value(self, value):
        if isinstance(value, Enum):
            return value.value  # Handle Enum values
        elif isinstance(value, (dict, list)):
            return json.dumps(value)
        elif isinstance(value, bool):
            return int(value)
        elif isinstance(value, datetime.datetime):
            return value.isoformat()
        elif value is None:
            return None
        return value

    async def exists(self) -> bool:
        """Check if any records matching the conditions exist."""
        db = await DATABASES.get_connection(self.db_alias)
        table_name = self.model_cls.get_table_name()
        conditions, values, _ = self._build_conditions()
        condition_str = " AND ".join(conditions) if conditions else "1"
        query = f"SELECT EXISTS(SELECT 1 FROM {table_name} WHERE {condition_str})"

        cursor = await db.execute(query, tuple(values))
        row = await cursor.fetchone()
        return bool(row[0]) if row else False

    async def delete(self) -> int:
        """Delete all records matching the current filters and return the count of deleted records."""
        db = await DATABASES.get_connection(self.db_alias)
        table_name = self.model_cls.get_table_name()
        conditions, values, _ = self._build_conditions()
        condition_str = " AND ".join(conditions) if conditions else "1"

        query = f"DELETE FROM {table_name} WHERE {condition_str}"

        cursor = await db.execute(query, tuple(values))
        await db.commit()

        return cursor.rowcount
