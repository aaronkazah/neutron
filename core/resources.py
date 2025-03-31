from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Type, ClassVar, Any
import re
import json
from abc import ABC
from inflection import pluralize
from dataclasses import dataclass, field


class ResourceError(Exception):
    def __init__(self, response_data: Dict, status_code: int):
        self.response = response_data
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {response_data}")


@dataclass
class ResourceData:
    """Holds the data for a resource instance"""

    _data: Dict = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        """Allow accessing data attributes directly"""
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any):
        """Allow setting data attributes directly"""
        if name == "_data":
            super().__setattr__(name, value)
        else:
            self._data[name] = value

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, key, value):
        self._data[key] = value

    def items(self):
        return self._data.items()


@dataclass
class QueryState:
    """Holds the state for a query"""

    filters: Dict = field(default_factory=dict)
    ordering: Optional[str] = None
    limit: Optional[int] = None
    page: int = 1
    page_size: int = 10


class ResourceQuerySet:
    """Handles query building and state management"""

    def __init__(self, resource_class: Type["Resource"]):
        self.resource_class = resource_class
        self.state = QueryState()
        self._results: Optional[list[ResourceData]] = None

    def filter(self, **kwargs) -> "ResourceQuerySet":
        """Add filters to the query."""
        self.state.filters.update(kwargs)
        return self

    def exclude(self, **kwargs) -> "ResourceQuerySet":
        """Exclude items matching the filters."""
        for key, value in kwargs.items():
            self.state.filters[f"not_{key}"] = value
        return self

    def order_by(self, field: str) -> "ResourceQuerySet":
        """Set the ordering field."""
        self.state.ordering = field
        return self

    def page(self, page: int) -> "ResourceQuerySet":
        """Set the page number."""
        self.state.page = page
        return self

    def page_size(self, size: int) -> "ResourceQuerySet":
        """Set the page size."""
        self.state.page_size = size
        return self

    def limit(self, value: int) -> "ResourceQuerySet":
        """Set the limit for results."""
        self.state.limit = value
        return self

    def _build_params(self) -> Dict:
        """Build query parameters from the current state."""
        params = {"page": self.state.page, "page_size": self.state.page_size}

        if self.state.filters:
            params["filters"] = json.dumps(self.state.filters)

        if self.state.ordering:
            if self.state.ordering.startswith("-"):
                params["ordering"] = self.state.ordering[1:]
                params["order_direction"] = "DESC"
            else:
                params["ordering"] = self.state.ordering
                params["order_direction"] = "ASC"

        if self.state.limit:
            params["limit"] = self.state.limit

        return params

    async def update(self, **kwargs) -> Any:
        """
        Update all matching resources with the given values.
        Returns either a single Resource object or an array of Resource objects.
        """
        results = await self.all()
        resources = results.get("data", [])

        if not resources:
            return []

        updated_resources = []
        for resource in resources:
            try:
                response = await self.resource_class.client.request(
                    "PATCH",
                    f"{self.resource_class.endpoint}/{resource.id}",
                    json=kwargs,
                )
                if response.status_code < 400:
                    # Create a new Resource instance with the response data
                    updated_resource = self.resource_class(data=response.json())
                    updated_resources.append(updated_resource)
            except Exception:
                continue

        # Return single Resource object if only one resource was updated
        if len(updated_resources) == 1:
            return updated_resources[0]

        # Return array of Resource objects if multiple resources were updated
        return updated_resources

    async def delete(self) -> Dict[str, Any]:
        """
        Delete all matching resources.
        Raises ResourceError if any deletions fail.
        Returns deletion summary if all succeed.
        """
        results = await self.all()
        resources = results.get("data", [])

        if not resources:
            return {"deleted": [], "object": "list"}

        deleted_ids = []
        errors = []

        for resource in resources:
            try:
                response = await self.resource_class.client.request(
                    "DELETE", f"{self.resource_class.endpoint}/{resource.id}"
                )

                if response.status_code < 400:
                    deleted_ids.append(resource.id)
                else:
                    errors.append(
                        {
                            "id": resource.id,
                            "error": response.json()
                            .get("error", {})
                            .get("message", "Unknown error"),
                            "status": response.status_code,
                        }
                    )

            except Exception as e:
                errors.append({"id": resource.id, "error": str(e), "status": 500})

        # If there were any errors, raise ResourceError
        if errors:
            raise ResourceError(
                {
                    "error": {
                        "message": "Failed to delete all resources",
                        "errors": errors,
                        "partial_deletes": deleted_ids,
                    }
                },
                status_code=400,
            )

        # All deletions successful
        return {"object": "list", "deleted": deleted_ids}

    async def first(self) -> Optional["Resource"]:
        """Get the first matching resource."""
        self.limit(1)
        results = await self.all()
        resources = results.get("data", [])
        return resources[0] if resources else None

    async def last(self) -> Optional[ResourceData]:
        """Get the last matching resource."""
        if not self.state.ordering:
            self.order_by("-id")
        self.limit(1)
        results = await self.all()
        resources = results.get("data", [])
        return resources[0] if resources else None

    async def retrieve(self, id: str) -> "Resource":
        """
        Retrieve a single resource by ID.
        Raises ResourceError with 404 if not found.
        """
        if id is None:
            raise ResourceError({"error": {"message": "ID is required"}}, 400)

        # Make direct request for the specific ID
        response = await self.resource_class.client.request(
            "GET", f"{self.resource_class.endpoint}/{id}", params=self._build_params()
        )

        if response.status_code >= 400:
            raise ResourceError(
                {"error": {"message": f"{self.resource_class.__name__} not found"}},
                status_code=404,
            )

        return self.resource_class(data=response.json())

    async def all(self, **params):
        """Get all results matching the current query."""
        if params is None:
            params = {}
        build_params = self._build_params()
        build_params.update(params)
        response = await self.resource_class.client.request(
            "GET", self.resource_class.endpoint, params=build_params
        )
        if response.status_code >= 400:
            raise ResourceError(response.json(), response.status_code)

        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Invalid response format")

        items = data.get("data", [])
        if not isinstance(items, list):
            items = [items]

        items = [self.resource_class(data=item) for item in items]
        data["data"] = items
        return data


class Resource(ABC):
    """Base class for making API calls with query-like interface."""

    client: ClassVar[Optional["APIClient"]] = None
    endpoint: ClassVar[Optional[str]] = None
    prefix: ClassVar[Optional[str]] = None

    def __repr__(self):
        data = self.serialize()
        fields = ", ".join(f"{k}={v}" for k, v in data.items())
        return f"<Resource({fields})>"

    def __init__(self, data: Dict = None):
        """Initialize a resource instance with data"""
        self._data = ResourceData(_data=data or {})

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the ResourceData instance"""
        if hasattr(self, "_data") and self._data is not None:
            try:
                return getattr(self._data, name)
            except AttributeError:
                pass
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    # Instance methods

    async def save(self, **kwargs) -> "Resource":
        data = self.serialize()
        return await self.update(**data)

    async def update(self, **kwargs) -> "Resource":
        """Update this resource instance."""
        if not hasattr(self, "id"):
            raise ValueError("Cannot update a resource without an id")

        update_data = {}

        update_data.update(kwargs)

        response = await self.client.request(
            "PATCH", f"{self.endpoint}/{self.id}", json=update_data
        )
        response_data = response.json()

        if response.status_code >= 400:
            raise ResourceError(response_data, response.status_code)

        self._data = ResourceData(_data=response_data)
        return self

    @classmethod
    async def create(cls, **kwargs) -> "Resource":
        """Create a new resource."""
        create_data = {}

        if kwargs:
            create_data.update(kwargs)

        response = await cls.client.request("POST", cls.endpoint, json=create_data)
        response_data = response.json()

        if response.status_code >= 400:
            raise ResourceError(response_data, response.status_code)

        # Remove .get("data", {})
        return cls(data=response_data)

    @classmethod
    async def retrieve(cls, id: str = None, **kwargs) -> "Resource":
        """Read (retrieve) a single resource by ID or filters."""
        if id is not None:
            endpoint = f"{cls.endpoint}/{id}"
            response = await cls.client.request("GET", endpoint)
            response_data = response.json()

            if response.status_code >= 400:
                raise ResourceError(response_data, response.status_code)

            return cls(data=response_data)

        queryset = ResourceQuerySet(cls)
        if kwargs:
            queryset.filter(**kwargs)
        resource_data = await queryset.first()
        if not resource_data:
            raise ResourceError({"error": "Not found"}, 404)
        return resource_data

    @classmethod
    async def patch(cls, id: str, **kwargs) -> "Resource":
        """Update a resource by ID."""
        update_data = {}

        if kwargs:
            update_data.update(kwargs)

        response = await cls.client.request(
            "PATCH", f"{cls.endpoint}/{id}", json=update_data
        )
        response_data = response.json()

        if response.status_code >= 400:
            raise ResourceError(response_data, response.status_code)

        # Remove .get("data", {})
        return cls(data=response_data)

    @classmethod
    async def put(cls, id: str, data: Dict = None, **kwargs) -> "Resource":
        """Update a resource by ID."""
        update_data = {}
        if data is not None:
            update_data.update(data if isinstance(data, dict) else {"data": data})
        if kwargs:
            update_data.update(kwargs)

        response = await cls.client.request(
            "PUT", f"{cls.endpoint}/{id}", json=update_data
        )
        response_data = response.json()

        if response.status_code >= 400:
            raise ResourceError(response_data, response.status_code)

        # Remove .get("data", {})
        return cls(data=response_data)

    async def refresh_from_api(self) -> "Resource":
        """Refresh this resource instance data from the API."""
        if not hasattr(self, "id"):
            raise ValueError("Cannot refresh a resource without an id")

        response = await self.client.request("GET", f"{self.endpoint}/{self.id}")
        fresh_data = response.json()
        self._data = ResourceData(_data=fresh_data)
        return self

    @classmethod
    async def destroy(cls, id: str) -> None:
        """Delete a resource by ID."""
        response = await cls.client.request("DELETE", f"{cls.endpoint}/{id}")
        response_data = response.json()

        if response.status_code >= 400:
            raise ResourceError(response_data, response.status_code)

    async def delete(self) -> None:
        """Delete this resource instance."""
        if not hasattr(self, "id"):
            raise ValueError("Cannot delete a resource without an id")

        response = await self.client.request("DELETE", f"{self.endpoint}/{self.id}")
        response_data = response.json()

        if response.status_code >= 400:
            raise ResourceError(response_data, response.status_code)
        return None

    @classmethod
    async def list(cls):
        """List all resources."""
        queryset = ResourceQuerySet(cls)
        resource_data = await queryset.all()
        return resource_data

    # Query methods
    @classmethod
    def filter(cls, **kwargs) -> ResourceQuerySet:
        """Start a new query with filters."""
        queryset = ResourceQuerySet(cls)
        queryset.state.filters.update(kwargs)
        return queryset

    @classmethod
    def exclude(cls, **kwargs) -> ResourceQuerySet:
        """Start a new query with excludes."""
        return ResourceQuerySet(cls).exclude(**kwargs)

    @classmethod
    def order_by(cls, field: str) -> ResourceQuerySet:
        """Start a new query with ordering."""
        return ResourceQuerySet(cls).order_by(field)

    @classmethod
    def page(cls, page: int) -> ResourceQuerySet:
        """Start a new query with page number."""
        return ResourceQuerySet(cls).page(page)

    @classmethod
    def page_size(cls, size: int) -> ResourceQuerySet:
        """Start a new query with page size."""
        return ResourceQuerySet(cls).page_size(size)

    @staticmethod
    def _generate_endpoint(resource_name: str) -> str:
        """
        Converts PascalCase to plural kebab-case for API endpoints.
        Example: CustomerPayment -> customer-payments
        """
        words = re.findall("[A-Z][^A-Z]*", resource_name)
        kebab_case = "-".join(word.lower() for word in words)
        return pluralize(kebab_case)

    # Serialization methods
    def serialize(
        self,
        fields: Optional[Dict[str, Dict[str, Any]]] = None,
        exclude: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Serialize the resource's data according to the field configurations.
        If no fields or exclude parameters are provided, all fields are serialized.
        """
        if fields is not None and exclude is not None:
            raise ValueError("Cannot specify both 'fields' and 'exclude' parameters")

        # If no fields specified, serialize all fields
        if fields is None:
            result = {}
            for field_name, value in self._data.items():
                # Skip excluded fields
                if exclude and field_name in exclude:
                    continue

                # Serialize the value based on its type
                if isinstance(value, datetime):
                    result[field_name] = value.strftime("%Y-%m-%d %H:%M:%S")
                elif isinstance(value, Enum):
                    result[field_name] = value.value
                elif isinstance(value, dict):
                    result[field_name] = self._serialize_dict(value)
                elif isinstance(value, list):
                    result[field_name] = [self._serialize_value(item) for item in value]
                elif isinstance(value, (bool, int, float, str)):
                    result[field_name] = value
                else:
                    result[field_name] = str(value)
            return result

        # If specific fields are provided
        result = {}
        for field_name, field_config in fields.items():
            if field_name not in self._data:
                continue

            value = self._data[field_name]
            field_type = field_config.get("type", "str")

            if value is None:
                result[field_name] = None
                continue

            if field_type == "datetime":
                date_format = field_config.get("format", "%Y-%m-%d %H:%M:%S")
                if isinstance(value, datetime):
                    result[field_name] = value.strftime(date_format)
                else:
                    try:
                        parsed_date = datetime.strptime(str(value), date_format)
                        result[field_name] = parsed_date.strftime(date_format)
                    except ValueError:
                        result[field_name] = str(value)
            elif field_type == "enum":
                result[field_name] = (
                    value.value if isinstance(value, Enum) else str(value)
                )
            elif field_type == "dict":
                result[field_name] = (
                    self._serialize_dict(value) if isinstance(value, dict) else {}
                )
            elif field_type == "list":
                result[field_name] = (
                    [self._serialize_value(item) for item in value]
                    if isinstance(value, list)
                    else []
                )
            elif field_type == "bool":
                result[field_name] = bool(value)
            elif field_type == "int":
                try:
                    result[field_name] = int(value)
                except (ValueError, TypeError):
                    result[field_name] = 0
            elif field_type == "float":
                try:
                    result[field_name] = float(value)
                except (ValueError, TypeError):
                    result[field_name] = 0.0
            else:
                result[field_name] = str(value)

        return result

    def _serialize_dict(self, data: Dict) -> Dict:
        """Helper method to serialize dictionary values"""
        result = {}
        for key, value in data.items():
            result[key] = self._serialize_value(value)
        return result

    def _serialize_value(self, value: Any) -> Any:
        """Helper method to serialize individual values"""
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(value, Enum):
            return value.value
        elif isinstance(value, dict):
            return self._serialize_dict(value)
        elif isinstance(value, list):
            return [self._serialize_value(item) for item in value]
        elif isinstance(value, (bool, int, float, str)):
            return value
        else:
            return str(value)
