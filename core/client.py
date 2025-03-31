from datetime import datetime
from enum import Enum
from typing import List, Set, Any, Union


from core.resources import ResourceError

from typing import Dict, Optional, Type
import httpx
import re
from core.resources import Resource


class InvalidResourceNameError(Exception):
    """Raised when a resource name doesn't follow PascalCase convention."""

    def __init__(self, resource_name: str):
        super().__init__(
            f"Invalid resource name '{resource_name}'. "
            f"Resource names must be in PascalCase (e.g., 'Customer', 'PaymentIntent')"
        )


class APIClient:
    """Main API client class that supports dynamic resource access using Resource class."""

    _shared_client: Optional[httpx.AsyncClient] = None
    _local_client: Optional[httpx.AsyncClient] = None

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        token: Optional[str] = None,
        prefix: Optional[str] = None,
        resources: List[str] = None,
        timeout: int = 30,
        pluralize=True,
        resource_methods: Dict[str, Dict[str, dict]] = None,
        force_local=False,
        headers: Optional[Dict[str, str]] = None,
    ):
        """Initialize API client with configuration."""
        self.base_url = base_url.rstrip("/")
        self.prefix = prefix
        self._api_key = api_key
        self._token = token
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._resource_name_pattern = re.compile(r"^[A-Z][a-zA-Z0-9]*$")
        self._resource_classes: Dict[str, Type[Resource]] = {}
        self._allowed_resources: Set[str] = set(resources) if resources else set()
        self.pluralize = pluralize
        self._resource_methods = resource_methods or {}
        self.force_local = force_local
        self.headers = headers or {}

        # Validate resource names at initialization
        for resource in self._allowed_resources:
            if not self._resource_name_pattern.match(resource):
                raise InvalidResourceNameError(resource)

        # Pre-create resource classes
        for resource in self._allowed_resources:
            self._create_resource_class(resource)

    @classmethod
    def set_local_client(cls, client: httpx.AsyncClient) -> None:
        """Set the local client instance."""
        cls._local_client = client

    @classmethod
    def get_local_client(cls) -> Optional[httpx.AsyncClient]:
        """Get the local client instance."""
        return cls._local_client

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if (
            self.base_url == "http://localhost:7777" or self.force_local
        ) and self._local_client:
            client = self._local_client
        elif self._client is not None:
            client = self._client
        else:
            client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
            self._client = client

        return client

    def _serialize_request_data(
        self, data: Dict, is_multipart: bool = False
    ) -> Union[Dict, Dict[str, tuple]]:
        """Serialize request data based on whether it's multipart or JSON."""
        if not data:
            return data

        if is_multipart:
            # Handle files separately for multipart requests
            files = data.pop("files", {})
            # Serialize remaining data
            form_data = {
                key: str(self._serialize_value(value)) for key, value in data.items()
            }
            # Add files back
            form_data.update(files)
            return form_data

        return {key: self._serialize_value(value) for key, value in data.items()}

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a single value based on its type."""
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(item) for item in value]
        return value

    def _create_resource_class(self, name: str) -> Type[Resource]:
        """Create a new resource class with custom methods."""
        if name not in self._resource_classes:
            custom_methods = self._resource_methods.get(name, {})
            method_dict = {
                "__abstractmethods__": set(),
                "client": self,
                "endpoint": (
                    Resource._generate_endpoint(name)
                    if self.pluralize
                    else name.lower()
                ),
                "prefix": self.prefix,
            }

            for method_name, method_config in custom_methods.items():
                method_dict[method_name] = self._create_custom_method(method_config)

            resource_class = type(name, (Resource,), method_dict)
            self._resource_classes[name] = resource_class

        return self._resource_classes[name]

    def _create_custom_method(self, config: dict):
        """Create a custom method based on configuration."""

        # Parse path template to get required parameters
        path_template = config.get("path", "/{id}")
        required_params = set()

        # Find all parameters in format /<type:name>/
        param_pattern = r"<(\w+):(\w+)>"
        path_params = re.findall(param_pattern, path_template)

        # Convert to actual path format and track required params
        for param_type, param_name in path_params:
            required_params.add(param_name)
            # Replace <type:name> with {name}
            path_template = path_template.replace(
                f"<{param_type}:{param_name}>", "{" + param_name + "}"
            )

        async def custom_method(cls, **kwargs):
            # Validate required parameters
            missing_params = required_params - set(kwargs.keys())
            if missing_params:
                raise ValueError(
                    f"Missing required parameters for {config['name']}: {missing_params}"
                )

            method = config.get("method", "POST")

            # Format path with provided parameters
            path = path_template.format(**kwargs)
            endpoint = f"{cls.endpoint}{path}"

            # Remove used path parameters from kwargs
            request_data = {k: v for k, v in kwargs.items() if k not in required_params}

            # Handle files if present
            has_files = "files" in request_data
            if has_files:
                files = request_data.pop("files", {})
                data = self._serialize_request_data(request_data, is_multipart=True)
                response = await cls.client.request(
                    method,
                    endpoint,
                    files=files,
                    data=data,
                    headers=cls.client._get_headers(),  # Use instance headers
                )
            else:
                response = await cls.client.request(
                    method,
                    endpoint,
                    json=self._serialize_request_data(request_data),
                    headers=cls.client._get_headers(),  # Use instance headers
                )

            if response.status_code >= 400:
                error_data = (
                    response.json()
                    if response.content
                    else {
                        "error": {
                            "message": "Unknown error",
                            "type": "internal_server_error",
                        }
                    }
                )
                raise ResourceError(error_data, status_code=response.status_code)

            return response.json() if response.content else None

        custom_method.__name__ = config.get("name", "custom_method")
        custom_method.__doc__ = config.get("description", "")
        return classmethod(custom_method)

    def __getattr__(self, name: str) -> Type[Resource]:
        """Return a Resource class for the given name."""
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

        if not self._resource_name_pattern.match(name):
            raise InvalidResourceNameError(name)

        if self._allowed_resources and name not in self._allowed_resources:
            raise ValueError(
                f"Resource '{name}' not available. "
                f"Available resources: {', '.join(sorted(self._allowed_resources))}"
            )

        return self._create_resource_class(name)

    def _get_headers(self) -> Dict[str, str]:
        """Get the headers for requests."""
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        # Add custom headers
        headers.update(self.headers)

        return headers

    def set_headers(self, headers: Dict[str, str]) -> None:
        """Set or update custom headers."""
        self.headers.update(headers)

    async def request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        json: Optional[Dict] = None,
        data: Optional[Dict] = None,
        files: Optional[Dict] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> httpx.Response:
        """Make an HTTP request to the API."""
        url = f"{path}"
        if self.prefix:
            url = f"{self.prefix.rstrip('/')}/{url.lstrip('/')}"

        kwargs = {"params": params}

        # Use explicitly provided headers or get them from the instance
        if headers is None:
            headers = self._get_headers()
        kwargs["headers"] = headers

        if files:
            kwargs["files"] = files
            kwargs["data"] = data
        elif json is not None:
            kwargs["json"] = self._serialize_request_data(json)
        elif data is not None:
            kwargs["data"] = data

        return await self.client.request(method, url, **kwargs)

    async def close(self):
        """Close the HTTP client."""
        if self._shared_client is None and self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "APIClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
