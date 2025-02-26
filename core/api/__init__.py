# core/api.py
import json
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import wraps
from json import JSONEncoder
from typing import Any, List, Tuple, Callable, Optional, Dict, TypeVar
from urllib.parse import parse_qs

from core import exceptions
from core.db.model import Model
from core.utils import Status

T = TypeVar("T", bound="Model")


class CustomJSONEncoder(JSONEncoder):
    """Custom JSON encoder that handles enums and datetimes."""

    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class Response:
    """HTTP Response handler."""

    def __init__(
        self,
        body: Any = None,
        status_code: int = 200,
        headers: List[Tuple[bytes, bytes]] = None,
        media_type: str = "application/json",
        indent: int = 2,
    ):
        self.body = body
        self.status_code = status_code
        self.headers = headers or []
        self.media_type = media_type
        self.indent = indent

        if not any(name.lower() == b"content-type" for name, _ in self.headers):
            self.headers.append((b"content-type", self.media_type.encode()))

    def __repr__(self):
        return (
            f"Response(body={self.body}, status_code={self.status_code}, "
            f"headers={self.headers}, media_type={self.media_type}, indent={self.indent})"
        )

    def __str__(self):
        return (
            f"Response(body={self.body}, status_code={self.status_code}, "
            f"headers={self.headers}, media_type={self.media_type}, indent={self.indent})"
        )

    async def __call__(self, scope, receive, send):
        """Send the response."""
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.headers,
            }
        )

        body_bytes = b""
        if self.body is not None:
            if self.media_type == "application/json" and isinstance(
                self.body, (dict, list)
            ):
                body_bytes = json.dumps(
                    self.body,
                    cls=CustomJSONEncoder,
                    indent=self.indent,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ": "),
                ).encode("utf-8")
            elif isinstance(self.body, str):
                body_bytes = self.body.encode("utf-8")
            elif isinstance(self.body, bytes):
                body_bytes = self.body

        await send(
            {"type": "http.response.body", "body": body_bytes, "more_body": False}
        )


@dataclass
class EndpointMetadata:
    """Metadata for API endpoints."""

    path: str
    methods: List[str]
    handler: Callable
    name: Optional[str] = None
    description: Optional[str] = None
    permission_classes: Optional[List[Any]] = None
    throttle_classes: Optional[List[Any]] = None


class API:
    """Base API class with optional CRUD functionality when model is specified."""

    model = None
    resource = ""
    authentication_class = None
    name = None
    page_size = 10
    lookup_field = "id"

    def __init__(
        self,
        resource="",
        routes=None,
        permission_classes=None,
        throttle_classes=None,
        authentication_class=None,
        name=None,
        page_size=None,
        lookup_field=None,
    ):
        self.resource = self.resource or resource.rstrip("/")
        self.routes = []
        self.kwargs = {}
        self.permission_classes = permission_classes or []
        self.throttle_classes = throttle_classes or []

        # Allow overriding configuration via init
        if page_size is not None:
            self.page_size = page_size
        if lookup_field is not None:
            self.lookup_field = lookup_field

        # Setup routes
        if routes:
            for route in routes:
                if len(route) == 2:
                    path, handler = route
                    self.add_route(path, handler)
                else:
                    path, handler, methods = route
                    self.add_route(path, handler, methods)

        self._register_endpoints()

        # Only setup CRUD routes if model is specified
        if self.model is not None:
            self._setup_default_routes()

    def params(self, scope):
        """Get the current request parameters."""
        query_string = scope.get("query_string", b"").decode()
        params = {}

        if query_string:
            parsed = parse_qs(query_string)
            # Convert all single-item lists to single values
            params = {k: v[0] if len(v) == 1 else v for k, v in parsed.items()}
        return params

    async def _process_client_filters(self, scope: Dict) -> Dict:
        """Process client-side filters and parameters."""
        params = self.params(scope)
        scope["params"] = params

        scope.update(params)

        # Handle pagination
        scope["page"] = int(params.get("page", "1"))
        scope["page_size"] = int(
            params.get("page_size", str(getattr(self, "page_size", 10)))
        )

        # Handle filters
        if "filters" in params:
            try:
                filters = json.loads(params["filters"])
                processed_filters = {}
                exclude_filters = []

                if isinstance(filters, dict):
                    processed_filters = filters
                else:
                    for filter_set in filters:
                        if isinstance(filter_set, dict):
                            if "__exclude__" in filter_set:
                                exclude_filters.append(filter_set["__exclude__"])
                            else:
                                processed_filters.update(filter_set)

                scope["processed_filters"] = processed_filters
                scope["exclude_filters"] = exclude_filters
            except json.JSONDecodeError:
                raise exceptions.ValidationError("Invalid filters format")

        # Handle ordering
        if "ordering" in params:
            scope["ordering"] = params["ordering"]
        if "order_direction" in params:
            scope["order_direction"] = params["order_direction"]

        return scope

    def _setup_default_routes(self):
        """Setup default CRUD routes if model is specified."""
        self.add_route(self.resource, self.list, methods=["GET"], name="list")
        self.add_route(self.resource, self.create, methods=["POST"], name="create")
        detail_path = f"{self.resource}/<str:{self.lookup_field}>"
        self.add_route(detail_path, self.retrieve, methods=["GET"], name="retrieve")
        self.add_route(detail_path, self.update, methods=["PATCH"], name="update")
        self.add_route(detail_path, self.destroy, methods=["DELETE"], name="destroy")

    @staticmethod
    def endpoint(
        path: str,
        methods=None,
        permission_classes=None,
        throttle_classes=None,
        name: Optional[str] = None,
    ):
        """Decorator for defining API endpoints."""
        if methods is None:
            methods = ["GET"]
        elif isinstance(methods, str):
            methods = [methods]

        methods = [m.upper() for m in methods]

        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(self, *args, **kwargs):
                return await func(self, *args, **kwargs)

            endpoint_name = name if name else func.__name__

            wrapper._endpoint_metadata = EndpointMetadata(
                path=path,
                methods=methods,
                handler=func,
                name=endpoint_name,
                description=func.__doc__,
                permission_classes=permission_classes,
                throttle_classes=throttle_classes,
            )
            return wrapper

        return decorator

    async def get_base_queryset(self, scope: Dict):
        """Get base queryset without filtering."""
        if not self.model:
            raise NotImplementedError("Model must be defined")
        return self.model.objects.using(self.model.db_alias)

    async def get_queryset(self, scope: Dict):
        """Get queryset with filters and ordering applied."""
        queryset = await self.get_base_queryset(scope)

        # Apply filters from processed_filters, not directly from filters
        filters = scope.get("processed_filters", {})  # This is the key change
        exclude_filters = scope.get("exclude_filters", [])

        # Apply normal filters
        for key, value in filters.items():
            queryset = queryset.filter(**{key: value})

        # Apply exclude filters
        for exclude_filter in exclude_filters:
            queryset = queryset.exclude(**exclude_filter)

        # Handle ordering
        if ordering := scope.get("ordering"):
            order_direction = scope.get("order_direction", "ASC")
            if ordering.startswith("-"):
                ordering = ordering[1:]
                order_direction = "DESC"

            queryset = queryset.order_by(
                f"{'-' if order_direction == 'DESC' else ''}{ordering}"
            )

        return queryset

    async def get_instance(self, scope: Dict, **kwargs) -> T:
        """Get a single instance based on scope and kwargs."""
        queryset = await self.get_queryset(scope)
        try:
            lookup_value = kwargs[self.lookup_field]
            return await queryset.get(**{self.lookup_field: lookup_value})
        except self.model.DoesNotExist as e:
            raise exceptions.NotFound(
                f"{self.model.__name__} with {self.lookup_field} not found"
            ) from e

    async def transform(self, data):
        """Transform the data before saving."""
        if data.get("id"):
            data.pop("id")
        return data

    async def list(self, scope, receive, send, **kwargs):
        """Handles list GET requests."""
        if not self.model:
            raise exceptions.MethodNotAllowed("GET", scope["path"])

        try:
            page = scope.get("page", 1)
            page_size = scope.get("page_size", self.page_size)

            queryset = await self.get_queryset(scope)
            count = await queryset.count()

            start = (page - 1) * page_size
            end = start + page_size

            # Evaluate queryset to get the actual records
            records = await queryset.all()
            records = list(records)
            page_items = records[start:end]
            data = [item.serialize() for item in page_items]

            response_data = {
                "object": "list",
                "data": data,
                "page": page,
                "page_size": page_size,
                "count": count,
                "num_pages": (count + page_size - 1) // page_size,
                "has_more": end < count,
            }
            return await self.response(response_data)
        except Exception as e:
            return await self.response(
                {"error": str(e)}, status=Status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    async def create(self, scope, receive, send, **kwargs):
        """Handles resource creation."""
        if not self.model:
            raise exceptions.MethodNotAllowed("POST", scope["path"])

        try:
            payload = kwargs.get("body", {})
            data = await self.transform(payload)
            instance = self.model(**data)
            await instance.validate()
            await instance.save()

            return await self.response(
                instance.serialize(),
                status=Status.HTTP_201_CREATED,
            )
        except exceptions.ValidationError as e:
            return await self.response(
                {"error": str(e)}, status=Status.HTTP_400_BAD_REQUEST
            )

    async def retrieve(self, scope, receive, send, **kwargs):
        """Handles detail GET requests."""
        if not self.model:
            raise exceptions.MethodNotAllowed("GET", scope["path"])

        try:
            instance = await self.get_instance(scope, **kwargs)
            return await self.response(instance.serialize())
        except self.model.DoesNotExist:
            return await self.response(
                {"error": f"{self.model.__name__} not found"},
                status=Status.HTTP_404_NOT_FOUND,
            )

    async def update(self, scope, receive, send, **kwargs):
        """Handles PATCH requests."""
        if not self.model:
            raise exceptions.MethodNotAllowed("PATCH", scope["path"])

        try:
            instance = await self.get_instance(scope, **kwargs)
            payload = kwargs.get("body", {})
            data = await self.transform(payload)
            await instance.update(data=data)
            await instance.validate()
            await instance.save()
            await instance.refresh_from_db()

            return await self.response(instance.serialize())
        except self.model.DoesNotExist:
            return await self.response(
                {"error": f"{self.model.__name__} not found"},
                status=Status.HTTP_404_NOT_FOUND,
            )
        except Exception as e:
            return await self.response(
                {"error": str(e)}, status=Status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    async def destroy(self, scope, receive, send, **kwargs):
        """Handles DELETE requests."""
        if not self.model:
            raise exceptions.MethodNotAllowed("DELETE", scope["path"])

        try:
            instance = await self.get_instance(scope, **kwargs)
            await instance.delete()
            return await self.response({}, status=Status.HTTP_204_NO_CONTENT)
        except self.model.DoesNotExist:
            return await self.response(
                {"error": f"{self.model.__name__} not found"},
                status=Status.HTTP_404_NOT_FOUND,
            )

    def _register_endpoints(self):
        """Registers endpoints decorated with @endpoint."""
        for attr_name in dir(self):
            if attr_name == "CORTEX":
                continue
            attr = getattr(self, attr_name)
            if hasattr(attr, "_endpoint_metadata"):
                metadata = attr._endpoint_metadata
                self.add_route(
                    metadata.path,
                    attr,
                    metadata.methods,
                    metadata.permission_classes,
                    metadata.throttle_classes,
                    name=f"{metadata.name}",
                )

    def add_route(
        self,
        path: str,
        handler: Callable,
        methods: List[str] = None,
        permission_classes: List[Any] = None,
        throttle_classes: List[Any] = None,
        name: Optional[str] = None,
    ):
        """Adds a new route to the API."""
        if methods is None:
            methods = ["GET"]

        if not path.startswith(self.resource):
            path = self.resource + path

        if not path.startswith("/") and path:
            path = "/" + path

        pattern = self._convert_path_to_regex(path)
        self.routes.append(
            (
                re.compile(pattern),
                handler,
                methods,
                permission_classes or self.permission_classes,
                throttle_classes or self.throttle_classes,
                name,
                path,
            )
        )

    @staticmethod
    def _convert_path_to_regex(path):
        """Convert path pattern to regex."""
        if not path or path == "/":
            return "^/?$"

        path = path.strip("/")
        parts = path.split("/")
        pattern = []

        for part in parts:
            if part.startswith("<") and part.endswith(">"):
                param_match = re.match(r"<(\w+):(\w+)>", part)
                if param_match:
                    param_type, param_name = param_match.groups()
                    if param_type == "int":
                        pattern.append(f"(?P<{param_name}>[0-9]+)")
                    elif param_type == "str":
                        pattern.append(f"(?P<{param_name}>[^/]+)")
                    elif param_type == "path":
                        pattern.append(f"(?P<{param_name}>.+)")
                    elif param_type == "slug":
                        pattern.append(f"(?P<{param_name}>[-a-zA-Z0-9_]+)")
            else:
                pattern.append(re.escape(part))

        return f"^/{'/'.join(pattern)}?$"

    @staticmethod
    async def check_permissions(scope, permission_classes):
        """Checks if the request has the required permissions."""
        user = scope.get("user")
        for permission_class in permission_classes:
            if not await permission_class().has_permission(scope, user):
                raise exceptions.PermissionDenied()

    @staticmethod
    async def check_throttles(scope, throttle_classes):
        """Checks if the request should be throttled."""
        for throttle_class in throttle_classes:
            throttle = throttle_class()
            if not await throttle.allow_request(scope, "some_rate"):
                wait_time = await throttle.wait()
                raise exceptions.Throttled(wait=wait_time)

    async def handle(self, scope, receive, send, **kwargs):
        """Enhanced handle method with form data support and raw body storage."""
        try:
            scope = await self._process_client_filters(scope)
            method = scope["method"]
            path = scope["path"].rstrip("/")

            handler, kwargs, permission_classes, throttle_classes, _, _ = (
                await self.match(path, method)
            )

            if handler is None:
                raise exceptions.MethodNotAllowed(method, path)

            if self.authentication_class:
                await self.authentication_class.authorize(scope)

            await self.check_permissions(scope, permission_classes)
            await self.check_throttles(scope, throttle_classes)

            kwargs.update(
                {
                    "page": scope.get("page", 1),
                    "page_size": scope.get("page_size", self.page_size),
                    "ordering": scope.get("ordering"),
                    "filters": scope.get("processed_filters", {}),
                    "exclude_filters": scope.get("exclude_filters", []),
                }
            )

            # Handle request body for POST/PUT/PATCH
            if method in ["POST", "PUT", "PATCH"]:
                headers = dict(scope.get("headers", []))
                content_type = headers.get(b"content-type", b"").decode()

                # Read the entire body
                raw_body = b""
                message = await receive()
                if message["type"] == "http.request":
                    raw_body = message.get("body", b"")
                    more_body = message.get("more_body", False)

                    while more_body:
                        message = await receive()
                        raw_body += message.get("body", b"")
                        more_body = message.get("more_body", False)

                # Store raw body in kwargs
                kwargs["raw"] = raw_body

                if content_type.startswith("multipart/form-data"):
                    try:
                        import cgi
                        from io import BytesIO

                        boundary = content_type.split("boundary=")[1]
                        environ = {
                            "REQUEST_METHOD": "POST",
                            "CONTENT_TYPE": content_type,
                            "CONTENT_LENGTH": str(len(raw_body)),
                        }

                        fp = BytesIO(raw_body)
                        form = cgi.FieldStorage(
                            fp=fp, environ=environ, keep_blank_values=True
                        )

                        data = {}
                        for field in form.keys():
                            if form[field].filename:
                                kwargs["file"] = form[field].file.read()
                            else:
                                data[field] = form[field].value

                        kwargs["body"] = data
                    except Exception as e:
                        raise exceptions.ValidationError(
                            f"Invalid multipart form data: {str(e)}"
                        )

                elif content_type == "application/x-www-form-urlencoded":
                    try:
                        from urllib.parse import parse_qs

                        data = parse_qs(raw_body.decode())
                        kwargs["body"] = {
                            k: v[0] if len(v) == 1 else v for k, v in data.items()
                        }
                    except Exception as e:
                        raise exceptions.ValidationError(f"Invalid form data: {str(e)}")

                else:
                    try:
                        kwargs["body"] = json.loads(raw_body) if raw_body else {}
                    except json.JSONDecodeError:
                        raise exceptions.ValidationError("Invalid JSON body")

            response = await handler(scope, receive, send, **kwargs)

            if isinstance(response, Response):
                return await response(scope, receive, send)
            else:
                raise ValueError(f"Invalid response type: {type(response)}")

        except exceptions.ValidationError as e:
            response = Response(
                body={"error": {"type": "validation_error", "message": str(e)}},
                status_code=Status.HTTP_400_BAD_REQUEST,
            )
            return await response(scope, receive, send)
        except exceptions.APIException as e:
            response = Response(body=e.to_dict(), status_code=e.status_code)
            return await response(scope, receive, send)
        except Exception as e:
            error = exceptions.APIException(
                message=str(e),
                type="internal_server_error",
                status=Status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
            response = Response(body=error.to_dict(), status_code=error.status_code)
            return await response(scope, receive, send)

    async def _parse_multipart(self, body: bytes, content_type: str) -> Dict:
        """Parse multipart form data."""
        try:
            import cgi
            from io import BytesIO

            environ = {
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": str(len(body)),
            }

            fp = BytesIO(body)
            fields = {}
            files = {}

            form = cgi.FieldStorage(fp=fp, environ=environ, keep_blank_values=True)

            for key in form.keys():
                item = form[key]
                if item.filename:
                    files[key] = item.file.read()
                else:
                    fields[key] = item.value

            return {"fields": fields, "files": files}
        except Exception as e:
            raise exceptions.ValidationError(f"Invalid multipart form data: {str(e)}")

    async def match(self, path: str, method: str = "GET"):
        """Matches a request path and method to a handler."""
        matched_handlers = []

        for (
            pattern,
            handler,
            allowed_methods,
            permission_classes,
            throttle_classes,
            name,
            original_path,
        ) in self.routes:
            match = pattern.match(path)
            if match:
                matched_handlers.append(
                    (
                        handler,
                        allowed_methods,
                        permission_classes,
                        throttle_classes,
                        match,
                        name,
                        original_path,
                    )
                )

        if not matched_handlers:
            raise exceptions.NotFound(path)

        for (
            handler,
            allowed_methods,
            permission_classes,
            throttle_classes,
            match,
            name,
            original_path,
        ) in matched_handlers:
            if method in allowed_methods:
                kwargs = match.groupdict()
                return (
                    handler,
                    {**kwargs},
                    permission_classes,
                    throttle_classes,
                    name,
                    original_path,
                )

        if matched_handlers:
            raise exceptions.MethodNotAllowed(method, path)

        raise exceptions.NotFound(path)

    @staticmethod
    async def response(
        data, status=Status.HTTP_200_OK, headers=None, media_type="application/json"
    ):
        """Sends an HTTP response."""
        resp = Response(
            body=data, status_code=status, headers=headers, media_type=media_type
        )
        return resp

    @staticmethod
    async def data(receive: Callable) -> Dict:
        """Extracts data from the request body."""
        raw_data = await receive()
        body = raw_data.get("body", b"")

        if not body:
            return {}

        if isinstance(body, str):
            body = body.encode("utf-8")

        try:
            return json.loads(body)
        except json.JSONDecodeError:
            try:
                return json.loads(body.decode("utf-8"))
            except json.JSONDecodeError as e:
                try:
                    parsed = parse_qs(body.decode("utf-8"))
                    return {k: v[0] if len(v) == 1 else v for k, v in parsed.items()}
                except Exception as e:
                    print(f"Query string parse error: {e}")
                raise ValueError(f"Unable to parse request body: {e}")

    def reverse(self, name: str, **kwargs) -> str:
        """Reverse URL lookup based on endpoint name and provided kwargs."""
        for (
            pattern,
            handler,
            allowed_methods,
            permission_classes,
            throttle_classes,
            route_name,
            original_path,
        ) in self.routes:
            if route_name == name:
                url = original_path
                # Correctly extract parameter names and types
                for param_match in re.findall(r"<(\w+):(\w+)>", original_path):
                    param_type, param_name = param_match
                    if param_name in kwargs:
                        url = url.replace(
                            f"<{param_type}:{param_name}>", str(kwargs[param_name])
                        )
                    else:
                        raise ValueError(
                            f"Missing parameter '{param_name}' for route '{name}'."
                        )
                # Check if all parameters have been replaced
                if "<" not in url and ">" not in url:
                    return url
                else:
                    remaining_params = [
                        param_name
                        for _, param_name in re.findall(r"<(\w+):(\w+)>", url)
                    ]
                    raise ValueError(
                        f"Missing parameters for route '{name}'. Required: {remaining_params}"
                    )
        raise ValueError(f"Reverse for '{name}' not found.")


class Paginator:
    """Async Paginator."""

    def __init__(self, queryset, page_size):
        self.queryset = queryset
        self.page_size = page_size
        self._count = None
        self._num_pages = None

    @property
    async def count(self):
        """get total count."""
        if self._count is None:
            self._count = await self.queryset.count()
        return self._count

    @property
    async def num_pages(self):
        """get num of pages."""
        if self._num_pages is None:
            count = await self.count
            if count == 0:
                self._num_pages = 0
            else:
                self._num_pages = int((count - 1) / self.page_size) + 1
        return self._num_pages

    async def get_page(self, number):
        """get page."""
        if number < 1:
            number = 1
        elif number > await self.num_pages:
            number = await self.num_pages

        start_index = (number - 1) * self.page_size
        end_index = number * self.page_size

        object_list = await self.queryset[start_index:end_index]
        return Page(object_list, number, self)


class Page:
    """Async Page."""

    def __init__(self, object_list, number, paginator):
        self.object_list = object_list
        self.number = number
        self.paginator = paginator

    @property
    async def has_next(self):
        """check if it has next."""
        return self.number < await self.paginator.num_pages

    @property
    async def has_previous(self):
        """check if it has previous."""
        return self.number > 1

    @property
    async def has_other_pages(self):
        """check if it has other pages."""
        return await self.has_previous or await self.has_next

    @property
    async def next_page_number(self):
        """get next page number."""
        if await self.has_next:
            return self.number + 1
        return None

    @property
    async def previous_page_number(self):
        """get previous page number."""
        if await self.has_previous:
            return self.number - 1
        return None

    @property
    async def start_index(self):
        """get start index."""
        if await self.paginator.count == 0:
            return 0
        return (self.paginator.page_size * (self.number - 1)) + 1

    @property
    async def end_index(self):
        """get end index."""
        if self.number == await self.paginator.num_pages:
            return await self.paginator.count
        return self.number * self.paginator.page_size


class CORS:
    def __init__(
        self,
        router: Callable,
        allowed_origins: List[str] = None,
        allow_all_origins: bool = False,
    ):
        self.router = router
        self.allowed_origins = allowed_origins
        self.allow_all_origins = allow_all_origins

        if not allow_all_origins and allowed_origins is None:
            raise ValueError(
                "Either 'allow_all_origins' must be True or 'allowed_origins' must be provided."
            )

    async def __call__(self, scope: Dict, receive: Callable, send: Callable, **kwargs):
        if scope["type"] != "http":
            await self.router(scope, receive, send, **kwargs)
            return

        headers = dict(scope.get("headers", []))
        origin = headers.get(b"origin", b"").decode("utf-8", "ignore")

        if scope["method"] == "OPTIONS":
            # Handle OPTIONS directly and do not forward to Thalamus
            if self.is_origin_allowed(origin):
                await self.handle_preflight(origin, send)
            else:
                await self.send_error_response(send, 403, "Forbidden")
            return  # Ensure no further processing for OPTIONS requests
        else:
            # Regular requests are processed as before
            kwargs["origin"] = origin
            await self.handle_simple(scope, receive, send, **kwargs)

    async def handle_preflight(self, origin: str, send: Callable):
        response_headers = [
            (b"Access-Control-Allow-Origin", origin.encode("utf-8")),
            (b"Access-Control-Allow-Credentials", b"true"),
            (
                b"Access-Control-Allow-Methods",
                b"GET, POST, PUT, PATCH, DELETE, OPTIONS",
            ),
            (b"Access-Control-Allow-Headers", b"Content-Type, Authorization"),
            (b"Access-Control-Max-Age", b"3600"),
            (b"Vary", b"Origin"),
            (b"Content-Length", b"0"),
        ]
        await send(
            {
                "type": "http.response.start",
                "status": 204,
                "headers": response_headers,
            }
        )
        await send({"type": "http.response.body", "body": b""})

    async def handle_simple(
        self, scope: Dict, receive: Callable, send: Callable, **kwargs
    ):
        origin = kwargs.get("origin")

        async def wrapped_send(response: Dict):
            if response["type"] == "http.response.start":
                if self.is_origin_allowed(origin):
                    response_headers = response.get("headers", [])
                    response_headers.extend(self.get_cors_headers(origin))
                    response["headers"] = response_headers
            await send(response)

        await self.router(scope, receive, wrapped_send)

    async def send_error_response(self, send: Callable, status_code: int, message: str):
        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": [(b"Content-Type", b"text/plain")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": message.encode("utf-8"),
            }
        )

    def is_origin_allowed(self, origin: str) -> bool:
        if self.allow_all_origins:
            return True
        if self.allowed_origins:
            return origin in self.allowed_origins
        return False

    def get_cors_headers(self, origin: str) -> List[Tuple[bytes, bytes]]:
        return [
            (b"Access-Control-Allow-Origin", origin.encode("utf-8")),
            (b"Access-Control-Allow-Credentials", b"true"),
            (b"Vary", b"Origin"),
        ]

    def reverse(self, name: str, **kwargs) -> str:
        return self.router.reverse(name, **kwargs)


class Router:
    """Router for handling multiple APIs."""

    def __init__(self, apis: Dict[str, API]):
        self.apis = {v.name: v for k, v in apis.items()}

    async def __call__(self, scope, receive, send, **kwargs):
        path = scope["path"]
        if path != "/":
            path = path.rstrip("/")
        method = scope.get("method", "")

        for api_name, api in self.apis.items():
            try:
                # Attempt to match the path against each API
                handler, kwargs, permission_classes, throttle_classes, _, _ = (
                    await api.match(path, method)
                )
                if handler:
                    await api.handle(scope, receive, send, **kwargs)
                    return
            except exceptions.NotFound:
                continue

        error = exceptions.NotFound(f"{method}: {path}")
        response = Response(body=error.to_dict(), status_code=error.status_code)
        await response(scope, receive, send)

    def reverse(self, name: str, **kwargs) -> str:
        """Reverse URL lookup for a named route across all registered APIs."""
        api_name, endpoint_name = name.split(":", 1)
        if api_name in self.apis:
            return self.apis[api_name].reverse(endpoint_name, **kwargs)
        else:
            raise ValueError(f"API '{name}' not found.")


class InternalAccessMiddleware:
    def __init__(self, router: Callable):
        self.router = router

    async def __call__(self, scope: Dict, receive: Callable, send: Callable, **kwargs):
        if scope["type"] != "http":
            await self.router(scope, receive, send, **kwargs)
            return

        path = scope.get("path", "")

        # Check if the path starts with /internal
        if path.startswith("/internal"):
            # Get headers from scope
            headers = dict(scope.get("headers", []))
            api_key = headers.get(b"x-api-key", b"").decode("utf-8")

            # Check for valid API key
            if self.is_valid_api_key(api_key):
                await self.router(scope, receive, send, **kwargs)
                return

            await self.send_unauthorized_response(send)
            return

        await self.router(scope, receive, send, **kwargs)

    def is_valid_api_key(self, api_key: str) -> bool:
        try:
            return api_key == settings.SECRET_KEY
        except AttributeError:
            return False

    async def send_unauthorized_response(self, send: Callable):
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b'{"error": "Invalid or missing API key for internal endpoint"}',
            }
        )

    def reverse(self, name: str, **kwargs) -> str:
        """Reverse URL lookup for a named route across all registered APIs."""
        return self.router.reverse(name, **kwargs)


def create_application(apis: Dict[str, API]):
    router = Router(apis)
    internal_middleware = InternalAccessMiddleware(router)
    return CORS(internal_middleware, allow_all_origins=True)
