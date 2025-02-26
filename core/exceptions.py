from core.utils import Status


class APIException(Exception):
    def __init__(
        self,
        message: str,
        type: str = "api_error",
        status: Status = Status.HTTP_500_INTERNAL_SERVER_ERROR,
    ):
        self.message = message
        self.type = type
        self.status_code = status
        super().__init__(message)

    def to_dict(self):
        error_dict = {"error": {"message": self.message, "type": self.type}}
        if hasattr(self, "doc_url"):
            error_dict["error"]["doc_url"] = self.doc_url
        if hasattr(self, "support_url"):
            error_dict["error"]["support_url"] = self.support_url
        return error_dict


class ValidationError(APIException):
    def __init__(self, message: str):
        super().__init__(
            message=message,
            type="invalid_request_error",
            status=Status.HTTP_400_BAD_REQUEST,
        )


class MethodNotAllowed(APIException):
    def __init__(self, method: str, path: str):
        super().__init__(
            message=f"Method {method} not allowed for URL ({path})",
            type="invalid_request_error",
            status=Status.HTTP_405_METHOD_NOT_ALLOWED,
        )


class NotFound(APIException):
    def __init__(self, path: str):
        super().__init__(
            message=f"Unrecognized request URL ({path}). If you are trying to list objects, "
            f"remove the trailing slash. If you are trying to retrieve an object, "
            f"make sure you passed a valid (non-empty) identifier.",
            type="invalid_request_error",
            status=Status.HTTP_404_NOT_FOUND,
        )


class AuthenticationFailed(APIException):
    def __init__(self, message: str):
        super().__init__(
            message=message,
            type="authentication_failed",
            status=Status.HTTP_401_UNAUTHORIZED,
        )


class PermissionDenied(APIException):
    def __init__(self, message: str = "Permission denied"):
        super().__init__(
            message=message,
            type="permission_denied",
            status=Status.HTTP_403_FORBIDDEN,
        )


class Throttled(APIException):
    def __init__(self, message: str = "Request was throttled", wait: int = None):
        super().__init__(
            message=message,
            type="throttled",
            status=Status.HTTP_429_TOO_MANY_REQUESTS,
        )
        self.wait = wait


class ModelException(Exception):
    pass


class MethodNotAllowedException(APIException):
    pass


class IntegrityError(APIException):
    pass


class DoesNotExist(APIException):
    pass
