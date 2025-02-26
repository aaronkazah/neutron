import datetime

import jwt

import bcrypt
from typing import Any, List, Optional
import abc


class Authentication(abc.ABC):
    @classmethod
    @abc.abstractmethod
    async def authenticate(cls, email: str, password: str) -> Optional[Any]:
        raise NotImplementedError("Subclasses must implement authenticate")

    @classmethod
    @abc.abstractmethod
    async def authorize(cls, scope: List[str]) -> bool:
        raise NotImplementedError("Subclasses must implement authorize")

    @staticmethod
    async def hash_password(password: str) -> str:
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(password.encode("utf-8"), salt)
        return hashed_password.decode("utf-8")

    @staticmethod
    async def check_password(hashed_password: str, plain_password: str) -> bool:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"), hashed_password.encode("utf-8")
        )
