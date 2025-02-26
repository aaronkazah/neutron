from typing import Optional


class BaseThrottle:
    async def allow_request(self, scope, rate: str) -> bool:
        raise NotImplementedError("Subclasses must implement allow_request")

    async def wait(self) -> Optional[float]:
        raise NotImplementedError("Subclasses must implement wait")
