from core.api import API
from core.utils import Status


class HealthAPI(API):
    def __init__(self, resource="/health"):
        super().__init__(resource=resource, name="health")
        self.add_route(resource, self.get, methods=["GET"], name="health")

    async def get(self, scope, receive, send, **kwargs):
        payload = {
            "status": "ok",
        }
        return await self.response(data=payload, status=Status.HTTP_200_OK)
