class BasePermission:
    async def has_permission(self, scope, user):
        return True


class IsAuthenticated(BasePermission):
    async def has_permission(self, scope, user):
        return user and user.is_authenticated
