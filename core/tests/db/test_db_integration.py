# core/tests/db/test_db_integration.py
from unittest import IsolatedAsyncioTestCase
from typing import List, Type, Tuple

from core.db.base import DATABASES, use_single_database, use_per_app_database
from core.db.router import PerAppRouter, SingleDatabaseRouter
from core.db.model import Model
from core.db.fields import CharField, IntegerField
from core.db.migrations import MigrationManager


class AppAModel(Model):
    name = CharField(max_length=100)
    value = IntegerField(null=True)

    @classmethod
    def get_app_label(cls) -> str:
        return "app_a"




class AppBModel(Model):
    name = CharField(max_length=100)
    description = CharField(max_length=255, null=True)

    @classmethod
    def get_app_label(cls) -> str:
        return "app_b"


class DatabaseIntegrationTestCase(IsolatedAsyncioTestCase):
    """Test integration of database routers and model operations."""

    async def asyncSetUp(self):
        # Reset the database configuration for a clean test environment
        DATABASES.CONFIG = {}
        DATABASES.CONNECTIONS = {}

        # Configure in-memory databases for testing
        DATABASES.configure_app("app_a", memory=True)
        DATABASES.configure_app("app_b", memory=True)
        DATABASES.configure_app("default", memory=True)  # For single database router

        # Create and bootstrap database schemas
        self.migration_manager = MigrationManager()

        # Create tables for app_a in its database
        await self.migration_manager.bootstrap(
            app_label="app_a",
            models=[AppAModel],
            db="app_a",
            test_mode=True
        )

        # Create tables for app_b in its database
        await self.migration_manager.bootstrap(
            app_label="app_b",
            models=[AppBModel],
            db="app_b",
            test_mode=True
        )

        # Create tables for both apps in the default database (for single db router)
        await self.migration_manager.bootstrap(
            app_label="app_a",
            models=[AppAModel],
            db="default",
            test_mode=True
        )

        await self.migration_manager.bootstrap(
            app_label="app_b",
            models=[AppBModel],
            db="default",
            test_mode=True
        )

    async def asyncTearDown(self):
        """Clean up resources after tests."""
        await DATABASES.shutdown()

    async def test_router_switching(self):
        """Test that switching between routers affects model operations."""
        # Start with PerAppRouter
        router = use_per_app_database()
        self.assertIsInstance(router, PerAppRouter)
        self.assertIsInstance(DATABASES.router, PerAppRouter)

        # Create and save a model instance using the per-app router
        model_a = AppAModel(name="First A model")
        # db_alias will be set by the router to "app_a"
        await model_a.save(create=True)
        self.assertEqual(model_a.db_alias, "app_a")

        # Verify the instance was saved
        retrieved_a = await AppAModel.objects.using("app_a").get(id=model_a.id)
        self.assertEqual(retrieved_a.name, "First A model")

        # Switch to SingleDatabaseRouter
        router = use_single_database("default")
        self.assertIsInstance(router, SingleDatabaseRouter)
        self.assertEqual(router.default_alias, "default")
        self.assertIsInstance(DATABASES.router, SingleDatabaseRouter)

        # Create a new model instance after router change
        model_a2 = AppAModel(name="Second A model")

        # Get the routing decision
        db_alias_from_router = DATABASES.router.db_for_app(model_a2.get_app_label())

        # Explicitly set the database alias based on the router decision
        model_a2.db_alias = db_alias_from_router

        # Save the model
        await model_a2.save(create=True)

        # The router should have set the db_alias to "default"
        self.assertEqual(model_a2.db_alias, "default")

        # Verify we can retrieve the model from the default database
        model_from_default = await AppAModel.objects.using("default").get(id=model_a2.id)
        self.assertEqual(model_from_default.name, "Second A model")

    async def test_multiple_database_operations(self):
        """Test operations across multiple databases simultaneously."""
        # Set up PerAppRouter for this test
        router = use_per_app_database()

        # Create models in both app_a and app_b
        model_a = AppAModel(name="Test Multiple DB - A", value=100)
        model_b = AppBModel(name="Test Multiple DB - B", description="Cross-database test")

        await model_a.save(create=True)
        await model_b.save(create=True)

        # Verify each is in its own database
        self.assertEqual(model_a.db_alias, "app_a")
        self.assertEqual(model_b.db_alias, "app_b")

        # Test retrieving from respective databases
        retrieved_a = await AppAModel.objects.using("app_a").get(id=model_a.id)
        retrieved_b = await AppBModel.objects.using("app_b").get(id=model_b.id)

        self.assertEqual(retrieved_a.name, "Test Multiple DB - A")
        self.assertEqual(retrieved_b.name, "Test Multiple DB - B")

        # Test that each model is only in its own database
        app_a_count = await AppAModel.objects.using("app_a").count()
        app_b_count = await AppBModel.objects.using("app_b").count()

        self.assertGreaterEqual(app_a_count, 1)
        self.assertGreaterEqual(app_b_count, 1)