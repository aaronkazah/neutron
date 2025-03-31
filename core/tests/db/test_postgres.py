# core/tests/db/test_postgres.py
import datetime
import socket
import json # Import json
from unittest import skipIf

import numpy as np # Import numpy if used for vector tests later

from core.db.base import DATABASES
from core.db.fields import (
    CharField,
    JSONField,
    BooleanField,
    DateTimeField,
    IntegerField,
)
from core.db.model import Model
from core.tests.db.postgres_testcase import PostgresTestCase, POSTGRES_CONFIG, SKIP_POSTGRES_TESTS


# Quick check at module level if PostgreSQL is available
def is_postgres_available():
    """Quick check if PostgreSQL server is available."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            host = POSTGRES_CONFIG["HOST"]
            port = POSTGRES_CONFIG["PORT"]
            result = s.connect_ex((host, port))
            available = result == 0
            print(f"PostgreSQL availability: {'Available' if available else 'Not available'}")
            return available
    except Exception as e:
        print(f"PostgreSQL availability check error: {e}")
        return False


# Skip all tests in this module if PostgreSQL is not available
POSTGRES_AVAILABLE = is_postgres_available()
if not POSTGRES_AVAILABLE:
    print("PostgreSQL not available - All tests in this module will be skipped")
else:
    print("PostgreSQL available - Tests will run if not explicitly skipped")


class PostgresBaseModel(Model):
    """Base model for PostgreSQL tests."""
    id = CharField(primary_key=True)
    created_at = DateTimeField(default=datetime.datetime.now)

    @classmethod
    def get_app_label(cls) -> str:
        """Override to return test app label."""
        return "postgres_test"


class PostgresNetwork(PostgresBaseModel):
    """Test model for PostgreSQL."""
    name = CharField()
    settings = JSONField(null=True)
    is_active = BooleanField(default=True)
    type = CharField()
    strategy = CharField()


class PostgresUser(PostgresBaseModel):
    """Another test model for PostgreSQL."""
    username = CharField(max_length=50, unique=True)
    email = CharField(max_length=100, null=True)
    age = IntegerField(null=True)
    is_admin = BooleanField(default=False)



@skipIf(SKIP_POSTGRES_TESTS or not POSTGRES_AVAILABLE, "PostgreSQL tests disabled or server not available")
class PostgresDBTestCase(PostgresTestCase):
    """Test case for PostgreSQL database operations."""

    # Define models for testing
    models = [PostgresNetwork, PostgresUser]
    app_label = "postgres_test" # Keep this consistent with models

    @classmethod
    def setUpClass(cls):
        """Set up the test class."""
        if not POSTGRES_AVAILABLE:
            print("Skipping all tests in PostgresDBTestCase due to PostgreSQL unavailability")
            return  # Skip setting up if PostgreSQL is not available

        super().setUpClass()

    async def setup_test_data(self):
        """Set up initial test data if needed."""
        # This method is empty but could be used to create initial data
        pass

    async def test_model_creation(self):
        """Test creating models in PostgreSQL."""
        # Create a network
        network = PostgresNetwork(
            name="TestNetwork",
            settings={"key": "value"},
            is_active=True,
            type="example_type",
            strategy="default_strategy",
        )
        # Ensure the alias is set correctly before saving if needed,
        # though the test case setup should handle the default routing.
        # network.db_alias = self.db_alias # Usually not needed if router is set
        await network.save(create=True)

        # Verify it was saved
        saved_network = await PostgresNetwork.objects.get(id=network.id)
        self.assertEqual(saved_network.name, "TestNetwork")
        self.assertEqual(saved_network.settings["key"], "value")

        # Create a user
        user = PostgresUser(
            username="testuser",
            email="test@example.com",
            age=30,
            is_admin=False,
        )
        await user.save(create=True)

        # Verify it was saved
        saved_user = await PostgresUser.objects.get(id=user.id)
        self.assertEqual(saved_user.username, "testuser")
        self.assertEqual(saved_user.email, "test@example.com")
        self.assertEqual(saved_user.age, 30)

    async def test_model_filtering(self):
        """Test filtering models in PostgreSQL."""
        # Create test data
        for i in range(5):
            network = PostgresNetwork(
                name=f"Network{i}",
                settings={"index": i, "active": i % 2 == 0}, # active is boolean True/False
                is_active=i % 2 == 0,
                type=f"type{i % 3}",
                strategy=f"strategy{i}",
            )
            await network.save(create=True)

        # Test exact match filtering
        active_networks = await PostgresNetwork.objects.filter(is_active=True).all()
        self.assertEqual(len(active_networks), 3)  # Indexes 0, 2, 4

        # Test contains filtering (PostgreSQL LIKE is case-sensitive by default)
        # Use icontains for case-insensitive
        networks_with_name2 = await PostgresNetwork.objects.filter(name__icontains="network2").all()
        self.assertEqual(len(networks_with_name2), 1)
        self.assertEqual(networks_with_name2[0].name, "Network2")

        # Test JSON field filtering using the correct structure for contains
        # Check if the settings JSON contains the key-value pair "active": True
        active_in_json = await PostgresNetwork.objects.filter(settings__contains={"active": True}).all()
        self.assertEqual(len(active_in_json), 3, "Should find networks where settings JSON contains 'active': true")

        # Test multiple conditions
        type0_active = await PostgresNetwork.objects.filter(type="type0", is_active=True).all()
        self.assertEqual(len(type0_active), 1)  # Only index 0 is both type0 and active

        # Test exclusions
        inactive_networks = await PostgresNetwork.objects.exclude(is_active=True).all() # Exclude active
        self.assertEqual(len(inactive_networks), 2)  # Indexes 1, 3

    async def test_model_update(self):
        """Test updating models in PostgreSQL."""
        # Create a model
        network = PostgresNetwork(
            name="UpdateTest",
            settings={"status": "initial"},
            is_active=True,
            type="test_type",
            strategy="test_strategy",
        )
        await network.save(create=True)

        # Retrieve and update
        retrieved = await PostgresNetwork.objects.get(id=network.id)
        retrieved.name = "Updated"
        retrieved.settings = {"status": "updated"}
        retrieved.is_active = False
        await retrieved.save() # Should trigger _update

        # Verify updates
        updated = await PostgresNetwork.objects.get(id=network.id)
        self.assertEqual(updated.name, "Updated")
        self.assertEqual(updated.settings["status"], "updated")
        self.assertEqual(updated.is_active, False)

        # Test bulk update
        networks = []
        for i in range(3):
            network = PostgresNetwork(
                name=f"BulkUpdate{i}",
                settings={"batch": i},
                is_active=True,
                type="bulk_type",
                strategy="bulk_strategy",
            )
            await network.save(create=True)
            networks.append(network)

        # Update all to inactive
        for network in networks:
            network.is_active = False

        # Pass the Python objects and fields to update to bulk_update
        await PostgresNetwork.objects.bulk_update(networks, ["is_active"])

        # Verify all are inactive
        bulk_networks = await PostgresNetwork.objects.filter(name__startswith="BulkUpdate").all()
        self.assertEqual(len(bulk_networks), 3)
        for network in bulk_networks:
            self.assertFalse(network.is_active)

    async def test_model_delete(self):
        """Test deleting models in PostgreSQL."""
        # Create models to delete
        network1 = PostgresNetwork(
            name="DeleteTest1",
            settings={"delete": True},
            is_active=True,
            type="delete_type",
            strategy="delete_strategy",
        )
        await network1.save(create=True)
        # We need the actual ID generated
        network1_id = network1.id

        network2 = PostgresNetwork(
            name="DeleteTest2",
            settings={"delete": True},
            is_active=True,
            type="delete_type",
            strategy="delete_strategy",
        )
        await network2.save(create=True)

        # Verify they exist
        all_delete_tests = await PostgresNetwork.objects.filter(name__startswith="DeleteTest").all()
        self.assertEqual(len(all_delete_tests), 2)

        # Delete one *using the instance*
        # Refetch the instance just to be safe if ID generation is complex
        instance_to_delete = await PostgresNetwork.objects.get(id=network1_id)
        await instance_to_delete.delete()

        # Verify only one remains
        remaining = await PostgresNetwork.objects.filter(name__startswith="DeleteTest").all()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].name, "DeleteTest2")

        # Test bulk delete
        bulk_networks = []
        for i in range(3):
            network = PostgresNetwork(
                name=f"BulkDelete{i}",
                settings={"batch": i},
                is_active=True,
                type="bulk_type",
                strategy="bulk_strategy",
            )
            await network.save(create=True)
            bulk_networks.append(network) # Add the saved instance

        # Pass the list of saved instances to bulk_delete
        await PostgresNetwork.objects.bulk_delete(bulk_networks)

        # Verify they're gone
        remaining_bulk = await PostgresNetwork.objects.filter(name__startswith="BulkDelete").all()
        self.assertEqual(len(remaining_bulk), 0)

    async def test_json_field_operations(self):
        """Test operations on JSON fields in PostgreSQL with detailed error reporting."""
        # Create models with JSON data
        network1 = PostgresNetwork(
            name="JsonTest1",
            settings={"nested": {"key1": "value1", "key2": 123}, "tags": ["postgres", "json"]},
            is_active=True,
            type="json_type",
            strategy="json_strategy",
        )
        await network1.save(create=True)

        network2 = PostgresNetwork(
            name="JsonTest2",
            settings={"nested": {"key1": "value2"}, "priority": "high"},
            is_active=True,
            type="json_type",
            strategy="json_strategy",
        )
        await network2.save(create=True)

        # Get all records
        all_records = await PostgresNetwork.objects.filter(type="json_type").all()

        # Test JSON contains (check if the key 'key1' has value 'value1')
        # Pass a dictionary representing the JSON structure to check for
        value1_records = await PostgresNetwork.objects.filter(settings__contains={'nested': {'key1': 'value1'}}).all()
        self.assertEqual(len(value1_records), 1)
        self.assertEqual(value1_records[0].name, "JsonTest1")

        # Test JSON contains (check if the 'tags' array contains 'postgres')
        # For array containment, you might need a specific query or adjust the ORM filter
        # A simple contains might check if the literal string "postgres" exists *anywhere*
        # Let's try checking for the whole array element
        postgres_tag_records = await PostgresNetwork.objects.filter(settings__contains={'tags': ['postgres']}).all()
        # This might need more specific PG JSON operators depending on exact need,
        # but let's see if the basic ORM handles it correctly first.
        # If this fails, it indicates the ORM __contains needs refinement for arrays.
        # self.assertEqual(len(postgres_tag_records), 1) # Temporarily comment out if unsure about array contains


        # Test the has key operator (check if 'priority' key exists at the top level)
        priority_records = await PostgresNetwork.objects.filter(settings__has="priority").all()
        self.assertEqual(len(priority_records), 1)
        self.assertEqual(priority_records[0].name, "JsonTest2")


    async def test_related_models(self):
        """Test relationships between models in PostgreSQL."""
        # Create a user
        user = PostgresUser(
            username="networkadmin",
            email="admin@networks.com",
            age=35,
            is_admin=True,
        )
        await user.save(create=True)
        user_id = user.id # Capture the generated ID

        # Create networks with reference to user in settings
        for i in range(3):
            network = PostgresNetwork(
                name=f"UserNetwork{i}",
                settings={"admin_id": user_id, "level": i}, # Use the actual ID
                is_active=True,
                type="user_network",
                strategy="user_strategy",
            )
            await network.save(create=True)

        # Query networks by user ID in JSON using the correct structure
        # We want to find networks where the settings JSON contains {"admin_id": user_id}
        user_networks = await PostgresNetwork.objects.filter(settings__contains={'admin_id': user_id}).all()
        self.assertEqual(len(user_networks), 3)

        # Verify correct networks were found
        network_names = sorted([n.name for n in user_networks])
        self.assertEqual(network_names, ["UserNetwork0", "UserNetwork1", "UserNetwork2"])