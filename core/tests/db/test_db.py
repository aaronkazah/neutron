import datetime
from unittest import IsolatedAsyncioTestCase

import numpy as np

from core.db.fields import (
    CharField,
    JSONField,
    BooleanField,
    DateTimeField,
    VectorField,
)
from core.db.migrations import MigrationManager
from core.db.model import Model
from core.db.base import DATABASES, BASE_PRAGMAS

# Make sure Model.get_app_label() returns values that match database configurations
class BaseModel(Model):
    """Base model class with common functionality."""

    id: str = CharField(primary_key=True)
    created_at: datetime.datetime = DateTimeField(default=datetime.datetime.now)

    @classmethod
    def get_app_label(cls) -> str:
        """Override to return test app label."""
        return "test_app"

    @classmethod
    def get_table_name(cls) -> str:
        """Override to return simple test table name."""
        return f"test_app_{cls.__name__.lower()}"

class Network(BaseModel):
    name: str = CharField()
    settings: dict = JSONField(null=True)
    is_active: bool = BooleanField(default=True)
    type: str = CharField()
    strategy: str = CharField()


class Embeddings(BaseModel):
    model = CharField()
    vector = VectorField(null=True)
    text: str = CharField()
    hash: str = CharField()
    type: str = CharField()


class ModelTestCase(IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.migration_manager = MigrationManager()

        # Setup database aliases
        self.test_db_alias_default = "test_db_default"
        self.test_db_alias_secondary = "test_db_secondary"

        # Database configuration
        self.CONFIG = {
            self.test_db_alias_default: {  # Changed from "test" to actual alias
                "ENGINE": "aiosqlite",
                "NAME": ":memory:",
                "PRAGMAS": BASE_PRAGMAS.copy(),
            },
            self.test_db_alias_secondary: {
                "ENGINE": "aiosqlite",
                "NAME": ":memory:",
                "PRAGMAS": BASE_PRAGMAS.copy(),
            },
        }

        DATABASES.apply_config(self.CONFIG)

        # Bootstrap each database
        for alias in [self.test_db_alias_default, self.test_db_alias_secondary]:
            _, connection = await self.migration_manager.bootstrap(
                app_label="test_app",
                models=[Network, Embeddings],
                db=alias,
                test_mode=True,
            )
            # Store connection for cleanup
            setattr(self, f"connection_{alias}", connection)

    async def asyncTearDown(self):
        await DATABASES.shutdown(
            [self.test_db_alias_default, self.test_db_alias_secondary]
        )

    async def test_model_creation(self):
        # Test record creation in the default test database
        network = Network(
            name="TestNetwork",
            settings={"key": "value"},
            is_active=True,
            type="example_type",
            strategy="default_strategy",
        )
        network.db_alias = self.test_db_alias_default
        await network.save()

        networks = (
            await Network.objects.using(self.test_db_alias_default)
            .filter(is_active=True)
            .all()
        )
        self.assertEqual(len(networks), 1)
        self.assertEqual(networks[0].name, "TestNetwork")

    async def test_model_retrieval(self):
        # Create a Network instance in the default test database
        network = Network(
            name="TestNetwork",
            settings={"key": "value"},
            is_active=True,
            type="example_type",
            strategy="default_strategy",
        )
        network.db_alias = self.test_db_alias_default

        await network.save()

        # Retrieve the same instance
        retrieved_network = await Network.objects.using(self.test_db_alias_default).get(
            id=network.id
        )
        self.assertEqual(retrieved_network.name, "TestNetwork")

    async def test_model_filtering(self):
        # Create test data
        network1 = Network(
            name="Network1",
            settings={"key": "value1"},
            is_active=True,
            type="type1",
            strategy="strategy1",
        )
        network1.db_alias = self.test_db_alias_default
        await network1.save()

        network2 = Network(
            name="Network2",
            settings={"key": "value2"},
            is_active=False,
            type="type2",
            strategy="strategy2",
        )
        network2.db_alias = self.test_db_alias_default
        await network2.save()

        network3 = Network(
            name="TestNetwork3",
            settings={"key": "value3"},
            is_active=True,
            type="type1",
            strategy="strategy3",
        )
        network3.db_alias = self.test_db_alias_default
        await network3.save()

        # Test exact match filtering
        exact_match = (
            await Network.objects.using(self.test_db_alias_default)
            .filter(name="Network1")
            .all()
        )
        self.assertEqual(len(exact_match), 1)
        self.assertEqual(exact_match[0].name, "Network1")

        # Test contains filtering
        contains_match = (
            await Network.objects.using(self.test_db_alias_default)
            .filter(name__contains="Network")
            .all()
        )
        self.assertEqual(len(contains_match), 3)

        # Test case-insensitive contains
        icontains_match = (
            await Network.objects.using(self.test_db_alias_default)
            .filter(name__icontains="network")
            .all()
        )
        self.assertEqual(len(icontains_match), 3)

        # Test starts with filtering
        starts_with = (
            await Network.objects.using(self.test_db_alias_default)
            .filter(name__startswith="Test")
            .all()
        )
        self.assertEqual(len(starts_with), 1)
        self.assertEqual(starts_with[0].name, "TestNetwork3")

        # Test multiple conditions
        multiple_conditions = (
            await Network.objects.using(self.test_db_alias_default)
            .filter(type="type1", is_active=True)
            .all()
        )
        self.assertEqual(len(multiple_conditions), 2)

        # Test IN clause
        in_filter = (
            await Network.objects.using(self.test_db_alias_default)
            .filter(name__in=["Network1", "Network2"])
            .all()
        )
        self.assertEqual(len(in_filter), 2)

        # Test NULL checking
        null_settings = (
            await Network.objects.using(self.test_db_alias_default)
            .filter(settings__isnull=False)
            .all()
        )

        self.assertEqual(len(null_settings), 3)  # Now it should be 3

        # Test exclusion
        excluded = (
            await Network.objects.using(self.test_db_alias_default)
            .filter(__exclude__={"type": "type1"})
            .all()
        )
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0].type, "type2")

    async def test_advanced_filtering(self):
        # Create test data with specific values for range testing
        for i in range(1, 6):
            network = Network(
                name=f"Network{i}",
                settings={"key": f"value{i}"},
                is_active=i % 2 == 0,  # alternate between True and False
                type=f"type{(i % 2) + 1}",
                strategy=f"strategy{i}",
            )
            network.db_alias = self.test_db_alias_default
            await network.save()

        # Test range filtering
        range_filter = (
            await Network.objects.using(self.test_db_alias_default)
            .filter(name__in=["Network2", "Network4"])
            .all()
        )
        self.assertEqual(len(range_filter), 2)

        # Test greater than
        gt_filter = (
            await Network.objects.using(self.test_db_alias_default)
            .filter(name__gt="Network3")
            .all()
        )
        self.assertEqual(len(gt_filter), 2)

        # Test less than or equal
        lte_filter = (
            await Network.objects.using(self.test_db_alias_default)
            .filter(name__lte="Network3")
            .all()
        )
        self.assertEqual(len(lte_filter), 3)

        # Test combined filters
        combined = (
            await Network.objects.using(self.test_db_alias_default)
            .filter(is_active=True, type="type1")
            .filter(name__contains="Network")
            .all()
        )
        self.assertEqual(len(combined), 2)

        # Test endswith
        endswith_filter = (
            await Network.objects.using(self.test_db_alias_default)
            .filter(name__endswith="1")
            .all()
        )
        self.assertEqual(len(endswith_filter), 1)
        self.assertEqual(endswith_filter[0].name, "Network1")

    async def test_model_ordering(self):
        # Create Network instances
        network_a = Network(
            name="NetworkA",
            settings={"key": "valueA"},
            is_active=True,
            type="typeA",
            strategy="strategyA",
        )
        network_a.db_alias = self.test_db_alias_default
        await network_a.save()

        network_b = Network(
            name="NetworkB",
            settings={"key": "valueB"},
            is_active=True,
            type="typeB",
            strategy="strategyB",
        )
        network_b.db_alias = self.test_db_alias_default
        await network_b.save()

        # Order networks by name
        ordered_networks = (
            await Network.objects.using(self.test_db_alias_default)
            .order_by("name")
            .all()
        )
        self.assertEqual(ordered_networks[0].name, "NetworkA")
        self.assertEqual(ordered_networks[1].name, "NetworkB")

    async def test_model_limit(self):
        # Create multiple Network instances
        for i in range(5):
            network = Network(
                name=f"Network{i}",
                settings={"key": f"value{i}"},
                is_active=True,
                type=f"type{i}",
                strategy=f"strategy{i}",
            )
            network.db_alias = self.test_db_alias_default
            await network.save()

        # Limit the number of retrieved networks
        limited_networks = (
            await Network.objects.using(self.test_db_alias_default).limit(2).all()
        )
        self.assertEqual(len(limited_networks), 2)
        self.assertEqual(limited_networks[0].name, "Network0")
        self.assertEqual(limited_networks[1].name, "Network1")

    async def test_model_delete(self):
        # Create and then delete a Network instance
        network = Network(
            name="TestNetwork",
            settings={"key": "value"},
            is_active=True,
            type="type_delete",
            strategy="strategy_delete",
        )
        network.db_alias = self.test_db_alias_default
        await network.save()
        await network.refresh_from_db()

        network_id = network.id
        await network.using(self.test_db_alias_default).delete()

        with self.assertRaises(Network.DoesNotExist):
            await Network.objects.using(self.test_db_alias_default).get(id=network_id)

    async def test_model_exclude(self):
        # Create Network instances
        network1 = Network(
            name="Network1",
            settings={"key": "value1"},
            is_active=True,
            type="type1",
            strategy="strategy1",
        )
        network1.db_alias = self.test_db_alias_default
        await network1.save()

        network2 = Network(
            name="Network2",
            settings={"key": "value2"},
            is_active=False,
            type="type2",
            strategy="strategy2",
        )
        network2.db_alias = self.test_db_alias_default
        await network2.save()

        # Exclude active networks and filter inactive ones
        excluded_networks = (
            await Network.objects.using(self.test_db_alias_default)
            .filter(is_active=False)
            .exclude(is_active=True)
            .all()
        )
        self.assertEqual(len(excluded_networks), 1)
        self.assertEqual(excluded_networks[0].name, "Network2")

    async def test_model_bulk_create(self):
        # Bulk create Network instances
        networks = [
            Network(
                name="Network1",
                settings={"key": "value1"},
                is_active=True,
                type="type1",
                strategy="strategy1",
            ),
            Network(
                name="Network2",
                settings={"key": "value2"},
                is_active=True,
                type="type2",
                strategy="strategy2",
            ),
        ]
        await Network.objects.using(self.test_db_alias_default).bulk_create(networks)
        retrieved_networks = await Network.objects.using(
            self.test_db_alias_default
        ).all()
        self.assertEqual(len(retrieved_networks), 2)
        self.assertTrue(any(n.name == "Network1" for n in retrieved_networks))
        self.assertTrue(any(n.name == "Network2" for n in retrieved_networks))

    async def test_model_bulk_update(self):
        # Bulk create and then update Network instances
        network1 = Network(
            name="Network1",
            settings={"key": "value1"},
            is_active=True,
            type="type1",
            strategy="strategy1",
        )
        network1.db_alias = self.test_db_alias_default
        await network1.save()

        network2 = Network(
            name="Network2",
            settings={"key": "value2"},
            is_active=True,
            type="type2",
            strategy="strategy2",
        )
        network2.db_alias = self.test_db_alias_default
        await network2.save()

        networks = [network1, network2]

        # Update is_active to False
        for network in networks:
            network.is_active = False
        await Network.objects.using(self.test_db_alias_default).bulk_update(
            networks, ["is_active"]
        )
        updated_networks = (
            await Network.objects.using(self.test_db_alias_default)
            .filter(is_active=False)
            .all()
        )
        self.assertEqual(len(updated_networks), 2)
        self.assertTrue(all(not n.is_active for n in updated_networks))

    async def test_model_bulk_delete(self):
        # Bulk create and then delete Network instances
        network1 = Network(
            name="Network1",
            settings={"key": "value1"},
            is_active=True,
            type="type1",
            strategy="strategy1",
        )
        network1.db_alias = self.test_db_alias_default
        await network1.save()

        network2 = Network(
            name="Network2",
            settings={"key": "value2"},
            is_active=True,
            type="type2",
            strategy="strategy2",
        )
        network2.db_alias = self.test_db_alias_default
        await network2.save()

        networks = [network1, network2]

        await Network.objects.using(self.test_db_alias_default).bulk_delete(networks)
        remaining_networks = await Network.objects.using(
            self.test_db_alias_default
        ).all()
        self.assertEqual(len(remaining_networks), 0)

    async def test_model_search(self):
        # Test vector similarity search
        vector = np.random.rand(384).astype(np.float32)

        embedding = Embeddings(
            model="bge-small-en-v1.5",
            vector=vector,
            text="This is a test",
            hash="test_hash",
            type="document",
        )
        embedding.db_alias = self.test_db_alias_default
        await embedding.save()

    async def test_json_field_contains_filter(self):
        network3 = Network(
            name="Network3",
            settings={"key": "value3", "another_key": "value4"},
            is_active=True,
            type="type3",
            strategy="strategy3",
        )
        network3.db_alias = self.test_db_alias_default
        await network3.save()

        network4 = Network(
            name="Network4",
            settings={"key": "value5"},
            is_active=True,
            type="type4",
            strategy="strategy4",
        )
        network4.db_alias = self.test_db_alias_default
        await network4.save()

        networks = (
            await Network.objects.using(self.test_db_alias_default)
            .filter(settings__contains="value4")
            .all()
        )
        self.assertEqual(len(networks), 1)
        self.assertEqual(networks[0].name, "Network3")

    async def test_json_field_has_key_filter(self):
        network5 = Network(
            name="Network5",
            settings={"key": "value5", "another_key": "value6"},
            is_active=True,
            type="type5",
            strategy="strategy5",
        )
        network5.db_alias = self.test_db_alias_default
        await network5.save()

        network6 = Network(
            name="Network6",
            settings={"key": "value7"},
            is_active=True,
            type="type6",
            strategy="strategy6",
        )
        network6.db_alias = self.test_db_alias_default
        await network6.save()

        networks = (
            await Network.objects.using(self.test_db_alias_default)
            .filter(settings__has="another_key")
            .all()
        )
        self.assertEqual(len(networks), 1)
        self.assertEqual(networks[0].name, "Network5")

    async def test_exclude_json_field_contains(self):
        network_a = Network(
            name="NetworkA",
            settings={"key": "valueA", "another_key": "valueB"},
            is_active=True,
            type="typeA",
            strategy="strategyA",
        )
        network_a.db_alias = self.test_db_alias_default
        await network_a.save()

        network_b = Network(
            name="NetworkB",
            settings={"key": "valueC"},
            is_active=True,
            type="typeB",
            strategy="strategyB",
        )
        network_b.db_alias = self.test_db_alias_default
        await network_b.save()

        # Exclude networks where settings contain "valueB"
        networks = (
            await Network.objects.using(self.test_db_alias_default)
            .exclude(settings__contains="valueB")
            .all()
        )
        self.assertEqual(len(networks), 1)
        self.assertEqual(networks[0].name, "NetworkB")

    async def test_exclude_json_field_has_key(self):
        network_c = Network(
            name="NetworkC",
            settings={"key": "valueD", "another_key": "valueE"},
            is_active=True,
            type="typeC",
            strategy="strategyC",
        )
        network_c.db_alias = self.test_db_alias_default
        await network_c.save()

        network_d = Network(
            name="NetworkD",
            settings={"key": "valueF"},
            is_active=True,
            type="typeD",
            strategy="strategyD",
        )
        network_d.db_alias = self.test_db_alias_default
        await network_d.save()

        # Exclude networks where settings have the key "another_key"
        networks = (
            await Network.objects.using(self.test_db_alias_default)
            .exclude(settings__has="another_key")
            .all()
        )
        self.assertEqual(len(networks), 1)
        self.assertEqual(networks[0].name, "NetworkD")

    async def test_exclude_non_json_field(self):
        network_e = Network(
            name="NetworkE",
            settings={"key": "valueG"},
            is_active=True,
            type="typeE",
            strategy="strategyE",
        )
        network_e.db_alias = self.test_db_alias_default
        await network_e.save()

        network_f = Network(
            name="NetworkF",
            settings={"key": "valueH"},
            is_active=False,
            type="typeF",
            strategy="strategyF",
        )
        network_f.db_alias = self.test_db_alias_default
        await network_f.save()

        # Exclude inactive networks
        networks = (
            await Network.objects.using(self.test_db_alias_default)
            .exclude(is_active=False)
            .all()
        )
        self.assertEqual(len(networks), 1)
        self.assertEqual(networks[0].name, "NetworkE")
