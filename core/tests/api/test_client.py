import unittest
import httpx
from unittest.mock import Mock
from typing import Dict, Any

from core.client import InvalidResourceNameError, APIClient
from core.resources import Resource


class MockResponse:
    def __init__(self, data: Dict[str, Any], status_code: int = 200):
        self._data = data
        self.status_code = status_code

    def json(self):
        return self._data


class TestAPIClient(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        """Set up test case"""
        self.client = APIClient(
            base_url="http://testserver",
            api_key=None,
            prefix="v1",
            resources=["Customer", "Order"],
            timeout=30,
        )

        # Setup mock client
        self.mock_client = Mock(spec=httpx.AsyncClient)
        self.client._client = self.mock_client

    async def test_invalid_resource_name(self):
        """Test resource name validation at registration time"""
        with self.assertRaises(InvalidResourceNameError):
            APIClient(
                base_url="http://testserver",
                api_key=None,
                resources=["invalid_name"],
            )

    async def test_resource_filtering(self):
        """Test resource filtering"""
        mock_response = MockResponse(
            {
                "object": "list",
                "data": [{"id": 1, "name": "Test Customer"}],
                "count": 1,
                "page": 1,
                "page_size": 10,
            }
        )

        self.mock_client.request.return_value = mock_response

        response = await self.client.Customer.filter(name="Test Customer").all()

        self.assertEqual(len(response["data"]), 1)
        # Access Resource instance attributes directly
        self.assertEqual(response["data"][0].name, "Test Customer")

        # Verify the request
        self.mock_client.request.assert_called_once()
        args, kwargs = self.mock_client.request.call_args
        self.assertEqual(args[0], "GET")
        self.assertEqual(args[1], "v1/customers")
        self.assertIn("filters", kwargs["params"])

    async def test_resource_creation(self):
        """Test resource creation"""
        mock_response = MockResponse({"id": 1, "name": "New Customer"})

        self.mock_client.request.return_value = mock_response

        response = await self.client.Customer.create(name="New Customer")

        # Verify the response is a Resource instance
        self.assertIsInstance(response, Resource)
        self.assertEqual(response.name, "New Customer")

        # Verify the request
        self.mock_client.request.assert_called_once()
        args, kwargs = self.mock_client.request.call_args
        self.assertEqual(args[0], "POST")
        self.assertEqual(args[1], "v1/customers")
        self.assertEqual(kwargs["json"]["name"], "New Customer")

    async def test_combined_operations(self):
        """Test combined filtering, sorting, and pagination"""
        mock_response = MockResponse(
            {
                "object": "list",
                "data": [
                    {"id": 1, "name": "Customer A", "created_at": "2023-01-01"},
                    {"id": 2, "name": "Customer B", "created_at": "2023-01-02"},
                ],
                "count": 2,
                "page": 1,
                "page_size": 2,
            }
        )

        self.mock_client.request.return_value = mock_response

        response = await (
            self.client.Customer.filter(active=True)
            .order_by("-created_at")
            .page(1)
            .page_size(2)
            .all()
        )

        # Verify the response
        self.assertEqual(len(response["data"]), 2)
        # Verify Resource instances
        self.assertIsInstance(response["data"][0], Resource)
        self.assertIsInstance(response["data"][1], Resource)
        self.assertEqual(response["data"][0].name, "Customer A")
        self.assertEqual(response["data"][1].name, "Customer B")
        self.assertEqual(response["page"], 1)
        self.assertEqual(response["page_size"], 2)

        # Verify the request parameters
        self.mock_client.request.assert_called_once()
        args, kwargs = self.mock_client.request.call_args
        self.assertEqual(kwargs["params"]["page"], 1)
        self.assertEqual(kwargs["params"]["page_size"], 2)
        self.assertIn("filters", kwargs["params"])
        self.assertEqual(kwargs["params"]["ordering"], "created_at")
        self.assertEqual(kwargs["params"]["order_direction"], "DESC")

    async def test_unauthorized_resource_access(self):
        """Test accessing an unauthorized resource"""
        with self.assertRaises(ValueError):
            await self.client.UnauthorizedResource.all()

    async def test_token_authentication(self):
        """Test token-based authentication"""
        client = APIClient(
            base_url="http://testserver",
            token="test-token",
            resources=["Customer"],
        )
        headers = client._get_headers()
        self.assertEqual(headers["Authorization"], "Bearer test-token")

    async def asyncTearDown(self):
        """Clean up after tests"""
        if self.client._client:
            await self.client.close()
