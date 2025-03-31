# API Client Documentation

## Core Concepts

### Dynamic Resource Handling with PascalCase

- The client uses Python's `__getattr__` to dynamically handle any resource.
- Resources MUST be in PascalCase format.
- Example: `Customer` → `/customers`, `PaymentIntent` → `/payment-intents`.
- Raises `InvalidResourceNameError` if not PascalCase.

### Resource Naming Examples

- **Correct**: `client.Customer`, `client.PaymentIntent`, `client.CustomerPayment`
- **Incorrect**: `client.customer`, `client.payment_intent`, `client._resource`

### Endpoint Generation

- `Customer` → `/customers`
- `PaymentIntent` → `/payment-intents`
- `CustomerPayment` → `/customer-payments`

## Method Chaining

- **Filters**: `client.Customer.filter(status="active")`
- **Ordering**: `client.Customer.order_by("-created_at")`
- **Pagination**: `client.Customer.page(1).page_size(10)`
- **Complete**: `client.Customer.filter(status="active").order_by("-created_at").all()`

## CRUD Operations

- **List**: `client.Customer.all()`
- **Get**: `client.Customer.get("123")`
- **Create**: `client.Customer.create(name="John")`
- **Update**: `client.Customer.update("123", name="John")`
- **Delete**: `client.Customer.delete("123")`

## Query Methods

- `filter()`: Add inclusion criteria
- `exclude()`: Add exclusion criteria
- `order_by()`: Specify sorting
- `page() / page_size()`: Control pagination

## Key Benefits

- Consistent interface across resources
- Automatic pluralization and kebab-case conversion
- Type-safe with full IDE support
- Enforced naming conventions
- Chainable query building

## Error Handling

- Invalid resource names raise `InvalidResourceNameError`
- Proper HTTP error handling
- Validation of response data

## Authentication

- **API Key**: `client = APIClient(api_key="key123")`
- **Bearer Token**: `client = APIClient(token="token123")`
- **Custom Headers**: Configurable through constructor

## Advanced Usage

- Nested filters: `filter(user__status="active")`
- Complex queries: `filter(age__gt=18, status="verified")`
- Custom endpoints: Supports prefix configuration
- Async context manager: `async with APIClient() as client:`