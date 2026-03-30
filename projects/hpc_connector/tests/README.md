# Testing Guide

This document describes how to run tests for the HPC Connector.

## Running Tests

### Install Dependencies

```bash
pip install pytest pytest-asyncio
pip install asyncssh  # Required for SSH tests
```

### Run All Tests

```bash
cd hpc_connector
python -m pytest tests/ -v
```

### Run Specific Test Categories

```bash
# Unit tests only
python -m pytest tests/ -v -m "not integration"

# Integration tests (requires cluster connection)
python -m pytest tests/ -v -m integration

# Specific test file
python -m pytest tests/test_hpc_connector.py::TestClusterConfig -v
```

### Test Coverage

```bash
pip install pytest-cov
python -m pytest tests/ --cov=hpc_connector --cov-report=html
```

## Writing Tests

### Unit Tests

Unit tests mock external dependencies (SSH connections, cluster commands):

```python
@pytest.mark.asyncio
async def test_job_submission():
    mock_connector = AsyncMock()
    mock_connector.execute = AsyncMock(return_value={
        'stdout': 'Submitted batch job 12345',
        'exit_code': 0
    })
    
    scheduler = SlurmScheduler(mock_connector)
    job_id = await scheduler.submit_job(job_config)
    
    assert job_id == "12345"
```

### Integration Tests

Integration tests require actual cluster access:

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_job():
    config = ClusterConfig.from_dict({
        "name": "test-cluster",
        "cluster_type": "slurm",
        "ssh": {
            "host": "login.cluster.edu",
            "user": "testuser",
            "auth_method": "key",
        }
    })
    
    client = HPCClient(config)
    # ... test with real cluster
```

## Mocking SSH Connections

Use `unittest.mock` to mock SSH connections:

```python
with patch('asyncssh.connect', new_callable=AsyncMock) as mock:
    mock_conn = AsyncMock()
    mock.return_value = mock_conn
    
    # Your test code here
```

## Continuous Integration

Example GitHub Actions workflow:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install pytest pytest-asyncio asyncssh
      - run: python -m pytest tests/ -v -m "not integration"
```
