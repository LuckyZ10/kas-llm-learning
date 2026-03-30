"""
DFT+LAMMPS Python SDK

Official Python client for the DFT+LAMMPS API Platform.

Installation:
    pip install dft-lammps-client

Quick Start:
    >>> from dft_lammps import Client
    >>> client = Client(api_key="your-api-key")
    >>> project = client.projects.create(name="My Project")
    >>> calculation = client.calculations.submit(
    ...     project_id=project.id,
    ...     structure="Li2S.cif",
    ...     calculation_type="dft"
    ... )
"""

__version__ = "1.0.0"

import os
import json
import time
import hmac
import hashlib
from typing import Optional, Dict, List, Any, Iterator, Union
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urljoin, urlencode

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Default configuration
DEFAULT_BASE_URL = "https://api.dft-lammps.org"
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 3


class DFTLAMMPSError(Exception):
    """Base exception for SDK errors"""
    pass


class AuthenticationError(DFTLAMMPSError):
    """Authentication failed"""
    pass


class RateLimitError(DFTLAMMPSError):
    """Rate limit exceeded"""
    pass


class NotFoundError(DFTLAMMPSError):
    """Resource not found"""
    pass


class ValidationError(DFTLAMMPSError):
    """Validation failed"""
    pass


@dataclass
class Project:
    """Project resource"""
    id: str
    name: str
    description: Optional[str]
    project_type: str
    status: str
    target_properties: Dict[str, Any]
    material_system: Optional[str]
    tags: List[str]
    total_structures: int
    completed_calculations: int
    failed_calculations: int
    created_at: datetime
    updated_at: Optional[datetime]
    owner_id: str


@dataclass
class Calculation:
    """Calculation resource"""
    id: str
    project_id: str
    calculation_type: str
    status: str
    structure: Dict[str, Any]
    parameters: Dict[str, Any]
    priority: int
    results: Optional[Dict[str, Any]]
    error_message: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]


@dataclass
class Structure:
    """Structure resource"""
    id: str
    name: str
    format: str
    cell: List[List[float]]
    species: List[str]
    positions: List[List[float]]
    metadata: Dict[str, Any]
    created_at: datetime


@dataclass
class WebhookSubscription:
    """Webhook subscription"""
    webhook_id: str
    url: str
    events: List[str]
    status: str
    created_at: str


class HTTPClient:
    """HTTP client with retries and authentication"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES
    ):
        self.api_key = api_key or os.getenv("DFT_LAMMPS_API_KEY")
        if not self.api_key:
            raise AuthenticationError("API key required. Set DFT_LAMMPS_API_KEY env var or pass to constructor.")
        
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"dft-lammps-python/{__version__}",
        })
        
        # Configure retries
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
    
    def _handle_response(self, response: requests.Response) -> dict:
        """Handle API response and errors"""
        if response.status_code == 200 or response.status_code == 201:
            return response.json() if response.content else {}
        elif response.status_code == 204:
            return {}
        elif response.status_code == 400:
            raise ValidationError(response.json().get("detail", "Validation failed"))
        elif response.status_code == 401:
            raise AuthenticationError("Invalid API key")
        elif response.status_code == 403:
            raise AuthenticationError("Permission denied")
        elif response.status_code == 404:
            raise NotFoundError("Resource not found")
        elif response.status_code == 429:
            raise RateLimitError("Rate limit exceeded. Try again later.")
        else:
            raise DFTLAMMPSError(f"API error: {response.status_code} - {response.text}")
    
    def request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict] = None,
        json_data: Optional[dict] = None,
        **kwargs
    ) -> dict:
        """Make HTTP request"""
        url = urljoin(self.base_url + "/", endpoint.lstrip("/"))
        
        response = self.session.request(
            method=method,
            url=url,
            params=params,
            json=json_data,
            timeout=self.timeout,
            **kwargs
        )
        
        return self._handle_response(response)
    
    def get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """GET request"""
        return self.request("GET", endpoint, params=params)
    
    def post(self, endpoint: str, json_data: Optional[dict] = None) -> dict:
        """POST request"""
        return self.request("POST", endpoint, json_data=json_data)
    
    def patch(self, endpoint: str, json_data: Optional[dict] = None) -> dict:
        """PATCH request"""
        return self.request("PATCH", endpoint, json_data=json_data)
    
    def delete(self, endpoint: str) -> dict:
        """DELETE request"""
        return self.request("DELETE", endpoint)


class ProjectsAPI:
    """Projects API client"""
    
    def __init__(self, client: HTTPClient):
        self.client = client
    
    def list(
        self,
        status: Optional[str] = None,
        project_type: Optional[str] = None,
        search: Optional[str] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """List projects with pagination"""
        params = {"page": page, "page_size": page_size}
        if status:
            params["status"] = status
        if project_type:
            params["project_type"] = project_type
        if search:
            params["search"] = search
        
        return self.client.get("/api/v1/projects", params=params)
    
    def create(
        self,
        name: str,
        description: Optional[str] = None,
        project_type: str = "battery_screening",
        target_properties: Optional[Dict[str, Any]] = None,
        material_system: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Project:
        """Create a new project"""
        data = {
            "name": name,
            "description": description,
            "project_type": project_type,
            "target_properties": target_properties or {},
            "material_system": material_system,
            "tags": tags or [],
        }
        
        response = self.client.post("/api/v1/projects", json_data=data)
        return Project(**response)
    
    def get(self, project_id: str) -> Project:
        """Get project by ID"""
        response = self.client.get(f"/api/v1/projects/{project_id}")
        return Project(**response)
    
    def update(self, project_id: str, **kwargs) -> Project:
        """Update project"""
        response = self.client.patch(f"/api/v1/projects/{project_id}", json_data=kwargs)
        return Project(**response)
    
    def delete(self, project_id: str) -> None:
        """Delete project"""
        self.client.delete(f"/api/v1/projects/{project_id}")


class CalculationsAPI:
    """Calculations API client"""
    
    def __init__(self, client: HTTPClient):
        self.client = client
    
    def submit(
        self,
        project_id: str,
        structure: Union[str, Dict[str, Any]],
        calculation_type: str = "dft",
        parameters: Optional[Dict[str, Any]] = None,
        priority: int = 5
    ) -> Calculation:
        """
        Submit a calculation.
        
        Args:
            project_id: Project ID
            structure: Structure file path or structure data dict
            calculation_type: Type of calculation (dft, lammps, ml)
            parameters: Calculation parameters
            priority: Priority (1-10)
        """
        # Handle file upload
        if isinstance(structure, str) and os.path.isfile(structure):
            with open(structure, 'r') as f:
                structure_data = f.read()
            # Upload structure first
            structure_obj = self._upload_structure(project_id, structure, structure_data)
            structure_dict = {
                "id": structure_obj.id,
                "format": structure_obj.format,
            }
        else:
            structure_dict = structure if isinstance(structure, dict) else {"data": structure}
        
        data = {
            "structure": structure_dict,
            "calculation_type": calculation_type,
            "parameters": parameters or {},
            "priority": priority,
        }
        
        response = self.client.post(
            f"/api/v1/projects/{project_id}/calculations",
            json_data=data
        )
        return Calculation(**response)
    
    def _upload_structure(
        self,
        project_id: str,
        file_path: str,
        data: str
    ) -> Structure:
        """Upload structure file"""
        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        name = os.path.basename(file_path)
        
        payload = {
            "name": name,
            "format": ext if ext in ["poscar", "cif", "xyz", "json"] else "poscar",
            "data": data,
        }
        
        response = self.client.post(
            f"/api/v1/projects/{project_id}/structures",
            json_data=payload
        )
        return Structure(**response)
    
    def get(self, calculation_id: str) -> Calculation:
        """Get calculation by ID"""
        response = self.client.get(f"/api/v1/calculations/{calculation_id}")
        return Calculation(**response)
    
    def list(
        self,
        project_id: str,
        status: Optional[str] = None,
        page: int = 1,
        page_size: int = 20
    ) -> List[Calculation]:
        """List calculations for a project"""
        params = {"page": page, "page_size": page_size}
        if status:
            params["status"] = status
        
        response = self.client.get(
            f"/api/v1/projects/{project_id}/calculations",
            params=params
        )
        return [Calculation(**c) for c in response.get("items", [])]
    
    def wait(
        self,
        calculation_id: str,
        timeout: Optional[int] = None,
        poll_interval: int = 5
    ) -> Calculation:
        """
        Wait for calculation to complete.
        
        Args:
            calculation_id: Calculation ID
            timeout: Maximum wait time in seconds
            poll_interval: Seconds between status checks
        
        Returns:
            Completed calculation
        
        Raises:
            TimeoutError: If timeout is reached
        """
        start_time = time.time()
        
        while True:
            calc = self.get(calculation_id)
            
            if calc.status in ["completed", "failed"]:
                return calc
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Calculation {calculation_id} did not complete within {timeout}s")
            
            time.sleep(poll_interval)
    
    def submit_batch(
        self,
        project_id: str,
        calculations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Submit batch of calculations"""
        data = {
            "project_id": project_id,
            "calculations": calculations,
        }
        return self.client.post(
            f"/api/v1/projects/{project_id}/calculations/batch",
            json_data=data
        )


class WebhooksAPI:
    """Webhooks API client"""
    
    def __init__(self, client: HTTPClient):
        self.client = client
    
    def subscribe(
        self,
        url: str,
        events: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> WebhookSubscription:
        """
        Subscribe to webhook events.
        
        Args:
            url: Webhook endpoint URL
            events: List of event types or ["*"] for all events
            metadata: Optional metadata
        """
        data = {
            "url": url,
            "events": events,
            "metadata": metadata or {},
        }
        
        response = self.client.post("/api/v1/webhooks/subscribe", json_data=data)
        return WebhookSubscription(**response)
    
    def list(self) -> List[WebhookSubscription]:
        """List webhook subscriptions"""
        response = self.client.get("/api/v1/webhooks")
        return [WebhookSubscription(**s) for s in response]
    
    def delete(self, webhook_id: str) -> None:
        """Delete webhook subscription"""
        self.client.delete(f"/api/v1/webhooks/{webhook_id}")
    
    @staticmethod
    def verify_signature(payload: str, signature: str, secret: str) -> bool:
        """
        Verify webhook signature.
        
        Args:
            payload: Raw request body
            signature: Signature from X-Webhook-Signature header
            secret: Webhook secret from subscription
        
        Returns:
            True if signature is valid
        """
        expected = hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        expected_sig = f"sha256={expected}"
        
        return hmac.compare_digest(expected_sig, signature)


class Client:
    """
    DFT+LAMMPS API Client
    
    Main entry point for the SDK.
    
    Args:
        api_key: API key for authentication
        base_url: API base URL
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries
    
    Example:
        >>> client = Client(api_key="your-key")
        >>> project = client.projects.create(name="My Project")
        >>> print(project.id)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES
    ):
        self._client = HTTPClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries
        )
        
        # API resource clients
        self.projects = ProjectsAPI(self._client)
        self.calculations = CalculationsAPI(self._client)
        self.webhooks = WebhooksAPI(self._client)
    
    def health(self) -> Dict[str, Any]:
        """Check API health"""
        return self._client.get("/health")
    
    def usage(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return self._client.get("/api/v1/usage")


# Convenience exports
__all__ = [
    "Client",
    "Project",
    "Calculation",
    "Structure",
    "WebhookSubscription",
    "DFTLAMMPSError",
    "AuthenticationError",
    "RateLimitError",
    "NotFoundError",
    "ValidationError",
]
