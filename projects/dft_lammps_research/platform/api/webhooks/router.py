"""
Webhook API Router

Endpoints for webhook subscription management
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel, Field, HttpUrl

from api_platform.gateway.main import gateway, limiter
from api_platform.webhooks.manager import WebhookEventType, webhook_manager

router = APIRouter()


class WebhookSubscribeRequest(BaseModel):
    """Subscribe to webhook events"""
    url: HttpUrl = Field(..., description="Webhook endpoint URL (HTTPS required in production)")
    events: List[str] = Field(
        ...,
        description="Event types to subscribe to. Use '*' for all events.",
        example=["calculation.completed", "project.completed"]
    )
    metadata: Optional[dict] = Field(None, description="Optional metadata for the webhook")
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://api.example.com/webhooks/dft-lammps",
                "events": ["calculation.completed", "calculation.failed"],
                "metadata": {"team": "materials-research"}
            }
        }


class WebhookResponse(BaseModel):
    """Webhook subscription response"""
    webhook_id: str
    url: str
    events: List[str]
    secret: str
    status: str
    created_at: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "webhook_id": "wh_abc123def456",
                "url": "https://api.example.com/webhooks/dft-lammps",
                "events": ["calculation.completed"],
                "secret": "whsec_xxxxxxxxxxxx",
                "status": "active",
                "created_at": "2024-01-15T10:30:00Z"
            }
        }


class WebhookListResponse(BaseModel):
    """Webhook list item"""
    webhook_id: str
    url: str
    events: List[str]
    status: str
    created_at: str
    updated_at: str
    failure_count: int
    last_delivery_at: Optional[str]
    metadata: Optional[dict]


class WebhookDeliveryResponse(BaseModel):
    """Webhook delivery status"""
    delivery_id: str
    event_id: str
    webhook_id: str
    status: str
    attempt_count: int
    created_at: str
    delivered_at: Optional[str]
    response_status: Optional[int]
    error_message: Optional[str]


class EventTypeInfo(BaseModel):
    """Event type information"""
    event_type: str
    description: str
    payload_schema: dict


@router.get(
    "/events",
    response_model=List[EventTypeInfo],
    summary="List event types",
    description="Get all available webhook event types"
)
async def list_event_types(
    auth: dict = Depends(gateway.authenticate_request)
):
    """
    List all available webhook event types with their descriptions and payload schemas.
    
    Subscribe to events using the `/subscribe` endpoint.
    """
    events = [
        EventTypeInfo(
            event_type=WebhookEventType.CALCULATION_COMPLETED.value,
            description="Triggered when a calculation completes successfully",
            payload_schema={
                "event_id": "string",
                "event_type": "calculation.completed",
                "timestamp": "ISO 8601 datetime",
                "data": {
                    "calculation_id": "string",
                    "project_id": "string",
                    "structure_id": "string",
                    "calculation_type": "string",
                    "results": {"object"},
                    "duration_seconds": "number"
                }
            }
        ),
        EventTypeInfo(
            event_type=WebhookEventType.CALCULATION_FAILED.value,
            description="Triggered when a calculation fails",
            payload_schema={
                "event_id": "string",
                "event_type": "calculation.failed",
                "timestamp": "ISO 8601 datetime",
                "data": {
                    "calculation_id": "string",
                    "project_id": "string",
                    "error_code": "string",
                    "error_message": "string",
                    "retryable": "boolean"
                }
            }
        ),
        EventTypeInfo(
            event_type=WebhookEventType.PROJECT_COMPLETED.value,
            description="Triggered when all calculations in a project are completed",
            payload_schema={
                "event_id": "string",
                "event_type": "project.completed",
                "timestamp": "ISO 8601 datetime",
                "data": {
                    "project_id": "string",
                    "total_calculations": "integer",
                    "completed_calculations": "integer",
                    "failed_calculations": "integer",
                    "duration_seconds": "number"
                }
            }
        ),
        EventTypeInfo(
            event_type=WebhookEventType.BATCH_COMPLETED.value,
            description="Triggered when a batch of calculations completes",
            payload_schema={
                "event_id": "string",
                "event_type": "batch.completed",
                "timestamp": "ISO 8601 datetime",
                "data": {
                    "batch_id": "string",
                    "project_id": "string",
                    "total": "integer",
                    "completed": "integer",
                    "failed": "integer"
                }
            }
        ),
        EventTypeInfo(
            event_type=WebhookEventType.QUOTA_EXCEEDED.value,
            description="Triggered when API quota is exceeded",
            payload_schema={
                "event_id": "string",
                "event_type": "system.quota_exceeded",
                "timestamp": "ISO 8601 datetime",
                "data": {
                    "quota_type": "string",
                    "current_usage": "number",
                    "limit": "number",
                    "reset_at": "ISO 8601 datetime"
                }
            }
        ),
    ]
    
    return events


@router.post(
    "/subscribe",
    response_model=WebhookResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Subscribe to webhooks",
    description="Create a new webhook subscription"
)
@limiter.limit("10/minute")
async def subscribe_webhook(
    request: Request,
    subscribe_data: WebhookSubscribeRequest,
    auth: dict = Depends(gateway.authenticate_request)
):
    """
    Create a new webhook subscription.
    
    The webhook secret will be returned only once. Store it securely for signature verification.
    
    **Event Types:**
    - `*` - All events
    - `calculation.completed` - Calculation completed
    - `calculation.failed` - Calculation failed
    - `project.completed` - Project completed
    - `batch.completed` - Batch completed
    - `system.quota_exceeded` - Quota exceeded
    
    **Signature Verification:**
    Webhooks are signed with HMAC-SHA256. Verify using the `X-Webhook-Signature` header:
    ```python
    import hmac
    import hashlib
    
    signature = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    
    expected = f"sha256={signature}"
    if not hmac.compare_digest(expected, request.headers["X-Webhook-Signature"]):
        raise ValueError("Invalid signature")
    ```
    """
    try:
        subscription = await webhook_manager.subscribe(
            client_id=auth["client_id"],
            url=str(subscribe_data.url),
            events=subscribe_data.events,
            metadata=subscribe_data.metadata
        )
        return WebhookResponse(**subscription)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get(
    "",
    response_model=List[WebhookListResponse],
    summary="List webhooks",
    description="List all webhook subscriptions"
)
async def list_webhooks(
    auth: dict = Depends(gateway.authenticate_request)
):
    """Get all webhook subscriptions for the authenticated user"""
    subscriptions = await webhook_manager.get_subscriptions(auth["client_id"])
    return subscriptions


@router.get(
    "/{webhook_id}",
    response_model=WebhookListResponse,
    summary="Get webhook",
    description="Get webhook subscription details"
)
async def get_webhook(
    webhook_id: str,
    auth: dict = Depends(gateway.authenticate_request)
):
    """Get webhook subscription by ID"""
    subscriptions = await webhook_manager.get_subscriptions(auth["client_id"])
    
    for sub in subscriptions:
        if sub["webhook_id"] == webhook_id:
            return WebhookListResponse(**sub)
    
    raise HTTPException(status_code=404, detail="Webhook not found")


@router.delete(
    "/{webhook_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Unsubscribe",
    description="Delete a webhook subscription"
)
async def unsubscribe_webhook(
    webhook_id: str,
    auth: dict = Depends(gateway.authenticate_request)
):
    """Delete a webhook subscription"""
    success = await webhook_manager.unsubscribe(webhook_id, auth["client_id"])
    
    if not success:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    return None


@router.get(
    "/{webhook_id}/deliveries",
    response_model=List[WebhookDeliveryResponse],
    summary="List deliveries",
    description="Get delivery history for a webhook"
)
async def list_deliveries(
    webhook_id: str,
    status: Optional[str] = None,
    auth: dict = Depends(gateway.authenticate_request)
):
    """
    Get delivery history for a webhook subscription.
    
    - **status**: Filter by delivery status (pending, delivered, failed)
    """
    # In production: query from database
    return []


@router.get(
    "/{webhook_id}/deliveries/{delivery_id}",
    response_model=WebhookDeliveryResponse,
    summary="Get delivery",
    description="Get delivery status"
)
async def get_delivery(
    webhook_id: str,
    delivery_id: str,
    auth: dict = Depends(gateway.authenticate_request)
):
    """Get delivery status by ID"""
    delivery = await webhook_manager.get_delivery(delivery_id, auth["client_id"])
    
    if not delivery:
        raise HTTPException(status_code=404, detail="Delivery not found")
    
    return WebhookDeliveryResponse(**delivery)


@router.post(
    "/{webhook_id}/deliveries/{delivery_id}/redeliver",
    response_model=WebhookDeliveryResponse,
    summary="Redeliver webhook",
    description="Retry a failed webhook delivery"
)
async def redeliver_webhook(
    webhook_id: str,
    delivery_id: str,
    auth: dict = Depends(gateway.authenticate_request)
):
    """Redeliver a failed webhook"""
    success = await webhook_manager.redeliver(delivery_id, auth["client_id"])
    
    if not success:
        raise HTTPException(status_code=404, detail="Delivery not found")
    
    # Return updated delivery
    delivery = await webhook_manager.get_delivery(delivery_id, auth["client_id"])
    return WebhookDeliveryResponse(**delivery)


@router.post(
    "/{webhook_id}/test",
    response_model=WebhookDeliveryResponse,
    summary="Test webhook",
    description="Send a test event to the webhook endpoint"
)
async def test_webhook(
    webhook_id: str,
    auth: dict = Depends(gateway.authenticate_request)
):
    """
    Send a test event to the webhook endpoint.
    
    Useful for verifying your endpoint is working correctly.
    """
    # In production: send test event
    return WebhookDeliveryResponse(
        delivery_id="test_delivery",
        event_id="test_event",
        webhook_id=webhook_id,
        status="delivered",
        attempt_count=1,
        created_at="2024-01-15T10:30:00Z",
        delivered_at="2024-01-15T10:30:01Z",
        response_status=200,
        error_message=None
    )
