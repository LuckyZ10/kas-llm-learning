"""
Webhook System

Event subscription and notification system with:
- Event subscription management
- Asynchronous notification delivery
- Retry mechanism with exponential backoff
- Idempotency guarantee
- Signature verification
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import hmac
import hashlib
import json
import secrets
import structlog
import uuid
import asyncio
from enum import Enum

import httpx

logger = structlog.get_logger()


class WebhookEventType(Enum):
    """Supported webhook event types"""
    # Project events
    PROJECT_CREATED = "project.created"
    PROJECT_UPDATED = "project.updated"
    PROJECT_DELETED = "project.deleted"
    PROJECT_COMPLETED = "project.completed"
    
    # Calculation events
    CALCULATION_SUBMITTED = "calculation.submitted"
    CALCULATION_STARTED = "calculation.started"
    CALCULATION_COMPLETED = "calculation.completed"
    CALCULATION_FAILED = "calculation.failed"
    
    # Structure events
    STRUCTURE_UPLOADED = "structure.uploaded"
    STRUCTURE_PROCESSED = "structure.processed"
    
    # Batch events
    BATCH_SUBMITTED = "batch.submitted"
    BATCH_COMPLETED = "batch.completed"
    BATCH_FAILED = "batch.failed"
    
    # System events
    RATE_LIMIT_WARNING = "system.rate_limit_warning"
    QUOTA_EXCEEDED = "system.quota_exceeded"


class WebhookStatus(Enum):
    """Webhook subscription status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"
    PENDING = "pending"


class WebhookDelivery:
    """Represents a webhook delivery attempt"""
    
    def __init__(
        self,
        delivery_id: str,
        webhook_id: str,
        event_id: str,
        payload: dict,
        url: str
    ):
        self.delivery_id = delivery_id
        self.webhook_id = webhook_id
        self.event_id = event_id
        self.payload = payload
        self.url = url
        self.attempt_count = 0
        self.created_at = datetime.utcnow()
        self.delivered_at: Optional[datetime] = None
        self.status = "pending"
        self.response_status: Optional[int] = None
        self.response_body: Optional[str] = None
        self.error_message: Optional[str] = None


class WebhookSubscription:
    """Represents a webhook subscription"""
    
    def __init__(
        self,
        webhook_id: str,
        client_id: str,
        url: str,
        events: List[str],
        secret: str,
        active: bool = True,
        metadata: Optional[dict] = None
    ):
        self.webhook_id = webhook_id
        self.client_id = client_id
        self.url = url
        self.events = events
        self.secret = secret
        self.active = active
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.failure_count = 0
        self.last_delivery_at: Optional[datetime] = None
        self.last_error: Optional[str] = None


class WebhookManager:
    """Manages webhook subscriptions and deliveries"""
    
    def __init__(self):
        self._subscriptions: Dict[str, WebhookSubscription] = {}
        self._client_webhooks: Dict[str, List[str]] = {}
        self._deliveries: Dict[str, WebhookDelivery] = {}
        self._event_webhooks: Dict[str, List[str]] = {}
        self._http_client: Optional[httpx.AsyncClient] = None
        
        # Configuration
        self.max_retries = 5
        self.retry_delays = [60, 300, 900, 3600, 7200]  # Exponential backoff
        self.timeout_seconds = 30
        self.signature_header = "X-Webhook-Signature"
        self.event_id_header = "X-Event-ID"
        self.delivery_id_header = "X-Delivery-ID"
    
    async def init_http_client(self):
        """Initialize HTTP client"""
        if not self._http_client:
            self._http_client = httpx.AsyncClient(
                timeout=self.timeout_seconds,
                follow_redirects=False,
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
            )
    
    async def close_http_client(self):
        """Close HTTP client"""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
    
    def generate_secret(self) -> str:
        """Generate a secure webhook secret"""
        return f"whsec_{secrets.token_urlsafe(32)}"
    
    def create_signature(self, payload: str, secret: str) -> str:
        """
        Create HMAC signature for webhook payload
        
        Uses SHA-256 for signature generation
        """
        signature = hmac.new(
            secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"
    
    def verify_signature(self, payload: str, signature: str, secret: str) -> bool:
        """Verify webhook signature"""
        expected = self.create_signature(payload, secret)
        return hmac.compare_digest(expected, signature)
    
    async def subscribe(
        self,
        client_id: str,
        url: str,
        events: List[str],
        metadata: Optional[dict] = None
    ) -> dict:
        """
        Create a new webhook subscription
        
        Args:
            client_id: Client identifier
            url: Webhook endpoint URL (must use HTTPS in production)
            events: List of event types to subscribe to
            metadata: Optional metadata for the subscription
        
        Returns:
            Subscription details including the secret
        """
        # Validate events
        valid_events = [e.value for e in WebhookEventType]
        invalid_events = [e for e in events if e not in valid_events and e != "*"]
        if invalid_events:
            raise ValueError(f"Invalid event types: {invalid_events}")
        
        webhook_id = f"wh_{uuid.uuid4().hex[:16]}"
        secret = self.generate_secret()
        
        subscription = WebhookSubscription(
            webhook_id=webhook_id,
            client_id=client_id,
            url=url,
            events=events,
            secret=secret,
            metadata=metadata
        )
        
        self._subscriptions[webhook_id] = subscription
        
        # Index by client
        if client_id not in self._client_webhooks:
            self._client_webhooks[client_id] = []
        self._client_webhooks[client_id].append(webhook_id)
        
        # Index by event
        for event in events:
            if event not in self._event_webhooks:
                self._event_webhooks[event] = []
            self._event_webhooks[event].append(webhook_id)
        
        logger.info(
            "webhook_subscribed",
            webhook_id=webhook_id,
            client_id=client_id,
            url=url,
            events=events
        )
        
        # Return subscription (secret is only shown once)
        return {
            "webhook_id": webhook_id,
            "url": url,
            "events": events,
            "secret": secret,
            "status": "active",
            "created_at": subscription.created_at.isoformat(),
        }
    
    async def unsubscribe(self, webhook_id: str, client_id: str) -> bool:
        """Delete a webhook subscription"""
        subscription = self._subscriptions.get(webhook_id)
        
        if not subscription or subscription.client_id != client_id:
            return False
        
        # Remove from indexes
        self._client_webhooks[client_id].remove(webhook_id)
        
        for event in subscription.events:
            if event in self._event_webhooks:
                if webhook_id in self._event_webhooks[event]:
                    self._event_webhooks[event].remove(webhook_id)
        
        del self._subscriptions[webhook_id]
        
        logger.info("webhook_unsubscribed", webhook_id=webhook_id, client_id=client_id)
        return True
    
    async def get_subscriptions(self, client_id: str) -> List[dict]:
        """Get all subscriptions for a client (without secrets)"""
        webhook_ids = self._client_webhooks.get(client_id, [])
        subscriptions = []
        
        for webhook_id in webhook_ids:
            sub = self._subscriptions.get(webhook_id)
            if sub:
                subscriptions.append({
                    "webhook_id": sub.webhook_id,
                    "url": sub.url,
                    "events": sub.events,
                    "status": "active" if sub.active else "inactive",
                    "created_at": sub.created_at.isoformat(),
                    "updated_at": sub.updated_at.isoformat(),
                    "failure_count": sub.failure_count,
                    "last_delivery_at": sub.last_delivery_at.isoformat() if sub.last_delivery_at else None,
                    "metadata": sub.metadata,
                })
        
        return subscriptions
    
    async def emit_event(
        self,
        event_type: WebhookEventType,
        payload: dict,
        project_id: Optional[str] = None
    ) -> List[str]:
        """
        Emit an event to all subscribed webhooks
        
        Returns:
            List of delivery IDs
        """
        event_id = f"evt_{uuid.uuid4().hex}"
        
        # Find all matching webhooks
        webhook_ids = self._event_webhooks.get(event_type.value, [])
        wildcard_ids = self._event_webhooks.get("*", [])
        all_webhook_ids = set(webhook_ids + wildcard_ids)
        
        delivery_ids = []
        
        for webhook_id in all_webhook_ids:
            subscription = self._subscriptions.get(webhook_id)
            if not subscription or not subscription.active:
                continue
            
            # Create delivery
            delivery_id = f"del_{uuid.uuid4().hex}"
            delivery = WebhookDelivery(
                delivery_id=delivery_id,
                webhook_id=webhook_id,
                event_id=event_id,
                payload={
                    "event_id": event_id,
                    "event_type": event_type.value,
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": payload,
                },
                url=subscription.url
            )
            
            self._deliveries[delivery_id] = delivery
            delivery_ids.append(delivery_id)
            
            # Queue for delivery
            asyncio.create_task(self._deliver_webhook(delivery, subscription))
        
        logger.info(
            "event_emitted",
            event_id=event_id,
            event_type=event_type.value,
            webhook_count=len(delivery_ids)
        )
        
        return delivery_ids
    
    async def _deliver_webhook(
        self,
        delivery: WebhookDelivery,
        subscription: WebhookSubscription
    ):
        """Deliver a webhook with retry logic"""
        await self.init_http_client()
        
        payload_json = json.dumps(delivery.payload, separators=(",", ":"), default=str)
        signature = self.create_signature(payload_json, subscription.secret)
        
        headers = {
            "Content-Type": "application/json",
            self.signature_header: signature,
            self.event_id_header: delivery.event_id,
            self.delivery_id_header: delivery.delivery_id,
            "User-Agent": "DFT-LAMMPS-Webhook/1.0",
        }
        
        for attempt in range(self.max_retries):
            delivery.attempt_count = attempt + 1
            
            try:
                response = await self._http_client.post(
                    delivery.url,
                    content=payload_json,
                    headers=headers
                )
                
                delivery.response_status = response.status_code
                delivery.response_body = response.text[:1000]  # Truncate
                
                # Success: 2xx status code
                if 200 <= response.status_code < 300:
                    delivery.status = "delivered"
                    delivery.delivered_at = datetime.utcnow()
                    subscription.last_delivery_at = datetime.utcnow()
                    subscription.failure_count = 0
                    
                    logger.info(
                        "webhook_delivered",
                        delivery_id=delivery.delivery_id,
                        webhook_id=subscription.webhook_id,
                        attempts=delivery.attempt_count,
                        status=response.status_code
                    )
                    return
                
                # Client error: don't retry
                if 400 <= response.status_code < 500:
                    delivery.status = "failed"
                    delivery.error_message = f"Client error: {response.status_code}"
                    
                    logger.warning(
                        "webhook_client_error",
                        delivery_id=delivery.delivery_id,
                        status=response.status_code
                    )
                    break
                
                # Server error: retry
                delivery.error_message = f"Server error: {response.status_code}"
                
            except httpx.TimeoutException:
                delivery.error_message = "Request timeout"
                logger.warning("webhook_timeout", delivery_id=delivery.delivery_id)
                
            except Exception as e:
                delivery.error_message = str(e)
                logger.error("webhook_delivery_error", delivery_id=delivery.delivery_id, error=str(e))
            
            # Wait before retry (exponential backoff)
            if attempt < self.max_retries - 1:
                delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                await asyncio.sleep(delay)
        
        # Max retries reached
        delivery.status = "failed"
        subscription.failure_count += 1
        subscription.last_error = delivery.error_message
        
        # Deactivate webhook after too many failures
        if subscription.failure_count >= 10:
            subscription.active = False
            logger.warning(
                "webhook_deactivated",
                webhook_id=subscription.webhook_id,
                failure_count=subscription.failure_count
            )
        
        logger.error(
            "webhook_delivery_failed",
            delivery_id=delivery.delivery_id,
            webhook_id=subscription.webhook_id,
            attempts=delivery.attempt_count
        )
    
    async def get_delivery(self, delivery_id: str, client_id: str) -> Optional[dict]:
        """Get delivery status"""
        delivery = self._deliveries.get(delivery_id)
        
        if not delivery:
            return None
        
        # Verify ownership
        subscription = self._subscriptions.get(delivery.webhook_id)
        if not subscription or subscription.client_id != client_id:
            return None
        
        return {
            "delivery_id": delivery.delivery_id,
            "event_id": delivery.event_id,
            "webhook_id": delivery.webhook_id,
            "status": delivery.status,
            "attempt_count": delivery.attempt_count,
            "created_at": delivery.created_at.isoformat(),
            "delivered_at": delivery.delivered_at.isoformat() if delivery.delivered_at else None,
            "response_status": delivery.response_status,
            "error_message": delivery.error_message,
        }
    
    async def redeliver(self, delivery_id: str, client_id: str) -> bool:
        """Redeliver a failed webhook"""
        delivery = self._deliveries.get(delivery_id)
        
        if not delivery:
            return False
        
        subscription = self._subscriptions.get(delivery.webhook_id)
        if not subscription or subscription.client_id != client_id:
            return False
        
        # Reset delivery status
        delivery.status = "pending"
        delivery.attempt_count = 0
        delivery.error_message = None
        
        # Queue for redelivery
        asyncio.create_task(self._deliver_webhook(delivery, subscription))
        
        return True


# Global webhook manager
webhook_manager = WebhookManager()
