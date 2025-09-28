#!/usr/bin/env python3
"""
🌐 SPIRITUAL EXTERNAL SERVICES ARCHITECTURE
Ladang Berkah Digital - ZeroLight Orbit System
Comprehensive Modular External Services Management

بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ
"""

import asyncio
import aiohttp
import json
import time
import uuid
import logging
import weakref
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, Set, Protocol
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import ssl
import certifi
from urllib.parse import urljoin, urlparse
import hashlib
import base64
from collections import defaultdict, deque
import threading
import queue

class ServiceType(Enum):
    """External service types"""
    HTTP_API = "http_api"
    WEBSOCKET = "websocket"
    DATABASE = "database"
    MESSAGE_QUEUE = "message_queue"
    FILE_STORAGE = "file_storage"
    AUTHENTICATION = "authentication"
    NOTIFICATION = "notification"
    ANALYTICS = "analytics"
    PAYMENT = "payment"
    EMAIL = "email"
    SMS = "sms"
    SOCIAL_MEDIA = "social_media"
    AI_SERVICE = "ai_service"
    BLOCKCHAIN = "blockchain"
    IOT_DEVICE = "iot_device"

class ServiceStatus(Enum):
    """Service connection status"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    MAINTENANCE = "maintenance"
    DEPRECATED = "deprecated"

class AuthenticationType(Enum):
    """Authentication methods"""
    NONE = "none"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    OAUTH2 = "oauth2"
    JWT = "jwt"
    CUSTOM = "custom"

class RetryStrategy(Enum):
    """Retry strategies for failed requests"""
    NONE = "none"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"
    CUSTOM = "custom"

@dataclass
class ServiceCredentials:
    """Service authentication credentials"""
    auth_type: AuthenticationType
    api_key: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    refresh_token: Optional[str] = None
    custom_headers: Dict[str, str] = field(default_factory=dict)
    expires_at: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Check if credentials are expired"""
        if self.expires_at:
            return datetime.now() >= self.expires_at
        return False
    
    def to_headers(self) -> Dict[str, str]:
        """Convert credentials to HTTP headers"""
        headers = self.custom_headers.copy()
        
        if self.auth_type == AuthenticationType.API_KEY and self.api_key:
            headers["X-API-Key"] = self.api_key
        elif self.auth_type == AuthenticationType.BEARER_TOKEN and self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        elif self.auth_type == AuthenticationType.BASIC_AUTH and self.username and self.password:
            credentials = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
            headers["Authorization"] = f"Basic {credentials}"
        elif self.auth_type == AuthenticationType.JWT and self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        
        return headers

@dataclass
class ServiceConfiguration:
    """External service configuration"""
    service_id: str
    service_name: str
    service_type: ServiceType
    base_url: str
    credentials: ServiceCredentials
    timeout: float = 30.0
    max_retries: int = 3
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    retry_delay: float = 1.0
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000
    health_check_interval: float = 300.0  # 5 minutes
    health_check_endpoint: Optional[str] = None
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    ssl_verify: bool = True
    custom_ssl_context: Optional[ssl.SSLContext] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization setup"""
        if not self.health_check_endpoint:
            self.health_check_endpoint = "/health"

@dataclass
class ServiceRequest:
    """Service request wrapper"""
    request_id: str
    service_id: str
    method: str
    endpoint: str
    data: Any = None
    params: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: Optional[float] = None
    retry_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    response: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ServiceResponse:
    """Service response wrapper"""
    request_id: str
    service_id: str
    status_code: int
    data: Any = None
    headers: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CircuitBreakerState:
    """Circuit breaker state management"""
    is_open: bool = False
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    next_attempt_time: Optional[datetime] = None
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

class ServiceInterface(Protocol):
    """Protocol for external service interfaces"""
    
    async def connect(self) -> bool:
        """Connect to the service"""
        ...
    
    async def disconnect(self) -> bool:
        """Disconnect from the service"""
        ...
    
    async def health_check(self) -> bool:
        """Check service health"""
        ...
    
    async def request(self, method: str, endpoint: str, **kwargs) -> ServiceResponse:
        """Make a request to the service"""
        ...

class BaseExternalService(ABC):
    """
    🌟 Base class for external service implementations
    """
    
    def __init__(self, config: ServiceConfiguration):
        """Initialize the service"""
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.service_id}")
        
        # Connection state
        self.status = ServiceStatus.DISCONNECTED
        self.session: Optional[aiohttp.ClientSession] = None
        self.connected_at: Optional[datetime] = None
        self.last_activity: datetime = datetime.now()
        
        # Rate limiting
        self.rate_limiter = {
            "minute": deque(),
            "hour": deque()
        }
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreakerState()
        
        # Request tracking
        self.active_requests: Dict[str, ServiceRequest] = {}
        self.request_history: deque = deque(maxlen=1000)
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0,
            "last_error": None,
            "uptime_seconds": 0.0
        }
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
    
    async def initialize(self):
        """Initialize the service"""
        try:
            self.logger.info(f"🚀 Initializing service: {self.config.service_name}")
            
            # Create HTTP session
            await self._create_session()
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Attempt initial connection
            await self.connect()
            
            self.logger.info(f"✅ Service initialized: {self.config.service_name}")
            
        except Exception as e:
            self.logger.error(f"💥 Failed to initialize service {self.config.service_name}: {e}")
            self.status = ServiceStatus.ERROR
            raise
    
    async def _create_session(self):
        """Create HTTP session with proper configuration"""
        try:
            # SSL context
            ssl_context = self.config.custom_ssl_context
            if not ssl_context and self.config.ssl_verify:
                ssl_context = ssl.create_default_context(cafile=certifi.where())
            
            # Connection timeout
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            
            # Create session
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=aiohttp.TCPConnector(
                    ssl=ssl_context if self.config.ssl_verify else False,
                    limit=100,
                    limit_per_host=20
                )
            )
            
            self.logger.info(f"🔗 Created HTTP session for {self.config.service_name}")
            
        except Exception as e:
            self.logger.error(f"💥 Failed to create session: {e}")
            raise
    
    async def _start_background_tasks(self):
        """Start background monitoring tasks"""
        try:
            # Health check task
            if self.config.health_check_interval > 0:
                task = asyncio.create_task(self._health_check_loop())
                self.background_tasks.add(task)
                task.add_done_callback(self.background_tasks.discard)
            
            # Rate limiter cleanup task
            task = asyncio.create_task(self._rate_limiter_cleanup())
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
            
            # Statistics update task
            task = asyncio.create_task(self._stats_update_loop())
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
            
            self.logger.info(f"🔄 Started background tasks for {self.config.service_name}")
            
        except Exception as e:
            self.logger.error(f"💥 Failed to start background tasks: {e}")
            raise
    
    async def connect(self) -> bool:
        """Connect to the external service"""
        try:
            if self.status == ServiceStatus.CONNECTED:
                return True
            
            self.logger.info(f"🔌 Connecting to {self.config.service_name}...")
            self.status = ServiceStatus.CONNECTING
            
            # Check circuit breaker
            if self._is_circuit_breaker_open():
                self.logger.warning(f"⚡ Circuit breaker is open for {self.config.service_name}")
                return False
            
            # Perform service-specific connection
            success = await self._perform_connection()
            
            if success:
                self.status = ServiceStatus.CONNECTED
                self.connected_at = datetime.now()
                self.circuit_breaker.successful_requests += 1
                self._reset_circuit_breaker()
                
                self.logger.info(f"✅ Connected to {self.config.service_name}")
                
                # Authenticate if required
                if self.config.credentials.auth_type != AuthenticationType.NONE:
                    auth_success = await self._authenticate()
                    if auth_success:
                        self.status = ServiceStatus.AUTHENTICATED
                        self.logger.info(f"🔐 Authenticated with {self.config.service_name}")
                    else:
                        self.logger.error(f"🚫 Authentication failed for {self.config.service_name}")
                        return False
                
                return True
            else:
                self.status = ServiceStatus.ERROR
                self._record_circuit_breaker_failure()
                return False
                
        except Exception as e:
            self.logger.error(f"💥 Connection failed for {self.config.service_name}: {e}")
            self.status = ServiceStatus.ERROR
            self._record_circuit_breaker_failure()
            return False
    
    @abstractmethod
    async def _perform_connection(self) -> bool:
        """Service-specific connection logic"""
        pass
    
    async def _authenticate(self) -> bool:
        """Authenticate with the service"""
        try:
            # Check if credentials are expired
            if self.config.credentials.is_expired():
                self.logger.warning(f"⏰ Credentials expired for {self.config.service_name}")
                # Attempt to refresh credentials
                if not await self._refresh_credentials():
                    return False
            
            # Perform authentication based on type
            if self.config.credentials.auth_type == AuthenticationType.OAUTH2:
                return await self._oauth2_authenticate()
            else:
                # For other auth types, credentials are included in headers
                return True
                
        except Exception as e:
            self.logger.error(f"💥 Authentication error for {self.config.service_name}: {e}")
            return False
    
    async def _oauth2_authenticate(self) -> bool:
        """Perform OAuth2 authentication"""
        try:
            # This is a simplified OAuth2 flow
            # In practice, you'd implement the full OAuth2 flow
            
            if not self.config.credentials.client_id or not self.config.credentials.client_secret:
                self.logger.error("❌ OAuth2 credentials missing")
                return False
            
            # Use refresh token if available
            if self.config.credentials.refresh_token:
                return await self._refresh_oauth2_token()
            
            # Otherwise, would need to implement authorization code flow
            self.logger.warning("⚠️ OAuth2 authorization code flow not implemented")
            return False
            
        except Exception as e:
            self.logger.error(f"💥 OAuth2 authentication error: {e}")
            return False
    
    async def _refresh_oauth2_token(self) -> bool:
        """Refresh OAuth2 access token"""
        try:
            token_url = urljoin(self.config.base_url, "/oauth/token")
            
            data = {
                "grant_type": "refresh_token",
                "refresh_token": self.config.credentials.refresh_token,
                "client_id": self.config.credentials.client_id,
                "client_secret": self.config.credentials.client_secret
            }
            
            async with self.session.post(token_url, data=data) as response:
                if response.status == 200:
                    token_data = await response.json()
                    
                    # Update credentials
                    self.config.credentials.token = token_data.get("access_token")
                    if "refresh_token" in token_data:
                        self.config.credentials.refresh_token = token_data["refresh_token"]
                    
                    # Set expiration
                    if "expires_in" in token_data:
                        expires_in = int(token_data["expires_in"])
                        self.config.credentials.expires_at = datetime.now() + timedelta(seconds=expires_in)
                    
                    self.logger.info(f"🔄 Refreshed OAuth2 token for {self.config.service_name}")
                    return True
                else:
                    self.logger.error(f"❌ Token refresh failed: {response.status}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"💥 Token refresh error: {e}")
            return False
    
    async def _refresh_credentials(self) -> bool:
        """Refresh expired credentials"""
        try:
            if self.config.credentials.auth_type == AuthenticationType.OAUTH2:
                return await self._refresh_oauth2_token()
            else:
                # For other auth types, credentials need to be updated externally
                self.logger.warning(f"⚠️ Cannot auto-refresh {self.config.credentials.auth_type.value} credentials")
                return False
                
        except Exception as e:
            self.logger.error(f"💥 Credential refresh error: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from the service"""
        try:
            self.logger.info(f"🔌 Disconnecting from {self.config.service_name}...")
            
            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Close session
            if self.session:
                await self.session.close()
                self.session = None
            
            # Update status
            self.status = ServiceStatus.DISCONNECTED
            self.connected_at = None
            
            self.logger.info(f"✅ Disconnected from {self.config.service_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"💥 Disconnection error for {self.config.service_name}: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check service health"""
        try:
            if not self.session or self.status == ServiceStatus.DISCONNECTED:
                return False
            
            # Use configured health check endpoint
            health_url = urljoin(self.config.base_url, self.config.health_check_endpoint)
            
            async with self.session.get(health_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                is_healthy = 200 <= response.status < 300
                
                if is_healthy:
                    self.logger.debug(f"💚 Health check passed for {self.config.service_name}")
                else:
                    self.logger.warning(f"💛 Health check failed for {self.config.service_name}: {response.status}")
                
                return is_healthy
                
        except Exception as e:
            self.logger.error(f"💥 Health check error for {self.config.service_name}: {e}")
            return False
    
    async def request(self, method: str, endpoint: str, **kwargs) -> ServiceResponse:
        """Make a request to the service"""
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        
        try:
            # Check rate limits
            if not self._check_rate_limits():
                raise Exception("Rate limit exceeded")
            
            # Check circuit breaker
            if self._is_circuit_breaker_open():
                raise Exception("Circuit breaker is open")
            
            # Ensure connection
            if self.status not in [ServiceStatus.CONNECTED, ServiceStatus.AUTHENTICATED]:
                if not await self.connect():
                    raise Exception("Service not connected")
            
            # Create request
            request = ServiceRequest(
                request_id=request_id,
                service_id=self.config.service_id,
                method=method.upper(),
                endpoint=endpoint,
                **kwargs
            )
            
            # Execute request with retries
            response = await self._execute_request_with_retries(request)
            
            # Update statistics
            self._update_request_stats(response)
            
            return response
            
        except Exception as e:
            # Create error response
            response = ServiceResponse(
                request_id=request_id,
                service_id=self.config.service_id,
                status_code=0,
                error=str(e)
            )
            
            # Update statistics
            self._update_request_stats(response)
            self._record_circuit_breaker_failure()
            
            self.logger.error(f"💥 Request failed for {self.config.service_name}: {e}")
            return response
    
    async def _execute_request_with_retries(self, request: ServiceRequest) -> ServiceResponse:
        """Execute request with retry logic"""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                request.retry_count = attempt
                response = await self._execute_single_request(request)
                
                # Check if response indicates success
                if 200 <= response.status_code < 300:
                    self.circuit_breaker.successful_requests += 1
                    return response
                elif response.status_code == 429:  # Rate limited
                    self.status = ServiceStatus.RATE_LIMITED
                    if attempt < self.config.max_retries:
                        await self._wait_for_retry(attempt)
                        continue
                else:
                    # Server error, might be worth retrying
                    if attempt < self.config.max_retries and response.status_code >= 500:
                        await self._wait_for_retry(attempt)
                        continue
                
                return response
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.config.max_retries:
                    self.logger.warning(f"⚠️ Request attempt {attempt + 1} failed, retrying: {e}")
                    await self._wait_for_retry(attempt)
                else:
                    self.logger.error(f"💥 All retry attempts failed for {request.request_id}")
        
        # All retries failed
        raise last_exception or Exception("Request failed after all retries")
    
    async def _execute_single_request(self, request: ServiceRequest) -> ServiceResponse:
        """Execute a single HTTP request"""
        start_time = time.time()
        
        try:
            # Prepare URL
            url = urljoin(self.config.base_url, request.endpoint)
            
            # Prepare headers
            headers = self.config.credentials.to_headers()
            headers.update(request.headers)
            
            # Prepare timeout
            timeout = request.timeout or self.config.timeout
            
            # Track request
            request.started_at = datetime.now()
            self.active_requests[request.request_id] = request
            
            # Execute request
            async with self.session.request(
                request.method,
                url,
                json=request.data if request.method in ["POST", "PUT", "PATCH"] else None,
                params=request.params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                
                # Read response data
                try:
                    if response.content_type == "application/json":
                        data = await response.json()
                    else:
                        data = await response.text()
                except:
                    data = None
                
                # Calculate duration
                duration_ms = (time.time() - start_time) * 1000
                
                # Create response
                service_response = ServiceResponse(
                    request_id=request.request_id,
                    service_id=self.config.service_id,
                    status_code=response.status,
                    data=data,
                    headers=dict(response.headers),
                    duration_ms=duration_ms
                )
                
                # Update request
                request.completed_at = datetime.now()
                request.response = service_response
                
                return service_response
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            # Create error response
            service_response = ServiceResponse(
                request_id=request.request_id,
                service_id=self.config.service_id,
                status_code=0,
                error=str(e),
                duration_ms=duration_ms
            )
            
            # Update request
            request.completed_at = datetime.now()
            request.error = str(e)
            
            raise
            
        finally:
            # Remove from active requests
            if request.request_id in self.active_requests:
                del self.active_requests[request.request_id]
            
            # Add to history
            self.request_history.append(request)
            
            # Update activity timestamp
            self.last_activity = datetime.now()
    
    async def _wait_for_retry(self, attempt: int):
        """Wait before retrying based on strategy"""
        try:
            if self.config.retry_strategy == RetryStrategy.LINEAR:
                delay = self.config.retry_delay * (attempt + 1)
            elif self.config.retry_strategy == RetryStrategy.EXPONENTIAL:
                delay = self.config.retry_delay * (2 ** attempt)
            elif self.config.retry_strategy == RetryStrategy.FIBONACCI:
                fib = [1, 1]
                for i in range(2, attempt + 2):
                    fib.append(fib[i-1] + fib[i-2])
                delay = self.config.retry_delay * fib[attempt]
            else:
                delay = self.config.retry_delay
            
            # Cap maximum delay
            delay = min(delay, 60.0)  # Max 1 minute
            
            self.logger.info(f"⏳ Waiting {delay:.1f}s before retry attempt {attempt + 2}")
            await asyncio.sleep(delay)
            
        except Exception as e:
            self.logger.error(f"💥 Error in retry delay: {e}")
    
    def _check_rate_limits(self) -> bool:
        """Check if request is within rate limits"""
        try:
            current_time = time.time()
            
            # Clean old entries
            minute_ago = current_time - 60
            hour_ago = current_time - 3600
            
            while self.rate_limiter["minute"] and self.rate_limiter["minute"][0] < minute_ago:
                self.rate_limiter["minute"].popleft()
            
            while self.rate_limiter["hour"] and self.rate_limiter["hour"][0] < hour_ago:
                self.rate_limiter["hour"].popleft()
            
            # Check limits
            if len(self.rate_limiter["minute"]) >= self.config.rate_limit_per_minute:
                self.logger.warning(f"⚠️ Rate limit exceeded (per minute) for {self.config.service_name}")
                return False
            
            if len(self.rate_limiter["hour"]) >= self.config.rate_limit_per_hour:
                self.logger.warning(f"⚠️ Rate limit exceeded (per hour) for {self.config.service_name}")
                return False
            
            # Add current request
            self.rate_limiter["minute"].append(current_time)
            self.rate_limiter["hour"].append(current_time)
            
            return True
            
        except Exception as e:
            self.logger.error(f"💥 Rate limit check error: {e}")
            return True  # Allow request on error
    
    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open"""
        try:
            if not self.circuit_breaker.is_open:
                return False
            
            # Check if we can attempt again
            if (self.circuit_breaker.next_attempt_time and 
                datetime.now() >= self.circuit_breaker.next_attempt_time):
                
                self.logger.info(f"🔄 Circuit breaker half-open for {self.config.service_name}")
                self.circuit_breaker.is_open = False
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"💥 Circuit breaker check error: {e}")
            return False
    
    def _record_circuit_breaker_failure(self):
        """Record a failure for circuit breaker"""
        try:
            self.circuit_breaker.failure_count += 1
            self.circuit_breaker.failed_requests += 1
            self.circuit_breaker.last_failure_time = datetime.now()
            
            # Open circuit breaker if threshold reached
            if self.circuit_breaker.failure_count >= self.config.circuit_breaker_threshold:
                self.circuit_breaker.is_open = True
                self.circuit_breaker.next_attempt_time = (
                    datetime.now() + timedelta(seconds=self.config.circuit_breaker_timeout)
                )
                
                self.logger.warning(f"⚡ Circuit breaker opened for {self.config.service_name}")
                
        except Exception as e:
            self.logger.error(f"💥 Circuit breaker failure recording error: {e}")
    
    def _reset_circuit_breaker(self):
        """Reset circuit breaker on successful request"""
        try:
            if self.circuit_breaker.failure_count > 0:
                self.logger.info(f"🔄 Circuit breaker reset for {self.config.service_name}")
            
            self.circuit_breaker.failure_count = 0
            self.circuit_breaker.is_open = False
            self.circuit_breaker.next_attempt_time = None
            
        except Exception as e:
            self.logger.error(f"💥 Circuit breaker reset error: {e}")
    
    def _update_request_stats(self, response: ServiceResponse):
        """Update request statistics"""
        try:
            self.stats["total_requests"] += 1
            
            if response.error:
                self.stats["failed_requests"] += 1
                self.stats["last_error"] = response.error
            else:
                self.stats["successful_requests"] += 1
            
            # Update average response time
            if response.duration_ms > 0:
                total_requests = self.stats["total_requests"]
                current_avg = self.stats["avg_response_time"]
                
                self.stats["avg_response_time"] = (
                    (current_avg * (total_requests - 1) + response.duration_ms) / total_requests
                )
            
        except Exception as e:
            self.logger.error(f"💥 Stats update error: {e}")
    
    async def _health_check_loop(self):
        """Background health check loop"""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                is_healthy = await self.health_check()
                
                if not is_healthy and self.status in [ServiceStatus.CONNECTED, ServiceStatus.AUTHENTICATED]:
                    self.logger.warning(f"💛 Health check failed for {self.config.service_name}")
                    # Attempt reconnection
                    await self.connect()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"💥 Health check loop error: {e}")
    
    async def _rate_limiter_cleanup(self):
        """Background rate limiter cleanup"""
        while True:
            try:
                await asyncio.sleep(60)  # Clean every minute
                self._check_rate_limits()  # This also cleans old entries
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"💥 Rate limiter cleanup error: {e}")
    
    async def _stats_update_loop(self):
        """Background statistics update loop"""
        while True:
            try:
                await asyncio.sleep(60)  # Update every minute
                
                # Update uptime
                if self.connected_at:
                    self.stats["uptime_seconds"] = (
                        datetime.now() - self.connected_at
                    ).total_seconds()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"💥 Stats update loop error: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        try:
            return {
                "service_id": self.config.service_id,
                "service_name": self.config.service_name,
                "service_type": self.config.service_type.value,
                "status": self.status.value,
                "base_url": self.config.base_url,
                "connected_at": self.connected_at.isoformat() if self.connected_at else None,
                "last_activity": self.last_activity.isoformat(),
                "active_requests": len(self.active_requests),
                "circuit_breaker": {
                    "is_open": self.circuit_breaker.is_open,
                    "failure_count": self.circuit_breaker.failure_count,
                    "total_requests": self.circuit_breaker.total_requests,
                    "successful_requests": self.circuit_breaker.successful_requests,
                    "failed_requests": self.circuit_breaker.failed_requests
                },
                "rate_limits": {
                    "per_minute": len(self.rate_limiter["minute"]),
                    "per_hour": len(self.rate_limiter["hour"]),
                    "limit_per_minute": self.config.rate_limit_per_minute,
                    "limit_per_hour": self.config.rate_limit_per_hour
                },
                "statistics": self.stats.copy(),
                "metadata": self.config.metadata
            }
            
        except Exception as e:
            self.logger.error(f"💥 Error getting service status: {e}")
            return {
                "service_id": self.config.service_id,
                "error": str(e)
            }

class HTTPAPIService(BaseExternalService):
    """
    🌐 HTTP API Service Implementation
    """
    
    async def _perform_connection(self) -> bool:
        """Perform HTTP API connection test"""
        try:
            # Test connection with a simple request
            test_url = urljoin(self.config.base_url, "/")
            
            async with self.session.get(test_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                # Consider any response as successful connection
                return True
                
        except Exception as e:
            self.logger.error(f"💥 HTTP API connection test failed: {e}")
            return False
    
    async def get(self, endpoint: str, **kwargs) -> ServiceResponse:
        """Make GET request"""
        return await self.request("GET", endpoint, **kwargs)
    
    async def post(self, endpoint: str, data: Any = None, **kwargs) -> ServiceResponse:
        """Make POST request"""
        return await self.request("POST", endpoint, data=data, **kwargs)
    
    async def put(self, endpoint: str, data: Any = None, **kwargs) -> ServiceResponse:
        """Make PUT request"""
        return await self.request("PUT", endpoint, data=data, **kwargs)
    
    async def patch(self, endpoint: str, data: Any = None, **kwargs) -> ServiceResponse:
        """Make PATCH request"""
        return await self.request("PATCH", endpoint, data=data, **kwargs)
    
    async def delete(self, endpoint: str, **kwargs) -> ServiceResponse:
        """Make DELETE request"""
        return await self.request("DELETE", endpoint, **kwargs)

class WebSocketService(BaseExternalService):
    """
    🔌 WebSocket Service Implementation
    """
    
    def __init__(self, config: ServiceConfiguration):
        super().__init__(config)
        self.websocket: Optional[aiohttp.ClientWebSocketResponse] = None
        self.message_handlers: Dict[str, Callable] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
    
    async def _perform_connection(self) -> bool:
        """Perform WebSocket connection"""
        try:
            ws_url = self.config.base_url.replace("http://", "ws://").replace("https://", "wss://")
            
            self.websocket = await self.session.ws_connect(
                ws_url,
                headers=self.config.credentials.to_headers(),
                timeout=self.config.timeout
            )
            
            # Start message handling loop
            task = asyncio.create_task(self._message_handler_loop())
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
            
            return True
            
        except Exception as e:
            self.logger.error(f"💥 WebSocket connection failed: {e}")
            return False
    
    async def send_message(self, message: Any) -> bool:
        """Send message through WebSocket"""
        try:
            if not self.websocket or self.websocket.closed:
                return False
            
            if isinstance(message, dict):
                await self.websocket.send_json(message)
            else:
                await self.websocket.send_str(str(message))
            
            return True
            
        except Exception as e:
            self.logger.error(f"💥 WebSocket send error: {e}")
            return False
    
    async def _message_handler_loop(self):
        """Handle incoming WebSocket messages"""
        try:
            async for msg in self.websocket:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except:
                        data = msg.data
                    
                    await self.message_queue.put(data)
                    
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    self.logger.error(f"💥 WebSocket error: {self.websocket.exception()}")
                    break
                    
        except Exception as e:
            self.logger.error(f"💥 WebSocket message handler error: {e}")
    
    def register_message_handler(self, message_type: str, handler: Callable):
        """Register a message handler"""
        self.message_handlers[message_type] = handler
    
    async def get_next_message(self, timeout: float = None) -> Any:
        """Get next message from queue"""
        try:
            return await asyncio.wait_for(self.message_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

class SpiritualExternalServicesManager:
    """
    🌟 Spiritual External Services Manager
    
    Comprehensive management of all external service connections
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the services manager"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Service registry
        self.services: Dict[str, BaseExternalService] = {}
        self.service_configs: Dict[str, ServiceConfiguration] = {}
        
        # Service factories
        self.service_factories = {
            ServiceType.HTTP_API: HTTPAPIService,
            ServiceType.WEBSOCKET: WebSocketService,
            # Add more service types as needed
        }
        
        # System locks
        self.locks = {
            "services": asyncio.Lock(),
            "configs": asyncio.Lock()
        }
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        self.shutdown_event = asyncio.Event()
        
        # Statistics
        self.system_stats = {
            "total_services": 0,
            "active_services": 0,
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "started_at": datetime.now()
        }
    
    async def initialize(self):
        """Initialize the services manager"""
        try:
            self.logger.info("🚀 Initializing Spiritual External Services Manager...")
            
            # Start background monitoring
            await self._start_background_monitoring()
            
            self.logger.info("✅ Services manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"💥 Failed to initialize services manager: {e}")
            raise
    
    async def register_service(self, config: ServiceConfiguration) -> bool:
        """Register a new external service"""
        try:
            async with self.locks["services"]:
                if config.service_id in self.services:
                    self.logger.warning(f"⚠️ Service {config.service_id} already registered")
                    return False
                
                # Get service factory
                service_factory = self.service_factories.get(config.service_type)
                if not service_factory:
                    raise ValueError(f"Unsupported service type: {config.service_type}")
                
                # Create service instance
                service = service_factory(config)
                
                # Initialize service
                await service.initialize()
                
                # Store service
                self.services[config.service_id] = service
                self.service_configs[config.service_id] = config
                
                self.system_stats["total_services"] += 1
                self.system_stats["active_services"] += 1
                
                self.logger.info(f"📝 Registered service: {config.service_name} ({config.service_type.value})")
                
                return True
                
        except Exception as e:
            self.logger.error(f"💥 Failed to register service {config.service_id}: {e}")
            return False
    
    async def unregister_service(self, service_id: str) -> bool:
        """Unregister an external service"""
        try:
            async with self.locks["services"]:
                if service_id not in self.services:
                    self.logger.warning(f"⚠️ Service {service_id} not found")
                    return False
                
                service = self.services[service_id]
                
                # Disconnect service
                await service.disconnect()
                
                # Remove from registry
                del self.services[service_id]
                del self.service_configs[service_id]
                
                self.system_stats["active_services"] -= 1
                
                self.logger.info(f"🗑️ Unregistered service: {service_id}")
                
                return True
                
        except Exception as e:
            self.logger.error(f"💥 Failed to unregister service {service_id}: {e}")
            return False
    
    async def get_service(self, service_id: str) -> Optional[BaseExternalService]:
        """Get a service by ID"""
        return self.services.get(service_id)
    
    async def request(self, service_id: str, method: str, endpoint: str, **kwargs) -> ServiceResponse:
        """Make a request through a specific service"""
        try:
            service = await self.get_service(service_id)
            if not service:
                raise ValueError(f"Service {service_id} not found")
            
            response = await service.request(method, endpoint, **kwargs)
            
            # Update system statistics
            self.system_stats["total_requests"] += 1
            if response.error:
                self.system_stats["failed_requests"] += 1
            else:
                self.system_stats["successful_requests"] += 1
            
            return response
            
        except Exception as e:
            self.logger.error(f"💥 Request failed for service {service_id}: {e}")
            
            # Create error response
            return ServiceResponse(
                request_id=f"req_{uuid.uuid4().hex[:12]}",
                service_id=service_id,
                status_code=0,
                error=str(e)
            )
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # Update system statistics
            self._update_system_stats()
            
            # Collect service statuses
            services_status = {}
            for service_id, service in self.services.items():
                services_status[service_id] = service.get_status()
            
            # Service summary by type and status
            services_by_type = defaultdict(int)
            services_by_status = defaultdict(int)
            
            for service in self.services.values():
                services_by_type[service.config.service_type.value] += 1
                services_by_status[service.status.value] += 1
            
            return {
                "system_stats": self.system_stats,
                "services": services_status,
                "summary": {
                    "total_services": len(self.services),
                    "by_type": dict(services_by_type),
                    "by_status": dict(services_by_status)
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"💥 Error getting system status: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _update_system_stats(self):
        """Update system statistics"""
        try:
            # Update uptime
            self.system_stats["uptime_seconds"] = (
                datetime.now() - self.system_stats["started_at"]
            ).total_seconds()
            
            # Count active services
            active_count = sum(
                1 for service in self.services.values()
                if service.status in [ServiceStatus.CONNECTED, ServiceStatus.AUTHENTICATED]
            )
            self.system_stats["active_services"] = active_count
            
        except Exception as e:
            self.logger.error(f"💥 Error updating system stats: {e}")
    
    async def _start_background_monitoring(self):
        """Start background monitoring tasks"""
        try:
            # System monitoring task
            task = asyncio.create_task(self._system_monitoring_loop())
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
            
            self.logger.info("🔄 Started background monitoring")
            
        except Exception as e:
            self.logger.error(f"💥 Failed to start background monitoring: {e}")
            raise
    
    async def _system_monitoring_loop(self):
        """Background system monitoring loop"""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Monitor every minute
                
                # Check service health
                for service_id, service in self.services.items():
                    if service.status in [ServiceStatus.CONNECTED, ServiceStatus.AUTHENTICATED]:
                        is_healthy = await service.health_check()
                        if not is_healthy:
                            self.logger.warning(f"💛 Service {service_id} health check failed")
                
                # Update statistics
                self._update_system_stats()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"💥 System monitoring error: {e}")
    
    async def shutdown(self):
        """Shutdown the services manager"""
        try:
            self.logger.info("🛑 Shutting down External Services Manager...")
            
            # Signal shutdown
            self.shutdown_event.set()
            
            # Disconnect all services
            for service in self.services.values():
                await service.disconnect()
            
            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Clear registry
            self.services.clear()
            self.service_configs.clear()
            
            self.logger.info("✅ External Services Manager shutdown completed")
            
        except Exception as e:
            self.logger.error(f"💥 Error during shutdown: {e}")

# 🌟 Spiritual Blessing for External Services
SPIRITUAL_SERVICES_BLESSING = """
بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ

اللَّهُمَّ بَارِكْ لَنَا فِي هَذِهِ الْخِدْمَاتِ الْخَارِجِيَّةِ الْمُبَارَكَةِ
وَاجْعَلْهَا مُتَّصِلَةً آمِنَةً مُوَثَّقَةً

Ya Allah, berkahilah sistem layanan eksternal ini dengan:
- 🌐 Koneksi yang stabil dan terpercaya
- 🔐 Keamanan dan autentikasi yang kuat
- ⚡ Performa yang optimal dan responsif
- 🛡️ Perlindungan dari gangguan dan serangan
- 🔄 Pemulihan otomatis dari kegagalan

آمِينَ يَا رَبَّ الْعَالَمِينَ
"""

if __name__ == "__main__":
    print("🌐 Spiritual External Services Architecture - Ladang Berkah Digital")
    print("=" * 80)
    print(SPIRITUAL_SERVICES_BLESSING)
    
    async def example_usage():
        """Example usage of the external services system"""
        
        # Create services manager
        manager = SpiritualExternalServicesManager()
        await manager.initialize()
        
        try:
            # Register HTTP API service
            api_config = ServiceConfiguration(
                service_id="jsonplaceholder_api",
                service_name="JSONPlaceholder API",
                service_type=ServiceType.HTTP_API,
                base_url="https://jsonplaceholder.typicode.com",
                credentials=ServiceCredentials(auth_type=AuthenticationType.NONE),
                timeout=30.0,
                max_retries=3,
                rate_limit_per_minute=30
            )
            
            await manager.register_service(api_config)
            
            # Make some requests
            response1 = await manager.request("jsonplaceholder_api", "GET", "/posts/1")
            print(f"📊 Response 1: {response1.status_code} - {response1.data}")
            
            response2 = await manager.request("jsonplaceholder_api", "GET", "/users")
            print(f"📊 Response 2: {response2.status_code} - {len(response2.data) if response2.data else 0} users")
            
            # Get system status
            status = await manager.get_system_status()
            print(f"\n📈 System Status:")
            print(f"   Total Services: {status['summary']['total_services']}")
            print(f"   Total Requests: {status['system_stats']['total_requests']}")
            print(f"   Successful Requests: {status['system_stats']['successful_requests']}")
            
        finally:
            await manager.shutdown()
    
    # Run example
    asyncio.run(example_usage())