"""
KAS Core - Network Access Controller
Network access control system for managing HTTP/HTTPS requests
"""

import os
import re
import fnmatch
import ipaddress
import logging
from urllib.parse import urlparse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class NetworkMode(Enum):
    ALLOW_ALL = "allow_all"
    DENY_ALL = "deny_all"
    WHITELIST = "whitelist"
    PROXY = "proxy"


class PolicyPreset(Enum):
    STRICT = "strict"
    MODERATE = "moderate"
    PERMISSIVE = "permissive"
    CUSTOM = "custom"


STRICT_ALLOWED_HOSTS = [
    "api.openai.com",
    "*.openai.com",
    "api.anthropic.com",
    "*.anthropic.com",
    "api.deepseek.com",
    "*.deepseek.com",
    "api.moonshot.cn",
    "*.moonshot.cn",
    "api.together.xyz",
    "*.together.xyz",
]

MODERATE_ALLOWED_HOSTS = [
    *STRICT_ALLOWED_HOSTS,
    "github.com",
    "*.github.com",
    "api.github.com",
    "pypi.org",
    "*.pypi.org",
    "npmjs.com",
    "*.npmjs.com",
    "stackoverflow.com",
    "*.stackoverflow.com",
    "google.com",
    "*.google.com",
]

PERMISSIVE_BLOCKED_HOSTS = [
    "malware.*",
    "phishing.*",
    "*.malware.*",
    "*.phishing.*",
    "known-bad-site.com",
    "*.known-bad-site.com",
]

DANGEROUS_PORTS = [
    23,
    25,
    445,
    3389,
]


@dataclass
class NetworkPolicy:
    mode: str = "allow_all"
    allowed_hosts: List[str] = field(default_factory=list)
    blocked_hosts: List[str] = field(default_factory=list)
    proxy_url: Optional[str] = None
    allowed_ports: List[int] = field(default_factory=lambda: [80, 443, 8080, 8443])
    allow_localhost: bool = True

    @classmethod
    def from_preset(cls, preset: PolicyPreset) -> "NetworkPolicy":
        if preset == PolicyPreset.STRICT:
            return cls(
                mode="whitelist",
                allowed_hosts=STRICT_ALLOWED_HOSTS.copy(),
                blocked_hosts=[],
                allowed_ports=[443],
                allow_localhost=True,
            )
        elif preset == PolicyPreset.MODERATE:
            return cls(
                mode="whitelist",
                allowed_hosts=MODERATE_ALLOWED_HOSTS.copy(),
                blocked_hosts=PERMISSIVE_BLOCKED_HOSTS.copy(),
                allowed_ports=[80, 443, 8080, 8443],
                allow_localhost=True,
            )
        elif preset == PolicyPreset.PERMISSIVE:
            return cls(
                mode="allow_all",
                allowed_hosts=[],
                blocked_hosts=PERMISSIVE_BLOCKED_HOSTS.copy(),
                allowed_ports=[80, 443, 8080, 8443],
                allow_localhost=True,
            )
        else:
            return cls()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "allowed_hosts": self.allowed_hosts,
            "blocked_hosts": self.blocked_hosts,
            "proxy_url": self.proxy_url,
            "allowed_ports": self.allowed_ports,
            "allow_localhost": self.allow_localhost,
        }


class NetworkAccessController:
    def __init__(self, policy: Optional[NetworkPolicy] = None):
        self.policy = policy or NetworkPolicy()
        self._compiled_allowed: List[re.Pattern] = []
        self._compiled_blocked: List[re.Pattern] = []
        self._compile_patterns()

    def _compile_patterns(self):
        self._compiled_allowed = []
        self._compiled_blocked = []

        for pattern in self.policy.allowed_hosts:
            self._compiled_allowed.append(self._host_to_regex(pattern))

        for pattern in self.policy.blocked_hosts:
            self._compiled_blocked.append(self._host_to_regex(pattern))

    def _host_to_regex(self, pattern: str) -> re.Pattern:
        if pattern.startswith("*."):
            base = pattern[2:]
            regex = rf"^({re.escape(base)}|.*\.{re.escape(base)})$"
        elif "*" in pattern:
            regex = "^" + fnmatch.translate(pattern).rstrip("\\Z") + "$"
        else:
            regex = f"^{re.escape(pattern)}$"
        return re.compile(regex, re.IGNORECASE)

    def _is_ip_in_cidr(self, ip_str: str, cidr: str) -> bool:
        try:
            ip = ipaddress.ip_address(ip_str)
            network = ipaddress.ip_network(cidr, strict=False)
            return ip in network
        except ValueError:
            return False

    def _match_host_pattern(self, host: str, patterns: List[re.Pattern]) -> bool:
        for pattern in patterns:
            if pattern.match(host):
                return True
        return False

    def _is_localhost(self, host: str) -> bool:
        localhost_ips = ["127.0.0.1", "::1", "0.0.0.0", "localhost"]
        if host.lower() in localhost_ips:
            return True
        try:
            ip = ipaddress.ip_address(host)
            return ip.is_loopback
        except ValueError:
            pass
        return False

    def is_allowed(self, host: str, port: int) -> bool:
        host = host.lower().strip()

        if self._is_localhost(host):
            if self.policy.allow_localhost:
                return port in self.policy.allowed_ports
            return False

        if port not in self.policy.allowed_ports:
            logger.debug(f"Port {port} not in allowed ports")
            return False

        if self._match_host_pattern(host, self._compiled_blocked):
            logger.debug(f"Host {host} is blocked")
            return False

        for pattern in self.policy.blocked_hosts:
            if "/" in pattern:
                if self._is_ip_in_cidr(host, pattern):
                    logger.debug(f"Host {host} is in blocked CIDR {pattern}")
                    return False

        if self.policy.mode == "allow_all":
            return True
        elif self.policy.mode == "deny_all":
            return False
        elif self.policy.mode == "whitelist":
            return self._match_host_pattern(host, self._compiled_allowed)
        elif self.policy.mode == "proxy":
            return self._match_host_pattern(host, self._compiled_allowed)

        return False

    def add_allowed_host(self, host: str):
        if host not in self.policy.allowed_hosts:
            self.policy.allowed_hosts.append(host)
            self._compiled_allowed.append(self._host_to_regex(host))
            logger.info(f"Added {host} to allowed hosts")

    def add_blocked_host(self, host: str):
        if host not in self.policy.blocked_hosts:
            self.policy.blocked_hosts.append(host)
            self._compiled_blocked.append(self._host_to_regex(host))
            logger.info(f"Added {host} to blocked hosts")

    def remove_allowed_host(self, host: str):
        if host in self.policy.allowed_hosts:
            self.policy.allowed_hosts.remove(host)
            self._compile_patterns()
            logger.info(f"Removed {host} from allowed hosts")

    def remove_blocked_host(self, host: str):
        if host in self.policy.blocked_hosts:
            self.policy.blocked_hosts.remove(host)
            self._compile_patterns()
            logger.info(f"Removed {host} from blocked hosts")

    def set_proxy(self, url: str):
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid proxy URL: {url}")
        self.policy.proxy_url = url
        if self.policy.mode not in ["proxy", "whitelist"]:
            self.policy.mode = "proxy"
        logger.info(f"Proxy set to {url}")

    def get_proxy_config(self) -> Dict[str, str]:
        if not self.policy.proxy_url:
            return {}

        return {
            "http": self.policy.proxy_url,
            "https": self.policy.proxy_url,
        }

    def validate_url(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False

            if parsed.scheme not in ["http", "https"]:
                logger.debug(f"Unsupported scheme: {parsed.scheme}")
                return False

            host = parsed.hostname
            if not host:
                return False

            port = parsed.port
            if port is None:
                port = 443 if parsed.scheme == "https" else 80

            return self.is_allowed(host, port)
        except Exception as e:
            logger.debug(f"URL validation error: {e}")
            return False

    def set_policy(self, policy: NetworkPolicy):
        self.policy = policy
        self._compile_patterns()
        logger.info(f"Policy updated to mode: {policy.mode}")

    def set_preset(self, preset: PolicyPreset):
        self.policy = NetworkPolicy.from_preset(preset)
        self._compile_patterns()
        logger.info(f"Policy set to preset: {preset.value}")

    def get_policy_info(self) -> Dict[str, Any]:
        return {
            "mode": self.policy.mode,
            "allowed_hosts_count": len(self.policy.allowed_hosts),
            "blocked_hosts_count": len(self.policy.blocked_hosts),
            "proxy_enabled": self.policy.proxy_url is not None,
            "allowed_ports": self.policy.allowed_ports,
            "allow_localhost": self.policy.allow_localhost,
        }


class NetworkInterceptor:
    def __init__(self, controller: Optional[NetworkAccessController] = None):
        self.controller = controller or NetworkAccessController()
        self._original_request = None
        self._original_request_async = None
        self._original_urlopen = None
        self._active = False

    def _wrap_requests(self):
        try:
            import requests

            original_request = requests.Session.request

            def wrapped_request(session_self, method, url, *args, **kwargs):
                if not self.controller.validate_url(url):
                    raise ConnectionError(
                        f"Network access denied: {url} is not allowed by policy"
                    )
                return original_request(session_self, method, url, *args, **kwargs)

            self._original_request = requests.Session.request
            requests.Session.request = wrapped_request
            logger.debug("Wrapped requests.Session.request")
        except ImportError:
            pass

    def _wrap_httpx(self):
        try:
            import httpx

            original_request = httpx.Client.request

            def wrapped_request(client_self, method, url, *args, **kwargs):
                if not self.controller.validate_url(str(url)):
                    raise ConnectionError(
                        f"Network access denied: {url} is not allowed by policy"
                    )
                return original_request(client_self, method, url, *args, **kwargs)

            self._original_httpx_request = httpx.Client.request
            httpx.Client.request = wrapped_request
            logger.debug("Wrapped httpx.Client.request")
        except ImportError:
            pass

    def _wrap_urllib(self):
        try:
            import urllib.request

            original_urlopen = urllib.request.urlopen

            def wrapped_urlopen(url, *args, **kwargs):
                url_str = str(url)
                if not self.controller.validate_url(url_str):
                    raise ConnectionError(
                        f"Network access denied: {url_str} is not allowed by policy"
                    )
                return original_urlopen(url, *args, **kwargs)

            self._original_urlopen = urllib.request.urlopen
            urllib.request.urlopen = wrapped_urlopen
            logger.debug("Wrapped urllib.request.urlopen")
        except ImportError:
            pass

    def _restore_requests(self):
        try:
            import requests

            if self._original_request:
                requests.Session.request = self._original_request
                self._original_request = None
        except ImportError:
            pass

    def _restore_httpx(self):
        try:
            import httpx

            if hasattr(self, "_original_httpx_request") and self._original_httpx_request:
                httpx.Client.request = self._original_httpx_request
                self._original_httpx_request = None
        except ImportError:
            pass

    def _restore_urllib(self):
        try:
            import urllib.request

            if self._original_urlopen:
                urllib.request.urlopen = self._original_urlopen
                self._original_urlopen = None
        except ImportError:
            pass

    def set_env_proxy(self):
        if self.controller.policy.proxy_url:
            os.environ["HTTP_PROXY"] = self.controller.policy.proxy_url
            os.environ["HTTPS_PROXY"] = self.controller.policy.proxy_url
            os.environ["http_proxy"] = self.controller.policy.proxy_url
            os.environ["https_proxy"] = self.controller.policy.proxy_url
            logger.debug(f"Set proxy environment variables to {self.controller.policy.proxy_url}")

    def clear_env_proxy(self):
        for var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
            os.environ.pop(var, None)
        logger.debug("Cleared proxy environment variables")

    def activate(self):
        if self._active:
            return
        self._wrap_requests()
        self._wrap_httpx()
        self._wrap_urllib()
        self.set_env_proxy()
        self._active = True
        logger.info("Network interceptor activated")

    def deactivate(self):
        if not self._active:
            return
        self._restore_requests()
        self._restore_httpx()
        self._restore_urllib()
        self.clear_env_proxy()
        self._active = False
        logger.info("Network interceptor deactivated")

    @contextmanager
    def intercept(self):
        self.activate()
        try:
            yield self
        finally:
            self.deactivate()

    def is_active(self) -> bool:
        return self._active


def create_strict_controller() -> NetworkAccessController:
    return NetworkAccessController(NetworkPolicy.from_preset(PolicyPreset.STRICT))


def create_moderate_controller() -> NetworkAccessController:
    return NetworkAccessController(NetworkPolicy.from_preset(PolicyPreset.MODERATE))


def create_permissive_controller() -> NetworkAccessController:
    return NetworkAccessController(NetworkPolicy.from_preset(PolicyPreset.PERMISSIVE))


_default_controller: Optional[NetworkAccessController] = None


def get_default_controller() -> NetworkAccessController:
    global _default_controller
    if _default_controller is None:
        _default_controller = create_moderate_controller()
    return _default_controller


def set_default_controller(controller: NetworkAccessController):
    global _default_controller
    _default_controller = controller
