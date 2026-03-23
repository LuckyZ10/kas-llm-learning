"""
Sensitive Information Filter
Detects and masks sensitive information in text
"""
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class MatchInfo:
    """Information about a detected sensitive item"""
    type: str
    start: int
    end: int
    value: str
    masked: str


@dataclass
class FilterConfig:
    """Configuration for sensitive info filter"""
    enabled_types: List[str] = field(default_factory=lambda: [
        "api_key", "password", "database", "jwt", "credit_card", "email", "phone"
    ])
    mask_char: str = "*"
    preserve_length: bool = True
    whitelist_patterns: List[str] = field(default_factory=list)
    mask_email: bool = False
    mask_phone: bool = False


class SensitiveInfoFilter:
    """
    Sensitive Information Filter
    
    Detects and masks various types of sensitive information:
    - API Keys (sk-xxx, xoxb-xxx, ghpat-xxx, etc.)
    - Passwords (password=xxx, pwd: xxx)
    - Database connection strings
    - JWT Tokens
    - Credit card numbers
    - Email addresses (optional)
    - Phone numbers (optional)
    """
    
    DEFAULT_RULES: Dict[str, str] = {
        "api_key": r"(?i)(?:sk-[a-zA-Z0-9]{8,}|xox[baprs]-[a-zA-Z0-9-]{10,}|ghp_[a-zA-Z0-9]{36}|ghpat_[a-zA-Z0-9]{22}|api[_-]?key\s*[=:]\s*['\"]?[a-zA-Z0-9_-]{20,}['\"]?|Bearer\s+[a-zA-Z0-9._-]{20,})",
        "password": r"(?i)(?:password|passwd|pwd)\s*[=:]\s*['\"]?[^'\"\s\n]{4,}['\"]?",
        "database": r"(?i)(?:mysql|postgres|postgresql|mongodb|redis)://[^\s'\"]+",
        "jwt": r"eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*",
        "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{13,16}\b",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b(?:\+?1[-.\s]?)?(?:\(?[0-9]{3}\)?[-.\s]?)?[0-9]{3}[-.\s]?[0-9]{4}\b|\b\+?[1-9]\d{6,14}\b",
        "aws_key": r"(?:A3T[A-Z0-9]|AKIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[0-9A-Z]{16}",
        "private_key": r"-----BEGIN (?:RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----",
        "secret": r"(?i)(?:secret|token|auth)[_-]?(?:key)?\s*[=:]\s*['\"]?[a-zA-Z0-9_-]{16,}['\"]?",
    }
    
    def __init__(self, config: Optional[FilterConfig] = None):
        self.config = config or FilterConfig()
        self._rules: Dict[str, str] = {}
        self._compiled_rules: Dict[str, re.Pattern] = {}
        self._whitelist_compiled: List[re.Pattern] = []
        
        self._init_rules()
        self._compile_whitelist()
    
    def _init_rules(self):
        """Initialize detection rules"""
        for rule_type in self.config.enabled_types:
            if rule_type in self.DEFAULT_RULES:
                self._rules[rule_type] = self.DEFAULT_RULES[rule_type]
                try:
                    self._compiled_rules[rule_type] = re.compile(self.DEFAULT_RULES[rule_type])
                except re.error as e:
                    logger.warning(f"Failed to compile rule {rule_type}: {e}")
    
    def _compile_whitelist(self):
        """Compile whitelist patterns"""
        self._whitelist_compiled = []
        for pattern in self.config.whitelist_patterns:
            try:
                self._whitelist_compiled.append(re.compile(pattern))
            except re.error as e:
                logger.warning(f"Failed to compile whitelist pattern {pattern}: {e}")
    
    def _is_whitelisted(self, text: str) -> bool:
        """Check if text matches any whitelist pattern"""
        for pattern in self._whitelist_compiled:
            if pattern.search(text):
                return True
        return False
    
    def _mask_value(self, value: str) -> str:
        """Mask a sensitive value"""
        if self.config.preserve_length:
            return self.config.mask_char * len(value)
        return self.config.mask_char * 8
    
    def add_rule(self, pattern: str, name: str) -> bool:
        """
        Add a custom detection rule
        
        Args:
            pattern: Regex pattern
            name: Rule name/type
            
        Returns:
            True if rule was added successfully
        """
        try:
            compiled = re.compile(pattern)
            self._rules[name] = pattern
            self._compiled_rules[name] = compiled
            if name not in self.config.enabled_types:
                self.config.enabled_types.append(name)
            logger.info(f"Added custom rule: {name}")
            return True
        except re.error as e:
            logger.error(f"Failed to add rule {name}: {e}")
            return False
    
    def remove_rule(self, name: str) -> bool:
        """Remove a detection rule"""
        if name in self._rules:
            del self._rules[name]
            del self._compiled_rules[name]
            if name in self.config.enabled_types:
                self.config.enabled_types.remove(name)
            return True
        return False
    
    def add_whitelist_pattern(self, pattern: str) -> bool:
        """Add a whitelist pattern"""
        try:
            compiled = re.compile(pattern)
            self.config.whitelist_patterns.append(pattern)
            self._whitelist_compiled.append(compiled)
            return True
        except re.error as e:
            logger.error(f"Failed to add whitelist pattern: {e}")
            return False
    
    def detect(self, text: str) -> List[MatchInfo]:
        """
        Detect sensitive information in text
        
        Args:
            text: Text to analyze
            
        Returns:
            List of MatchInfo objects for detected items
        """
        if self._is_whitelisted(text):
            return []
        
        matches: List[MatchInfo] = []
        
        for rule_type, compiled in self._compiled_rules.items():
            if rule_type == "email" and not self.config.mask_email:
                continue
            if rule_type == "phone" and not self.config.mask_phone:
                continue
            
            for match in compiled.finditer(text):
                value = match.group()
                start = match.start()
                end = match.end()
                
                if self._is_whitelisted(value):
                    continue
                
                masked = self._mask_value(value)
                
                matches.append(MatchInfo(
                    type=rule_type,
                    start=start,
                    end=end,
                    value=value,
                    masked=masked
                ))
        
        matches.sort(key=lambda x: x.start)
        
        filtered_matches = []
        for match in matches:
            is_overlapping = False
            for existing in filtered_matches:
                if (match.start >= existing.start and match.start < existing.end) or \
                   (match.end > existing.start and match.end <= existing.end):
                    is_overlapping = True
                    break
            if not is_overlapping:
                filtered_matches.append(match)
        
        return filtered_matches
    
    def filter(self, text: str) -> str:
        """
        Filter text by masking sensitive information
        
        Args:
            text: Text to filter
            
        Returns:
            Filtered text with sensitive info masked
        """
        matches = self.detect(text)
        
        if not matches:
            return text
        
        result = []
        last_end = 0
        
        for match in matches:
            result.append(text[last_end:match.start])
            result.append(match.masked)
            last_end = match.end
        
        result.append(text[last_end:])
        
        return "".join(result)
    
    def is_sensitive(self, text: str) -> bool:
        """
        Check if text contains sensitive information
        
        Args:
            text: Text to check
            
        Returns:
            True if sensitive info detected
        """
        return len(self.detect(text)) > 0
    
    def get_sensitive_types(self, text: str) -> List[str]:
        """
        Get list of sensitive information types found in text
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected types
        """
        matches = self.detect(text)
        return list(set(m.type for m in matches))
    
    def filter_dict(self, data: Dict, recursive: bool = True) -> Dict:
        """
        Filter sensitive info in dictionary values
        
        Args:
            data: Dictionary to filter
            recursive: Whether to process nested dicts/lists
            
        Returns:
            Filtered dictionary
        """
        result = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self.filter(value)
            elif isinstance(value, dict) and recursive:
                result[key] = self.filter_dict(value, recursive)
            elif isinstance(value, list) and recursive:
                result[key] = self.filter_list(value)
            else:
                result[key] = value
        
        return result
    
    def filter_list(self, data: List, recursive: bool = True) -> List:
        """
        Filter sensitive info in list values
        
        Args:
            data: List to filter
            recursive: Whether to process nested dicts/lists
            
        Returns:
            Filtered list
        """
        result = []
        
        for item in data:
            if isinstance(item, str):
                result.append(self.filter(item))
            elif isinstance(item, dict) and recursive:
                result.append(self.filter_dict(item, recursive))
            elif isinstance(item, list) and recursive:
                result.append(self.filter_list(item, recursive))
            else:
                result.append(item)
        
        return result


_default_filter: Optional[SensitiveInfoFilter] = None


def get_default_filter() -> SensitiveInfoFilter:
    """Get the global default filter instance"""
    global _default_filter
    if _default_filter is None:
        _default_filter = SensitiveInfoFilter()
    return _default_filter


def filter_text(text: str) -> str:
    """Quick filter function using default filter"""
    return get_default_filter().filter(text)


def is_sensitive(text: str) -> bool:
    """Quick check function using default filter"""
    return get_default_filter().is_sensitive(text)


def detect_sensitive(text: str) -> List[MatchInfo]:
    """Quick detect function using default filter"""
    return get_default_filter().detect(text)
