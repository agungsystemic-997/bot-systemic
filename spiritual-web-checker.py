#!/usr/bin/env python3
"""
🌐 SPIRITUAL WEB CHECKER SYSTEM
Ladang Berkah Digital - ZeroLight Orbit System
Comprehensive Web, Domain, and Subdomain Status Checker

بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ
"""

import asyncio
import aiohttp
import socket
import ssl
import dns.resolver
import dns.exception
import whois
import requests
import json
import time
import uuid
import logging
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse, urljoin
import concurrent.futures
from collections import defaultdict
import ipaddress
import certifi

class DomainStatus(Enum):
    """Domain status types"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    SUSPENDED = "suspended"
    PENDING = "pending"
    UNKNOWN = "unknown"
    ERROR = "error"

class WebStatus(Enum):
    """Web service status"""
    ONLINE = "online"
    OFFLINE = "offline"
    SLOW = "slow"
    ERROR = "error"
    TIMEOUT = "timeout"
    SSL_ERROR = "ssl_error"
    DNS_ERROR = "dns_error"
    REDIRECT = "redirect"

class ReadinessLevel(Enum):
    """Service readiness levels"""
    READY = "ready"
    PARTIALLY_READY = "partially_ready"
    NOT_READY = "not_ready"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"

@dataclass
class SSLInfo:
    """SSL certificate information"""
    is_valid: bool = False
    issuer: str = ""
    subject: str = ""
    expires_at: Optional[datetime] = None
    days_until_expiry: int = 0
    is_expired: bool = False
    is_self_signed: bool = False
    error: Optional[str] = None

@dataclass
class DNSRecord:
    """DNS record information"""
    record_type: str
    value: str
    ttl: int = 0

@dataclass
class DomainInfo:
    """Domain information"""
    domain: str
    status: DomainStatus = DomainStatus.UNKNOWN
    registrar: str = ""
    creation_date: Optional[datetime] = None
    expiration_date: Optional[datetime] = None
    days_until_expiry: int = 0
    nameservers: List[str] = field(default_factory=list)
    dns_records: Dict[str, List[DNSRecord]] = field(default_factory=dict)
    whois_info: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

@dataclass
class WebServiceInfo:
    """Web service information"""
    url: str
    status: WebStatus = WebStatus.UNKNOWN
    status_code: int = 0
    response_time_ms: float = 0.0
    ssl_info: Optional[SSLInfo] = None
    headers: Dict[str, str] = field(default_factory=dict)
    server: str = ""
    content_type: str = ""
    content_length: int = 0
    redirect_url: Optional[str] = None
    error: Optional[str] = None
    last_checked: datetime = field(default_factory=datetime.now)

@dataclass
class SubdomainInfo:
    """Subdomain information"""
    subdomain: str
    ip_addresses: List[str] = field(default_factory=list)
    web_service: Optional[WebServiceInfo] = None
    dns_records: Dict[str, List[DNSRecord]] = field(default_factory=dict)
    is_active: bool = False
    error: Optional[str] = None

@dataclass
class WebInfrastructureStatus:
    """Complete web infrastructure status"""
    domain_info: DomainInfo
    subdomains: List[SubdomainInfo] = field(default_factory=list)
    main_website: Optional[WebServiceInfo] = None
    readiness_level: ReadinessLevel = ReadinessLevel.UNKNOWN
    overall_health_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    scan_timestamp: datetime = field(default_factory=datetime.now)
    scan_duration_seconds: float = 0.0

class SpiritualWebChecker:
    """
    🌟 Spiritual Web Checker System
    
    Comprehensive checker for web infrastructure, domains, and subdomains
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the web checker"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.timeout = self.config.get("timeout", 30.0)
        self.max_concurrent = self.config.get("max_concurrent", 10)
        self.user_agent = self.config.get("user_agent", 
            "SpiritualWebChecker/1.0 (Ladang Berkah Digital)")
        
        # Common subdomains to check
        self.common_subdomains = [
            "www", "api", "app", "admin", "blog", "shop", "mail", "ftp",
            "cdn", "static", "assets", "img", "images", "media", "files",
            "dev", "test", "staging", "beta", "demo", "docs", "help",
            "support", "forum", "community", "news", "events", "store",
            "portal", "dashboard", "panel", "control", "manage", "secure",
            "ssl", "vpn", "remote", "cloud", "backup", "archive", "old",
            "new", "mobile", "m", "wap", "touch", "amp", "accelerated"
        ]
        
        # DNS resolvers
        self.dns_resolvers = [
            "8.8.8.8",      # Google
            "1.1.1.1",      # Cloudflare
            "208.67.222.222", # OpenDNS
            "9.9.9.9"       # Quad9
        ]
        
        # Session for HTTP requests
        self.session = None
        
        # Results cache
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    async def initialize(self):
        """Initialize the web checker"""
        try:
            self.logger.info("🚀 Initializing Spiritual Web Checker...")
            
            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            connector = aiohttp.TCPConnector(
                ssl=ssl.create_default_context(cafile=certifi.where()),
                limit=self.max_concurrent,
                limit_per_host=5
            )
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={"User-Agent": self.user_agent}
            )
            
            self.logger.info("✅ Web checker initialized successfully")
            
        except Exception as e:
            self.logger.error(f"💥 Failed to initialize web checker: {e}")
            raise
    
    async def check_web_infrastructure(self, domain: str, 
                                     check_subdomains: bool = True,
                                     subdomain_list: Optional[List[str]] = None) -> WebInfrastructureStatus:
        """
        🔍 Check complete web infrastructure
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"🌐 Checking web infrastructure for: {domain}")
            
            # Check domain info
            domain_info = await self.check_domain_info(domain)
            
            # Check main website
            main_website = await self.check_web_service(f"https://{domain}")
            if main_website.status in [WebStatus.ERROR, WebStatus.TIMEOUT]:
                # Try HTTP if HTTPS fails
                main_website = await self.check_web_service(f"http://{domain}")
            
            # Check subdomains
            subdomains = []
            if check_subdomains:
                subdomain_list = subdomain_list or self.common_subdomains
                subdomains = await self.scan_subdomains(domain, subdomain_list)
            
            # Calculate readiness and health score
            readiness_level, health_score, recommendations = self._analyze_infrastructure(
                domain_info, main_website, subdomains
            )
            
            # Create infrastructure status
            infrastructure_status = WebInfrastructureStatus(
                domain_info=domain_info,
                subdomains=subdomains,
                main_website=main_website,
                readiness_level=readiness_level,
                overall_health_score=health_score,
                recommendations=recommendations,
                scan_duration_seconds=time.time() - start_time
            )
            
            self.logger.info(f"✅ Infrastructure check completed for {domain}")
            return infrastructure_status
            
        except Exception as e:
            self.logger.error(f"💥 Infrastructure check failed for {domain}: {e}")
            
            # Return error status
            return WebInfrastructureStatus(
                domain_info=DomainInfo(domain=domain, status=DomainStatus.ERROR, error=str(e)),
                readiness_level=ReadinessLevel.UNKNOWN,
                scan_duration_seconds=time.time() - start_time
            )
    
    async def check_domain_info(self, domain: str) -> DomainInfo:
        """
        🏷️ Check domain information and DNS records
        """
        try:
            self.logger.info(f"🔍 Checking domain info for: {domain}")
            
            domain_info = DomainInfo(domain=domain)
            
            # Get WHOIS information
            try:
                whois_info = whois.whois(domain)
                if whois_info:
                    domain_info.registrar = str(whois_info.registrar or "")
                    domain_info.creation_date = whois_info.creation_date
                    domain_info.expiration_date = whois_info.expiration_date
                    domain_info.nameservers = list(whois_info.name_servers or [])
                    domain_info.whois_info = dict(whois_info)
                    
                    # Calculate days until expiry
                    if domain_info.expiration_date:
                        if isinstance(domain_info.expiration_date, list):
                            domain_info.expiration_date = domain_info.expiration_date[0]
                        
                        days_until = (domain_info.expiration_date - datetime.now()).days
                        domain_info.days_until_expiry = max(0, days_until)
                        
                        if days_until > 30:
                            domain_info.status = DomainStatus.ACTIVE
                        elif days_until > 0:
                            domain_info.status = DomainStatus.PENDING
                        else:
                            domain_info.status = DomainStatus.EXPIRED
                    else:
                        domain_info.status = DomainStatus.ACTIVE
                        
            except Exception as e:
                self.logger.warning(f"⚠️ WHOIS lookup failed for {domain}: {e}")
                domain_info.error = f"WHOIS error: {str(e)}"
            
            # Get DNS records
            dns_records = await self._get_dns_records(domain)
            domain_info.dns_records = dns_records
            
            # If WHOIS failed but DNS works, assume active
            if domain_info.status == DomainStatus.UNKNOWN and dns_records:
                domain_info.status = DomainStatus.ACTIVE
            
            return domain_info
            
        except Exception as e:
            self.logger.error(f"💥 Domain info check failed for {domain}: {e}")
            return DomainInfo(domain=domain, status=DomainStatus.ERROR, error=str(e))
    
    async def _get_dns_records(self, domain: str) -> Dict[str, List[DNSRecord]]:
        """Get DNS records for domain"""
        dns_records = defaultdict(list)
        
        record_types = ["A", "AAAA", "CNAME", "MX", "TXT", "NS", "SOA"]
        
        for record_type in record_types:
            try:
                resolver = dns.resolver.Resolver()
                resolver.nameservers = self.dns_resolvers[:2]  # Use first 2 resolvers
                
                answers = resolver.resolve(domain, record_type)
                
                for answer in answers:
                    dns_record = DNSRecord(
                        record_type=record_type,
                        value=str(answer),
                        ttl=answers.ttl
                    )
                    dns_records[record_type].append(dns_record)
                    
            except dns.exception.DNSException:
                # Record type not found, continue
                continue
            except Exception as e:
                self.logger.warning(f"⚠️ DNS lookup failed for {domain} {record_type}: {e}")
        
        return dict(dns_records)
    
    async def check_web_service(self, url: str) -> WebServiceInfo:
        """
        🌐 Check web service status and information
        """
        try:
            self.logger.debug(f"🔍 Checking web service: {url}")
            
            start_time = time.time()
            web_info = WebServiceInfo(url=url)
            
            try:
                async with self.session.get(url, allow_redirects=False) as response:
                    web_info.status_code = response.status
                    web_info.response_time_ms = (time.time() - start_time) * 1000
                    web_info.headers = dict(response.headers)
                    web_info.server = response.headers.get("Server", "")
                    web_info.content_type = response.headers.get("Content-Type", "")
                    web_info.content_length = int(response.headers.get("Content-Length", 0))
                    
                    # Check for redirects
                    if 300 <= response.status < 400:
                        web_info.status = WebStatus.REDIRECT
                        web_info.redirect_url = response.headers.get("Location")
                    elif 200 <= response.status < 300:
                        web_info.status = WebStatus.ONLINE
                    else:
                        web_info.status = WebStatus.ERROR
                    
                    # Check response time
                    if web_info.response_time_ms > 5000:  # 5 seconds
                        web_info.status = WebStatus.SLOW
                        
            except aiohttp.ClientTimeout:
                web_info.status = WebStatus.TIMEOUT
                web_info.error = "Request timeout"
                
            except aiohttp.ClientSSLError as e:
                web_info.status = WebStatus.SSL_ERROR
                web_info.error = f"SSL error: {str(e)}"
                
            except aiohttp.ClientConnectorError as e:
                web_info.status = WebStatus.OFFLINE
                web_info.error = f"Connection error: {str(e)}"
                
            except Exception as e:
                web_info.status = WebStatus.ERROR
                web_info.error = str(e)
            
            # Check SSL certificate if HTTPS
            if url.startswith("https://"):
                ssl_info = await self._check_ssl_certificate(url)
                web_info.ssl_info = ssl_info
            
            return web_info
            
        except Exception as e:
            self.logger.error(f"💥 Web service check failed for {url}: {e}")
            return WebServiceInfo(url=url, status=WebStatus.ERROR, error=str(e))
    
    async def _check_ssl_certificate(self, url: str) -> SSLInfo:
        """Check SSL certificate information"""
        try:
            parsed_url = urlparse(url)
            hostname = parsed_url.hostname
            port = parsed_url.port or 443
            
            # Get certificate
            context = ssl.create_default_context()
            
            with socket.create_connection((hostname, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    
                    ssl_info = SSLInfo()
                    ssl_info.is_valid = True
                    ssl_info.issuer = dict(x[0] for x in cert.get('issuer', []))
                    ssl_info.subject = dict(x[0] for x in cert.get('subject', []))
                    
                    # Parse expiration date
                    not_after = cert.get('notAfter')
                    if not_after:
                        ssl_info.expires_at = datetime.strptime(not_after, '%b %d %H:%M:%S %Y %Z')
                        ssl_info.days_until_expiry = (ssl_info.expires_at - datetime.now()).days
                        ssl_info.is_expired = ssl_info.days_until_expiry <= 0
                    
                    # Check if self-signed
                    issuer_cn = ssl_info.issuer.get('commonName', '')
                    subject_cn = ssl_info.subject.get('commonName', '')
                    ssl_info.is_self_signed = issuer_cn == subject_cn
                    
                    return ssl_info
                    
        except Exception as e:
            return SSLInfo(is_valid=False, error=str(e))
    
    async def scan_subdomains(self, domain: str, subdomain_list: List[str]) -> List[SubdomainInfo]:
        """
        🔍 Scan for active subdomains
        """
        try:
            self.logger.info(f"🔍 Scanning subdomains for: {domain}")
            
            # Create tasks for concurrent subdomain checking
            tasks = []
            for subdomain in subdomain_list:
                full_subdomain = f"{subdomain}.{domain}"
                task = asyncio.create_task(self._check_subdomain(full_subdomain))
                tasks.append(task)
            
            # Execute tasks with concurrency limit
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            async def limited_task(task):
                async with semaphore:
                    return await task
            
            limited_tasks = [limited_task(task) for task in tasks]
            results = await asyncio.gather(*limited_tasks, return_exceptions=True)
            
            # Filter active subdomains
            active_subdomains = []
            for result in results:
                if isinstance(result, SubdomainInfo) and result.is_active:
                    active_subdomains.append(result)
            
            self.logger.info(f"✅ Found {len(active_subdomains)} active subdomains for {domain}")
            return active_subdomains
            
        except Exception as e:
            self.logger.error(f"💥 Subdomain scan failed for {domain}: {e}")
            return []
    
    async def _check_subdomain(self, subdomain: str) -> SubdomainInfo:
        """Check individual subdomain"""
        try:
            subdomain_info = SubdomainInfo(subdomain=subdomain)
            
            # DNS resolution
            try:
                resolver = dns.resolver.Resolver()
                resolver.nameservers = self.dns_resolvers[:2]
                
                # Try A record
                try:
                    answers = resolver.resolve(subdomain, 'A')
                    subdomain_info.ip_addresses = [str(answer) for answer in answers]
                    subdomain_info.is_active = True
                except dns.exception.DNSException:
                    pass
                
                # Try AAAA record if no A record
                if not subdomain_info.ip_addresses:
                    try:
                        answers = resolver.resolve(subdomain, 'AAAA')
                        subdomain_info.ip_addresses = [str(answer) for answer in answers]
                        subdomain_info.is_active = True
                    except dns.exception.DNSException:
                        pass
                
                # Get DNS records
                if subdomain_info.is_active:
                    dns_records = await self._get_dns_records(subdomain)
                    subdomain_info.dns_records = dns_records
                    
                    # Check web service
                    web_service = await self.check_web_service(f"https://{subdomain}")
                    if web_service.status in [WebStatus.ERROR, WebStatus.TIMEOUT]:
                        web_service = await self.check_web_service(f"http://{subdomain}")
                    
                    subdomain_info.web_service = web_service
                    
            except Exception as e:
                subdomain_info.error = str(e)
            
            return subdomain_info
            
        except Exception as e:
            return SubdomainInfo(subdomain=subdomain, error=str(e))
    
    def _analyze_infrastructure(self, domain_info: DomainInfo, 
                              main_website: Optional[WebServiceInfo],
                              subdomains: List[SubdomainInfo]) -> Tuple[ReadinessLevel, float, List[str]]:
        """Analyze infrastructure and provide recommendations"""
        try:
            health_score = 0.0
            recommendations = []
            
            # Domain health (30% of total score)
            domain_score = 0.0
            if domain_info.status == DomainStatus.ACTIVE:
                domain_score = 30.0
                if domain_info.days_until_expiry < 30:
                    recommendations.append(f"⚠️ Domain expires in {domain_info.days_until_expiry} days - renew soon")
                    domain_score -= 5.0
            elif domain_info.status == DomainStatus.EXPIRED:
                recommendations.append("🚨 Domain has expired - renew immediately")
            else:
                recommendations.append("⚠️ Domain status unclear - verify registration")
            
            health_score += domain_score
            
            # Main website health (40% of total score)
            website_score = 0.0
            if main_website:
                if main_website.status == WebStatus.ONLINE:
                    website_score = 40.0
                    if main_website.response_time_ms > 3000:
                        recommendations.append("⚠️ Website response time is slow (>3s)")
                        website_score -= 5.0
                elif main_website.status == WebStatus.SLOW:
                    website_score = 30.0
                    recommendations.append("⚠️ Website is responding slowly")
                elif main_website.status == WebStatus.REDIRECT:
                    website_score = 35.0
                    recommendations.append(f"🔄 Website redirects to: {main_website.redirect_url}")
                else:
                    recommendations.append(f"🚨 Main website is {main_website.status.value}")
                
                # SSL check
                if main_website.ssl_info:
                    if main_website.ssl_info.is_valid:
                        website_score += 5.0
                        if main_website.ssl_info.days_until_expiry < 30:
                            recommendations.append(f"⚠️ SSL certificate expires in {main_website.ssl_info.days_until_expiry} days")
                    else:
                        recommendations.append("🚨 SSL certificate is invalid")
                elif main_website.url.startswith("https://"):
                    recommendations.append("⚠️ HTTPS enabled but SSL check failed")
            else:
                recommendations.append("🚨 Main website is not accessible")
            
            health_score += website_score
            
            # Subdomains health (20% of total score)
            subdomain_score = 0.0
            if subdomains:
                active_count = len(subdomains)
                online_count = sum(1 for sub in subdomains 
                                 if sub.web_service and sub.web_service.status == WebStatus.ONLINE)
                
                subdomain_score = (online_count / active_count) * 20.0 if active_count > 0 else 0.0
                
                if active_count > 0:
                    recommendations.append(f"📊 Found {active_count} active subdomains, {online_count} online")
            
            health_score += subdomain_score
            
            # DNS health (10% of total score)
            dns_score = 0.0
            if domain_info.dns_records:
                dns_score = 10.0
                
                # Check for important records
                if 'A' not in domain_info.dns_records and 'AAAA' not in domain_info.dns_records:
                    recommendations.append("⚠️ No A or AAAA records found")
                    dns_score -= 3.0
                
                if 'MX' not in domain_info.dns_records:
                    recommendations.append("ℹ️ No MX records found - email may not work")
            else:
                recommendations.append("🚨 No DNS records found")
            
            health_score += dns_score
            
            # Determine readiness level
            if health_score >= 80:
                readiness_level = ReadinessLevel.READY
            elif health_score >= 60:
                readiness_level = ReadinessLevel.PARTIALLY_READY
            elif health_score >= 30:
                readiness_level = ReadinessLevel.NOT_READY
            else:
                readiness_level = ReadinessLevel.UNKNOWN
            
            # Add general recommendations
            if health_score < 80:
                recommendations.append("💡 Consider implementing monitoring and alerting")
            
            if not main_website or main_website.status != WebStatus.ONLINE:
                recommendations.append("🔧 Fix main website accessibility issues")
            
            return readiness_level, min(health_score, 100.0), recommendations
            
        except Exception as e:
            self.logger.error(f"💥 Infrastructure analysis failed: {e}")
            return ReadinessLevel.UNKNOWN, 0.0, [f"Analysis error: {str(e)}"]
    
    def generate_report(self, infrastructure_status: WebInfrastructureStatus) -> str:
        """
        📊 Generate comprehensive infrastructure report
        """
        try:
            report_lines = []
            
            # Header
            report_lines.append("🌐 SPIRITUAL WEB INFRASTRUCTURE REPORT")
            report_lines.append("=" * 60)
            report_lines.append(f"Domain: {infrastructure_status.domain_info.domain}")
            report_lines.append(f"Scan Time: {infrastructure_status.scan_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"Scan Duration: {infrastructure_status.scan_duration_seconds:.2f}s")
            report_lines.append(f"Overall Health Score: {infrastructure_status.overall_health_score:.1f}/100")
            report_lines.append(f"Readiness Level: {infrastructure_status.readiness_level.value.upper()}")
            report_lines.append("")
            
            # Domain Information
            domain = infrastructure_status.domain_info
            report_lines.append("🏷️ DOMAIN INFORMATION")
            report_lines.append("-" * 30)
            report_lines.append(f"Status: {domain.status.value}")
            if domain.registrar:
                report_lines.append(f"Registrar: {domain.registrar}")
            if domain.expiration_date:
                report_lines.append(f"Expires: {domain.expiration_date.strftime('%Y-%m-%d')}")
                report_lines.append(f"Days Until Expiry: {domain.days_until_expiry}")
            if domain.nameservers:
                report_lines.append(f"Nameservers: {', '.join(domain.nameservers[:3])}")
            if domain.error:
                report_lines.append(f"Error: {domain.error}")
            report_lines.append("")
            
            # DNS Records
            if domain.dns_records:
                report_lines.append("🔍 DNS RECORDS")
                report_lines.append("-" * 20)
                for record_type, records in domain.dns_records.items():
                    report_lines.append(f"{record_type}:")
                    for record in records[:3]:  # Show first 3 records
                        report_lines.append(f"  - {record.value}")
                report_lines.append("")
            
            # Main Website
            if infrastructure_status.main_website:
                website = infrastructure_status.main_website
                report_lines.append("🌐 MAIN WEBSITE")
                report_lines.append("-" * 20)
                report_lines.append(f"URL: {website.url}")
                report_lines.append(f"Status: {website.status.value}")
                if website.status_code:
                    report_lines.append(f"HTTP Status: {website.status_code}")
                if website.response_time_ms:
                    report_lines.append(f"Response Time: {website.response_time_ms:.0f}ms")
                if website.server:
                    report_lines.append(f"Server: {website.server}")
                if website.redirect_url:
                    report_lines.append(f"Redirects to: {website.redirect_url}")
                
                # SSL Information
                if website.ssl_info:
                    ssl = website.ssl_info
                    report_lines.append(f"SSL Valid: {'Yes' if ssl.is_valid else 'No'}")
                    if ssl.expires_at:
                        report_lines.append(f"SSL Expires: {ssl.expires_at.strftime('%Y-%m-%d')}")
                        report_lines.append(f"SSL Days Left: {ssl.days_until_expiry}")
                
                if website.error:
                    report_lines.append(f"Error: {website.error}")
                report_lines.append("")
            
            # Subdomains
            if infrastructure_status.subdomains:
                report_lines.append("🔍 ACTIVE SUBDOMAINS")
                report_lines.append("-" * 25)
                for subdomain in infrastructure_status.subdomains[:10]:  # Show first 10
                    status = "Unknown"
                    if subdomain.web_service:
                        status = subdomain.web_service.status.value
                    
                    report_lines.append(f"• {subdomain.subdomain}")
                    report_lines.append(f"  Status: {status}")
                    if subdomain.ip_addresses:
                        report_lines.append(f"  IPs: {', '.join(subdomain.ip_addresses[:2])}")
                
                if len(infrastructure_status.subdomains) > 10:
                    report_lines.append(f"... and {len(infrastructure_status.subdomains) - 10} more")
                report_lines.append("")
            
            # Recommendations
            if infrastructure_status.recommendations:
                report_lines.append("💡 RECOMMENDATIONS")
                report_lines.append("-" * 25)
                for recommendation in infrastructure_status.recommendations:
                    report_lines.append(f"• {recommendation}")
                report_lines.append("")
            
            # Footer
            report_lines.append("=" * 60)
            report_lines.append("🌟 Ladang Berkah Digital - ZeroLight Orbit System")
            report_lines.append("بارك الله فيكم")
            
            return "\n".join(report_lines)
            
        except Exception as e:
            self.logger.error(f"💥 Report generation failed: {e}")
            return f"Report generation error: {str(e)}"
    
    async def shutdown(self):
        """Shutdown the web checker"""
        try:
            if self.session:
                await self.session.close()
            self.logger.info("✅ Web checker shutdown completed")
        except Exception as e:
            self.logger.error(f"💥 Shutdown error: {e}")

# 🌟 Spiritual Blessing for Web Checker
SPIRITUAL_WEB_BLESSING = """
بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ

اللَّهُمَّ بَارِكْ لَنَا فِي هَذَا النِّظَامِ الْمُبَارَكِ لِفَحْصِ الْمَوَاقِعِ
وَاجْعَلْهُ دَقِيقًا سَرِيعًا مُفِيدًا

Ya Allah, berkahilah sistem pemeriksa web ini dengan:
- 🔍 Akurasi dalam deteksi status domain dan subdomain
- ⚡ Kecepatan dalam pemindaian infrastruktur
- 🛡️ Keamanan dalam mengakses layanan eksternal
- 📊 Laporan yang jelas dan bermanfaat
- 🌐 Kemampuan monitoring yang komprehensif

آمِينَ يَا رَبَّ الْعَالَمِينَ
"""

if __name__ == "__main__":
    print("🌐 Spiritual Web Checker System - Ladang Berkah Digital")
    print("=" * 60)
    print(SPIRITUAL_WEB_BLESSING)
    
    async def example_usage():
        """Example usage of the web checker"""
        
        # Create web checker
        checker = SpiritualWebChecker({
            "timeout": 30.0,
            "max_concurrent": 10
        })
        
        await checker.initialize()
        
        try:
            # Check a domain (example)
            domain = "google.com"
            
            print(f"\n🔍 Checking infrastructure for: {domain}")
            infrastructure_status = await checker.check_web_infrastructure(
                domain=domain,
                check_subdomains=True,
                subdomain_list=["www", "mail", "drive", "docs", "maps"]
            )
            
            # Generate and print report
            report = checker.generate_report(infrastructure_status)
            print("\n" + report)
            
        finally:
            await checker.shutdown()
    
    # Run example
    asyncio.run(example_usage())