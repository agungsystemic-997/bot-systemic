#!/usr/bin/env python3
"""
🌐 SPIRITUAL WEB INTEGRATION SYSTEM
Ladang Berkah Digital - ZeroLight Orbit System
Integrated Web, Domain, and Subdomain Discovery & Analysis System
"""

import asyncio
import sys
import os
import json
import sqlite3
import requests
import socket
import ssl
import dns.resolver
import whois
from datetime import datetime, timedelta
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import logging

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

@dataclass
class DomainDiscovery:
    """Domain discovery result"""
    domain: str
    ip_address: str
    http_status: Optional[int] = None
    https_status: Optional[int] = None
    ssl_valid: bool = False
    ssl_expiry: Optional[str] = None
    registrar: Optional[str] = None
    expiry_date: Optional[str] = None
    dns_records: Dict[str, int] = None
    discovered_at: str = None
    
    def __post_init__(self):
        if self.dns_records is None:
            self.dns_records = {}
        if self.discovered_at is None:
            self.discovered_at = datetime.now().isoformat()

@dataclass
class SubdomainDiscovery:
    """Subdomain discovery result"""
    subdomain: str
    parent_domain: str
    ip_address: str
    http_accessible: bool = False
    https_accessible: bool = False
    http_status: Optional[int] = None
    https_status: Optional[int] = None
    discovered_at: str = None
    
    def __post_init__(self):
        if self.discovered_at is None:
            self.discovered_at = datetime.now().isoformat()

@dataclass
class WebDiscoveryStats:
    """Web discovery statistics"""
    total_domains_checked: int = 0
    total_subdomains_found: int = 0
    active_domains: int = 0
    active_subdomains: int = 0
    ssl_enabled_domains: int = 0
    expired_domains: int = 0
    scan_duration: float = 0.0
    last_scan: str = None
    
    def __post_init__(self):
        if self.last_scan is None:
            self.last_scan = datetime.now().isoformat()

class SpiritualWebIntegration:
    """Integrated Web Discovery and Analysis System"""
    
    def __init__(self, db_path: str = "spiritual_web_discovery.db"):
        self.db_path = db_path
        self.logger = self._setup_logging()
        self.stats = WebDiscoveryStats()
        self.discovered_domains: Dict[str, DomainDiscovery] = {}
        self.discovered_subdomains: Dict[str, SubdomainDiscovery] = {}
        self.common_subdomains = [
            'www', 'mail', 'ftp', 'blog', 'shop', 'api', 'admin', 'test', 'dev',
            'staging', 'cdn', 'static', 'img', 'images', 'assets', 'media',
            'docs', 'help', 'support', 'forum', 'community', 'news', 'events',
            'app', 'mobile', 'secure', 'login', 'auth', 'account', 'dashboard',
            'panel', 'control', 'manage', 'config', 'status', 'monitor',
            'analytics', 'stats', 'reports', 'data', 'backup', 'archive'
        ]
        self.top_level_domains = [
            '.com', '.org', '.net', '.edu', '.gov', '.mil', '.int',
            '.co.id', '.ac.id', '.or.id', '.web.id', '.sch.id',
            '.co.uk', '.org.uk', '.ac.uk', '.gov.uk',
            '.de', '.fr', '.jp', '.cn', '.ru', '.br', '.au', '.ca'
        ]
        self._init_database()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('SpiritualWebIntegration')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _init_database(self):
        """Initialize SQLite database for storing discoveries"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS domains (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain TEXT UNIQUE NOT NULL,
                    ip_address TEXT,
                    http_status INTEGER,
                    https_status INTEGER,
                    ssl_valid BOOLEAN,
                    ssl_expiry TEXT,
                    registrar TEXT,
                    expiry_date TEXT,
                    dns_records TEXT,
                    discovered_at TEXT,
                    last_checked TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS subdomains (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subdomain TEXT UNIQUE NOT NULL,
                    parent_domain TEXT,
                    ip_address TEXT,
                    http_accessible BOOLEAN,
                    https_accessible BOOLEAN,
                    http_status INTEGER,
                    https_status INTEGER,
                    discovered_at TEXT,
                    last_checked TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS discovery_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_domains_checked INTEGER,
                    total_subdomains_found INTEGER,
                    active_domains INTEGER,
                    active_subdomains INTEGER,
                    ssl_enabled_domains INTEGER,
                    expired_domains INTEGER,
                    scan_duration REAL,
                    scan_timestamp TEXT
                )
            ''')
            
            conn.commit()
    
    def check_domain_comprehensive(self, domain: str) -> DomainDiscovery:
        """Comprehensive domain checking"""
        self.logger.info(f"🔍 Checking domain: {domain}")
        
        try:
            # DNS Resolution
            ip_address = socket.gethostbyname(domain)
            discovery = DomainDiscovery(domain=domain, ip_address=ip_address)
            
            # HTTP Check
            try:
                response = requests.get(f'http://{domain}', timeout=10, allow_redirects=True)
                discovery.http_status = response.status_code
            except Exception as e:
                self.logger.debug(f"HTTP check failed for {domain}: {e}")
            
            # HTTPS Check
            try:
                response = requests.get(f'https://{domain}', timeout=10, allow_redirects=True)
                discovery.https_status = response.status_code
                
                # SSL Certificate check
                try:
                    context = ssl.create_default_context()
                    with socket.create_connection((domain, 443), timeout=10) as sock:
                        with context.wrap_socket(sock, server_hostname=domain) as ssock:
                            cert = ssock.getpeercert()
                            discovery.ssl_valid = True
                            discovery.ssl_expiry = cert.get('notAfter')
                except Exception as ssl_e:
                    self.logger.debug(f"SSL check failed for {domain}: {ssl_e}")
            except Exception as e:
                self.logger.debug(f"HTTPS check failed for {domain}: {e}")
            
            # WHOIS Information
            try:
                w = whois.whois(domain)
                if w.registrar:
                    discovery.registrar = w.registrar
                if w.expiration_date:
                    exp_date = w.expiration_date[0] if isinstance(w.expiration_date, list) else w.expiration_date
                    discovery.expiry_date = exp_date.isoformat() if hasattr(exp_date, 'isoformat') else str(exp_date)
            except Exception as e:
                self.logger.debug(f"WHOIS check failed for {domain}: {e}")
            
            # DNS Records
            dns_counts = {}
            for record_type in ['A', 'AAAA', 'MX', 'NS', 'TXT', 'CNAME']:
                try:
                    answers = dns.resolver.resolve(domain, record_type)
                    dns_counts[record_type] = len(answers)
                except:
                    dns_counts[record_type] = 0
            
            discovery.dns_records = dns_counts
            
            # Store in memory and database
            self.discovered_domains[domain] = discovery
            self._save_domain_to_db(discovery)
            
            return discovery
            
        except Exception as e:
            self.logger.error(f"Failed to check domain {domain}: {e}")
            return None
    
    def discover_subdomains(self, domain: str, custom_subdomains: List[str] = None) -> List[SubdomainDiscovery]:
        """Discover subdomains for a given domain"""
        self.logger.info(f"🔍 Discovering subdomains for: {domain}")
        
        subdomains_to_check = self.common_subdomains.copy()
        if custom_subdomains:
            subdomains_to_check.extend(custom_subdomains)
        
        discovered = []
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_subdomain = {
                executor.submit(self._check_subdomain, f"{sub}.{domain}", domain): sub 
                for sub in subdomains_to_check
            }
            
            for future in as_completed(future_to_subdomain):
                result = future.result()
                if result:
                    discovered.append(result)
                    self.discovered_subdomains[result.subdomain] = result
                    self._save_subdomain_to_db(result)
        
        self.logger.info(f"✅ Found {len(discovered)} active subdomains for {domain}")
        return discovered
    
    def _check_subdomain(self, subdomain: str, parent_domain: str) -> Optional[SubdomainDiscovery]:
        """Check individual subdomain"""
        try:
            ip_address = socket.gethostbyname(subdomain)
            discovery = SubdomainDiscovery(
                subdomain=subdomain,
                parent_domain=parent_domain,
                ip_address=ip_address
            )
            
            # HTTP check
            try:
                response = requests.get(f'http://{subdomain}', timeout=5, allow_redirects=True)
                discovery.http_accessible = True
                discovery.http_status = response.status_code
            except:
                discovery.http_accessible = False
            
            # HTTPS check
            try:
                response = requests.get(f'https://{subdomain}', timeout=5, allow_redirects=True)
                discovery.https_accessible = True
                discovery.https_status = response.status_code
            except:
                discovery.https_accessible = False
            
            return discovery
            
        except Exception:
            return None
    
    def batch_domain_scan(self, domains: List[str]) -> Dict[str, any]:
        """Batch scan multiple domains"""
        self.logger.info(f"🚀 Starting batch scan for {len(domains)} domains")
        start_time = time.time()
        
        results = {
            'domains': {},
            'subdomains': {},
            'stats': {}
        }
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Check domains
            domain_futures = {
                executor.submit(self.check_domain_comprehensive, domain): domain 
                for domain in domains
            }
            
            for future in as_completed(domain_futures):
                domain = domain_futures[future]
                result = future.result()
                if result:
                    results['domains'][domain] = asdict(result)
                    
                    # Discover subdomains for each domain
                    subdomains = self.discover_subdomains(domain)
                    results['subdomains'][domain] = [asdict(sub) for sub in subdomains]
        
        # Calculate statistics
        scan_duration = time.time() - start_time
        self._update_stats(scan_duration)
        results['stats'] = asdict(self.stats)
        
        self.logger.info(f"✅ Batch scan completed in {scan_duration:.2f} seconds")
        return results
    
    def discover_related_domains(self, base_domain: str) -> List[str]:
        """Discover related domains by trying different TLDs"""
        self.logger.info(f"🔍 Discovering related domains for: {base_domain}")
        
        # Extract domain name without TLD
        domain_parts = base_domain.split('.')
        if len(domain_parts) >= 2:
            domain_name = '.'.join(domain_parts[:-1])
        else:
            domain_name = base_domain
        
        related_domains = []
        
        for tld in self.top_level_domains:
            test_domain = f"{domain_name}{tld}"
            if test_domain != base_domain:
                try:
                    socket.gethostbyname(test_domain)
                    related_domains.append(test_domain)
                    self.logger.info(f"✅ Found related domain: {test_domain}")
                except:
                    pass
        
        return related_domains
    
    def _update_stats(self, scan_duration: float):
        """Update discovery statistics"""
        self.stats.total_domains_checked = len(self.discovered_domains)
        self.stats.total_subdomains_found = len(self.discovered_subdomains)
        self.stats.active_domains = sum(1 for d in self.discovered_domains.values() 
                                      if d.http_status or d.https_status)
        self.stats.active_subdomains = sum(1 for s in self.discovered_subdomains.values() 
                                         if s.http_accessible or s.https_accessible)
        self.stats.ssl_enabled_domains = sum(1 for d in self.discovered_domains.values() 
                                           if d.ssl_valid)
        self.stats.scan_duration = scan_duration
        self.stats.last_scan = datetime.now().isoformat()
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO discovery_stats 
                (total_domains_checked, total_subdomains_found, active_domains, 
                 active_subdomains, ssl_enabled_domains, expired_domains, 
                 scan_duration, scan_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.stats.total_domains_checked,
                self.stats.total_subdomains_found,
                self.stats.active_domains,
                self.stats.active_subdomains,
                self.stats.ssl_enabled_domains,
                self.stats.expired_domains,
                self.stats.scan_duration,
                self.stats.last_scan
            ))
            conn.commit()
    
    def _save_domain_to_db(self, domain: DomainDiscovery):
        """Save domain discovery to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO domains 
                (domain, ip_address, http_status, https_status, ssl_valid, 
                 ssl_expiry, registrar, expiry_date, dns_records, discovered_at, last_checked)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                domain.domain, domain.ip_address, domain.http_status, domain.https_status,
                domain.ssl_valid, domain.ssl_expiry, domain.registrar, domain.expiry_date,
                json.dumps(domain.dns_records), domain.discovered_at, datetime.now().isoformat()
            ))
            conn.commit()
    
    def _save_subdomain_to_db(self, subdomain: SubdomainDiscovery):
        """Save subdomain discovery to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO subdomains 
                (subdomain, parent_domain, ip_address, http_accessible, https_accessible,
                 http_status, https_status, discovered_at, last_checked)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                subdomain.subdomain, subdomain.parent_domain, subdomain.ip_address,
                subdomain.http_accessible, subdomain.https_accessible,
                subdomain.http_status, subdomain.https_status,
                subdomain.discovered_at, datetime.now().isoformat()
            ))
            conn.commit()
    
    def get_discovery_report(self) -> Dict[str, any]:
        """Generate comprehensive discovery report"""
        report = {
            'summary': asdict(self.stats),
            'domains': {domain: asdict(info) for domain, info in self.discovered_domains.items()},
            'subdomains': {sub: asdict(info) for sub, info in self.discovered_subdomains.items()},
            'top_registrars': self._get_top_registrars(),
            'ssl_status': self._get_ssl_status(),
            'expiring_domains': self._get_expiring_domains(),
            'generated_at': datetime.now().isoformat()
        }
        return report
    
    def _get_top_registrars(self) -> Dict[str, int]:
        """Get top registrars by count"""
        registrars = {}
        for domain in self.discovered_domains.values():
            if domain.registrar:
                registrars[domain.registrar] = registrars.get(domain.registrar, 0) + 1
        return dict(sorted(registrars.items(), key=lambda x: x[1], reverse=True)[:10])
    
    def _get_ssl_status(self) -> Dict[str, int]:
        """Get SSL status summary"""
        ssl_enabled = sum(1 for d in self.discovered_domains.values() if d.ssl_valid)
        ssl_disabled = len(self.discovered_domains) - ssl_enabled
        return {'ssl_enabled': ssl_enabled, 'ssl_disabled': ssl_disabled}
    
    def _get_expiring_domains(self) -> List[Dict[str, str]]:
        """Get domains expiring within 30 days"""
        expiring = []
        cutoff_date = datetime.now() + timedelta(days=30)
        
        for domain in self.discovered_domains.values():
            if domain.expiry_date:
                try:
                    expiry = datetime.fromisoformat(domain.expiry_date.replace('Z', '+00:00'))
                    if expiry <= cutoff_date:
                        expiring.append({
                            'domain': domain.domain,
                            'expiry_date': domain.expiry_date,
                            'days_remaining': (expiry - datetime.now()).days
                        })
                except:
                    pass
        
        return sorted(expiring, key=lambda x: x['days_remaining'])

def main():
    """Main function for testing"""
    print("🌟 SPIRITUAL WEB INTEGRATION SYSTEM 🌟")
    print("Ladang Berkah Digital - ZeroLight Orbit System")
    print("=" * 60)
    
    # Initialize the system
    web_integration = SpiritualWebIntegration()
    
    # Test domains
    test_domains = [
        'google.com',
        'github.com', 
        'stackoverflow.com',
        'python.org',
        'microsoft.com'
    ]
    
    print(f"\n🚀 Starting comprehensive scan for {len(test_domains)} domains...")
    
    # Perform batch scan
    results = web_integration.batch_domain_scan(test_domains)
    
    # Generate report
    report = web_integration.get_discovery_report()
    
    # Display results
    print(f"\n📊 DISCOVERY RESULTS:")
    print(f"✅ Total Domains Checked: {report['summary']['total_domains_checked']}")
    print(f"✅ Total Subdomains Found: {report['summary']['total_subdomains_found']}")
    print(f"✅ Active Domains: {report['summary']['active_domains']}")
    print(f"✅ Active Subdomains: {report['summary']['active_subdomains']}")
    print(f"✅ SSL Enabled Domains: {report['summary']['ssl_enabled_domains']}")
    print(f"⏱️ Scan Duration: {report['summary']['scan_duration']:.2f} seconds")
    
    print(f"\n🔒 SSL Status:")
    ssl_status = report['ssl_status']
    print(f"✅ SSL Enabled: {ssl_status['ssl_enabled']}")
    print(f"❌ SSL Disabled: {ssl_status['ssl_disabled']}")
    
    if report['top_registrars']:
        print(f"\n🏢 Top Registrars:")
        for registrar, count in list(report['top_registrars'].items())[:5]:
            print(f"  • {registrar}: {count} domains")
    
    print(f"\n✨ Scan completed successfully!")
    print(f"📁 Data saved to: {web_integration.db_path}")

if __name__ == "__main__":
    main()