#!/usr/bin/env python3
"""
Simple Web Infrastructure Checker Test
"""

import requests
import socket
import ssl
import dns.resolver
import whois
from datetime import datetime
import sys

def check_domain_basic(domain):
    """Basic domain check"""
    print(f"\n🌐 Checking domain: {domain}")
    print("=" * 50)
    
    # 1. DNS Resolution
    try:
        result = socket.gethostbyname(domain)
        print(f"✅ DNS Resolution: {result}")
    except Exception as e:
        print(f"❌ DNS Resolution failed: {e}")
        return
    
    # 2. HTTP Check
    try:
        response = requests.get(f"http://{domain}", timeout=10)
        print(f"✅ HTTP Status: {response.status_code}")
    except Exception as e:
        print(f"❌ HTTP failed: {e}")
    
    # 3. HTTPS Check
    try:
        response = requests.get(f"https://{domain}", timeout=10)
        print(f"✅ HTTPS Status: {response.status_code}")
        
        # SSL Certificate check
        context = ssl.create_default_context()
        with socket.create_connection((domain, 443), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname=domain) as ssock:
                cert = ssock.getpeercert()
                print(f"✅ SSL Certificate: Valid until {cert['notAfter']}")
    except Exception as e:
        print(f"❌ HTTPS/SSL failed: {e}")
    
    # 4. WHOIS Information
    try:
        w = whois.whois(domain)
        if w.expiration_date:
            exp_date = w.expiration_date[0] if isinstance(w.expiration_date, list) else w.expiration_date
            print(f"✅ Domain expires: {exp_date}")
        if w.registrar:
            print(f"✅ Registrar: {w.registrar}")
    except Exception as e:
        print(f"❌ WHOIS failed: {e}")
    
    # 5. DNS Records
    try:
        for record_type in ['A', 'AAAA', 'MX', 'NS', 'TXT']:
            try:
                answers = dns.resolver.resolve(domain, record_type)
                print(f"✅ {record_type} records: {len(answers)} found")
            except:
                pass
    except Exception as e:
        print(f"❌ DNS records check failed: {e}")

def check_subdomain(subdomain):
    """Check subdomain status"""
    print(f"\n🔍 Checking subdomain: {subdomain}")
    print("-" * 30)
    
    try:
        ip = socket.gethostbyname(subdomain)
        print(f"✅ {subdomain} -> {ip}")
        
        # HTTP check
        try:
            response = requests.get(f"http://{subdomain}", timeout=5)
            print(f"✅ HTTP: {response.status_code}")
        except:
            print(f"❌ HTTP: Not accessible")
        
        # HTTPS check
        try:
            response = requests.get(f"https://{subdomain}", timeout=5)
            print(f"✅ HTTPS: {response.status_code}")
        except:
            print(f"❌ HTTPS: Not accessible")
            
    except Exception as e:
        print(f"❌ {subdomain}: Not found")

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_web_checker.py <domain>")
        print("Example: python test_web_checker.py google.com")
        return
    
    domain = sys.argv[1]
    
    print("🌟 SPIRITUAL WEB INFRASTRUCTURE CHECKER 🌟")
    print("Ladang Berkah Digital - ZeroLight Orbit System")
    print(f"Started at: {datetime.now()}")
    
    # Check main domain
    check_domain_basic(domain)
    
    # Check common subdomains
    common_subdomains = ['www', 'mail', 'ftp', 'blog', 'shop', 'api', 'admin', 'test']
    print(f"\n🔍 Checking common subdomains for {domain}:")
    print("=" * 50)
    
    for sub in common_subdomains:
        subdomain = f"{sub}.{domain}"
        check_subdomain(subdomain)
    
    print(f"\n✨ Check completed at: {datetime.now()}")
    print("🌟 May this information serve you well! 🌟")

if __name__ == "__main__":
    main()