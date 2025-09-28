#!/usr/bin/env python3
"""
📊 SPIRITUAL WEB REPORTER
Ladang Berkah Digital - ZeroLight Orbit System
Comprehensive Web/Domain/Subdomain Counter and Reporter
"""

import sqlite3
import json
import csv
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
import sys

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from spiritual_web_integration import SpiritualWebIntegration
except ImportError:
    print("⚠️ Could not import SpiritualWebIntegration")
    SpiritualWebIntegration = None

class SpiritualWebReporter:
    """Comprehensive Web Discovery Reporter and Counter"""
    
    def __init__(self, db_path: str = "spiritual_web_discovery.db"):
        self.db_path = db_path
        self.web_integration = None
        
        # Initialize web integration if available
        if SpiritualWebIntegration:
            try:
                self.web_integration = SpiritualWebIntegration()
            except Exception as e:
                print(f"⚠️ Web integration initialization failed: {e}")
    
    def get_comprehensive_counts(self) -> Dict[str, Any]:
        """Get comprehensive counts of all discovered web assets"""
        counts = {
            'domains': {
                'total': 0,
                'active': 0,
                'inactive': 0,
                'ssl_enabled': 0,
                'ssl_disabled': 0,
                'by_registrar': {},
                'by_tld': {},
                'recent_discoveries': 0  # Last 24 hours
            },
            'subdomains': {
                'total': 0,
                'active': 0,
                'inactive': 0,
                'http_accessible': 0,
                'https_accessible': 0,
                'by_parent_domain': {},
                'recent_discoveries': 0  # Last 24 hours
            },
            'web_services': {
                'total_endpoints': 0,
                'http_only': 0,
                'https_only': 0,
                'both_protocols': 0,
                'status_codes': {},
                'response_times': {
                    'fast': 0,      # < 1s
                    'medium': 0,    # 1-3s
                    'slow': 0       # > 3s
                }
            },
            'security': {
                'ssl_certificates': {
                    'valid': 0,
                    'expired': 0,
                    'self_signed': 0,
                    'ca_issued': 0
                },
                'security_headers': {
                    'hsts_enabled': 0,
                    'csp_enabled': 0,
                    'x_frame_options': 0
                }
            },
            'discovery_timeline': {
                'last_24h': 0,
                'last_week': 0,
                'last_month': 0,
                'total_scans': 0
            }
        }
        
        if not os.path.exists(self.db_path):
            print(f"⚠️ Database not found: {self.db_path}")
            return counts
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Count domains
                domains = conn.execute("SELECT * FROM domains").fetchall()
                counts['domains']['total'] = len(domains)
                
                # Analyze domains
                for domain in domains:
                    # Active/inactive status
                    if domain['http_status'] and domain['http_status'] != 'N/A':
                        counts['domains']['active'] += 1
                    else:
                        counts['domains']['inactive'] += 1
                    
                    # SSL status
                    if domain['ssl_valid']:
                        counts['domains']['ssl_enabled'] += 1
                    else:
                        counts['domains']['ssl_disabled'] += 1
                    
                    # Registrar analysis
                    registrar = domain['registrar'] or 'Unknown'
                    counts['domains']['by_registrar'][registrar] = counts['domains']['by_registrar'].get(registrar, 0) + 1
                    
                    # TLD analysis
                    tld = domain['domain'].split('.')[-1] if '.' in domain['domain'] else 'unknown'
                    counts['domains']['by_tld'][tld] = counts['domains']['by_tld'].get(tld, 0) + 1
                    
                    # Recent discoveries (last 24 hours)
                    if domain['last_checked']:
                        try:
                            last_checked = datetime.fromisoformat(domain['last_checked'])
                            if datetime.now() - last_checked < timedelta(hours=24):
                                counts['domains']['recent_discoveries'] += 1
                        except:
                            pass
                
                # Count subdomains
                subdomains = conn.execute("SELECT * FROM subdomains").fetchall()
                counts['subdomains']['total'] = len(subdomains)
                
                # Analyze subdomains
                for subdomain in subdomains:
                    # Active/inactive status
                    if subdomain['http_accessible'] or subdomain['https_accessible']:
                        counts['subdomains']['active'] += 1
                    else:
                        counts['subdomains']['inactive'] += 1
                    
                    # Protocol accessibility
                    if subdomain['http_accessible']:
                        counts['subdomains']['http_accessible'] += 1
                    if subdomain['https_accessible']:
                        counts['subdomains']['https_accessible'] += 1
                    
                    # Parent domain analysis
                    parent = subdomain['parent_domain']
                    counts['subdomains']['by_parent_domain'][parent] = counts['subdomains']['by_parent_domain'].get(parent, 0) + 1
                    
                    # Recent discoveries
                    if subdomain['last_checked']:
                        try:
                            last_checked = datetime.fromisoformat(subdomain['last_checked'])
                            if datetime.now() - last_checked < timedelta(hours=24):
                                counts['subdomains']['recent_discoveries'] += 1
                        except:
                            pass
                
                # Web services analysis
                total_endpoints = counts['domains']['total'] + counts['subdomains']['total']
                counts['web_services']['total_endpoints'] = total_endpoints
                
                # Calculate protocol distribution
                http_only = 0
                https_only = 0
                both_protocols = 0
                
                for domain in domains:
                    has_http = domain['http_status'] and domain['http_status'] != 'N/A'
                    has_https = domain['https_status'] and domain['https_status'] != 'N/A'
                    
                    if has_http and has_https:
                        both_protocols += 1
                    elif has_http:
                        http_only += 1
                    elif has_https:
                        https_only += 1
                
                for subdomain in subdomains:
                    has_http = subdomain['http_accessible']
                    has_https = subdomain['https_accessible']
                    
                    if has_http and has_https:
                        both_protocols += 1
                    elif has_http:
                        http_only += 1
                    elif has_https:
                        https_only += 1
                
                counts['web_services']['http_only'] = http_only
                counts['web_services']['https_only'] = https_only
                counts['web_services']['both_protocols'] = both_protocols
                
                # Security analysis
                counts['security']['ssl_certificates']['valid'] = counts['domains']['ssl_enabled']
                counts['security']['ssl_certificates']['expired'] = counts['domains']['ssl_disabled']
                
                # Timeline analysis
                counts['discovery_timeline']['last_24h'] = counts['domains']['recent_discoveries'] + counts['subdomains']['recent_discoveries']
                counts['discovery_timeline']['total_scans'] = len(domains)
                
        except Exception as e:
            print(f"❌ Error analyzing database: {e}")
        
        return counts
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report"""
        counts = self.get_comprehensive_counts()
        
        report = []
        report.append("🌟 SPIRITUAL WEB DISCOVERY REPORT 🌟")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Database: {self.db_path}")
        report.append("")
        
        # Domain Summary
        report.append("🌐 DOMAIN SUMMARY")
        report.append("-" * 20)
        report.append(f"Total Domains Discovered: {counts['domains']['total']}")
        report.append(f"Active Domains: {counts['domains']['active']}")
        report.append(f"Inactive Domains: {counts['domains']['inactive']}")
        report.append(f"SSL Enabled: {counts['domains']['ssl_enabled']}")
        report.append(f"SSL Disabled: {counts['domains']['ssl_disabled']}")
        report.append(f"Recent Discoveries (24h): {counts['domains']['recent_discoveries']}")
        report.append("")
        
        # Subdomain Summary
        report.append("🔍 SUBDOMAIN SUMMARY")
        report.append("-" * 22)
        report.append(f"Total Subdomains Discovered: {counts['subdomains']['total']}")
        report.append(f"Active Subdomains: {counts['subdomains']['active']}")
        report.append(f"Inactive Subdomains: {counts['subdomains']['inactive']}")
        report.append(f"HTTP Accessible: {counts['subdomains']['http_accessible']}")
        report.append(f"HTTPS Accessible: {counts['subdomains']['https_accessible']}")
        report.append(f"Recent Discoveries (24h): {counts['subdomains']['recent_discoveries']}")
        report.append("")
        
        # Web Services Summary
        report.append("⚡ WEB SERVICES SUMMARY")
        report.append("-" * 25)
        report.append(f"Total Endpoints: {counts['web_services']['total_endpoints']}")
        report.append(f"HTTP Only: {counts['web_services']['http_only']}")
        report.append(f"HTTPS Only: {counts['web_services']['https_only']}")
        report.append(f"Both Protocols: {counts['web_services']['both_protocols']}")
        report.append("")
        
        # Top Registrars
        if counts['domains']['by_registrar']:
            report.append("🏢 TOP REGISTRARS")
            report.append("-" * 16)
            sorted_registrars = sorted(counts['domains']['by_registrar'].items(), key=lambda x: x[1], reverse=True)
            for registrar, count in sorted_registrars[:10]:
                report.append(f"  {registrar}: {count} domains")
            report.append("")
        
        # Top TLDs
        if counts['domains']['by_tld']:
            report.append("🌍 TOP TLDs")
            report.append("-" * 11)
            sorted_tlds = sorted(counts['domains']['by_tld'].items(), key=lambda x: x[1], reverse=True)
            for tld, count in sorted_tlds[:10]:
                report.append(f"  .{tld}: {count} domains")
            report.append("")
        
        # Top Parent Domains by Subdomain Count
        if counts['subdomains']['by_parent_domain']:
            report.append("🔗 TOP DOMAINS BY SUBDOMAIN COUNT")
            report.append("-" * 35)
            sorted_parents = sorted(counts['subdomains']['by_parent_domain'].items(), key=lambda x: x[1], reverse=True)
            for parent, count in sorted_parents[:10]:
                report.append(f"  {parent}: {count} subdomains")
            report.append("")
        
        # Security Summary
        report.append("🔒 SECURITY SUMMARY")
        report.append("-" * 18)
        report.append(f"Valid SSL Certificates: {counts['security']['ssl_certificates']['valid']}")
        report.append(f"Expired/Invalid SSL: {counts['security']['ssl_certificates']['expired']}")
        ssl_percentage = (counts['security']['ssl_certificates']['valid'] / max(counts['domains']['total'], 1)) * 100
        report.append(f"SSL Adoption Rate: {ssl_percentage:.1f}%")
        report.append("")
        
        # Discovery Timeline
        report.append("📅 DISCOVERY TIMELINE")
        report.append("-" * 20)
        report.append(f"Discoveries in Last 24h: {counts['discovery_timeline']['last_24h']}")
        report.append(f"Total Scans Performed: {counts['discovery_timeline']['total_scans']}")
        report.append("")
        
        # Grand Total
        total_assets = counts['domains']['total'] + counts['subdomains']['total']
        report.append("🎯 GRAND TOTAL")
        report.append("-" * 13)
        report.append(f"Total Web Assets Discovered: {total_assets}")
        report.append(f"  ├─ Domains: {counts['domains']['total']}")
        report.append(f"  └─ Subdomains: {counts['subdomains']['total']}")
        report.append("")
        
        report.append("✨ Ladang Berkah Digital - ZeroLight Orbit System ✨")
        
        return "\n".join(report)
    
    def export_to_json(self, filename: str = None) -> str:
        """Export comprehensive data to JSON"""
        if not filename:
            filename = f"spiritual_web_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        counts = self.get_comprehensive_counts()
        
        # Add detailed data
        detailed_data = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'database_path': self.db_path,
                'report_version': '1.0'
            },
            'summary': counts,
            'detailed_domains': [],
            'detailed_subdomains': []
        }
        
        # Get detailed domain and subdomain data
        if os.path.exists(self.db_path):
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    
                    # Get detailed domain data
                    domains = conn.execute("SELECT * FROM domains ORDER BY domain").fetchall()
                    detailed_data['detailed_domains'] = [dict(row) for row in domains]
                    
                    # Get detailed subdomain data
                    subdomains = conn.execute("SELECT * FROM subdomains ORDER BY parent_domain, subdomain").fetchall()
                    detailed_data['detailed_subdomains'] = [dict(row) for row in subdomains]
                    
            except Exception as e:
                print(f"❌ Error exporting detailed data: {e}")
        
        # Write to file
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(detailed_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ JSON report exported to: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Error writing JSON file: {e}")
            return ""
    
    def export_to_csv(self, filename: str = None) -> str:
        """Export data to CSV format"""
        if not filename:
            filename = f"spiritual_web_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        if not os.path.exists(self.db_path):
            print(f"⚠️ Database not found: {self.db_path}")
            return ""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Get all data
                domains = conn.execute("SELECT * FROM domains").fetchall()
                subdomains = conn.execute("SELECT * FROM subdomains").fetchall()
                
                with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    # Write domains section
                    writer.writerow(['=== DOMAINS ==='])
                    if domains:
                        writer.writerow(list(domains[0].keys()))
                        for domain in domains:
                            writer.writerow(list(domain))
                    
                    writer.writerow([])  # Empty row
                    
                    # Write subdomains section
                    writer.writerow(['=== SUBDOMAINS ==='])
                    if subdomains:
                        writer.writerow(list(subdomains[0].keys()))
                        for subdomain in subdomains:
                            writer.writerow(list(subdomain))
            
            print(f"✅ CSV data exported to: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Error writing CSV file: {e}")
            return ""
    
    def print_quick_stats(self):
        """Print quick statistics to console"""
        counts = self.get_comprehensive_counts()
        
        print("\n🌟 QUICK SPIRITUAL WEB STATS 🌟")
        print("=" * 40)
        print(f"📊 Total Domains: {counts['domains']['total']}")
        print(f"🔍 Total Subdomains: {counts['subdomains']['total']}")
        print(f"⚡ Active Domains: {counts['domains']['active']}")
        print(f"🔗 Active Subdomains: {counts['subdomains']['active']}")
        print(f"🔒 SSL Enabled: {counts['domains']['ssl_enabled']}")
        print(f"🌐 Total Web Assets: {counts['domains']['total'] + counts['subdomains']['total']}")
        print(f"📅 Recent Discoveries (24h): {counts['discovery_timeline']['last_24h']}")
        print("=" * 40)
    
    def run_comprehensive_analysis(self):
        """Run comprehensive analysis and generate all reports"""
        print("🚀 Starting Comprehensive Web Discovery Analysis...")
        
        # Print quick stats
        self.print_quick_stats()
        
        # Generate summary report
        print("\n📝 Generating Summary Report...")
        summary = self.generate_summary_report()
        print(summary)
        
        # Export to JSON
        print("\n💾 Exporting to JSON...")
        json_file = self.export_to_json()
        
        # Export to CSV
        print("\n📊 Exporting to CSV...")
        csv_file = self.export_to_csv()
        
        print(f"\n✨ Analysis Complete!")
        print(f"📁 Files generated:")
        if json_file:
            print(f"  - JSON Report: {json_file}")
        if csv_file:
            print(f"  - CSV Data: {csv_file}")

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Spiritual Web Discovery Reporter")
    parser.add_argument('--db', default='spiritual_web_discovery.db', help='Database path')
    parser.add_argument('--quick', action='store_true', help='Show quick stats only')
    parser.add_argument('--json', help='Export to JSON file')
    parser.add_argument('--csv', help='Export to CSV file')
    parser.add_argument('--full', action='store_true', help='Run full comprehensive analysis')
    
    args = parser.parse_args()
    
    reporter = SpiritualWebReporter(args.db)
    
    if args.quick:
        reporter.print_quick_stats()
    elif args.json:
        reporter.export_to_json(args.json)
    elif args.csv:
        reporter.export_to_csv(args.csv)
    elif args.full:
        reporter.run_comprehensive_analysis()
    else:
        # Default: show summary report
        print(reporter.generate_summary_report())

if __name__ == "__main__":
    main()