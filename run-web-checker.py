#!/usr/bin/env python3
"""
🌐 SPIRITUAL WEB CHECKER RUNNER
Ladang Berkah Digital - ZeroLight Orbit System
Main runner for web infrastructure checking system
"""

import sys
import os
import asyncio
import argparse
import logging
from typing import List, Optional

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from spiritual_web_checker import SpiritualWebChecker
    from spiritual_web_dashboard import SpiritualWebDashboard
except ImportError as e:
    print(f"💥 Import error: {e}")
    print("Please ensure all required dependencies are installed:")
    print("pip install -r web-checker-requirements.txt")
    sys.exit(1)

# Import datetime for logging
from datetime import datetime

# Spiritual blessings
SPIRITUAL_WEB_BLESSING = """
🌟 بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيم 🌟
May this web infrastructure check bring clarity and insight
Ladang Berkah Digital - ZeroLight Orbit System
"""

SPIRITUAL_DASHBOARD_BLESSING = """
🌟 بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيم 🌟
May this dashboard serve with wisdom and guidance
Ladang Berkah Digital - ZeroLight Orbit System
"""

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'web-checker-{datetime.now().strftime("%Y%m%d")}.log')
        ]
    )

async def run_single_check(domain: str, verbose: bool = False):
    """Run a single domain check"""
    try:
        print("🌐 SPIRITUAL WEB CHECKER - Single Domain Check")
        print("=" * 60)
        print(SPIRITUAL_WEB_BLESSING)
        print("=" * 60)
        
        # Create and initialize checker
        checker = SpiritualWebChecker({
            "timeout": 30.0,
            "max_concurrent": 10
        })
        
        await checker.initialize()
        
        try:
            print(f"\n🔍 Checking infrastructure for: {domain}")
            print("⏳ This may take a few minutes...")
            
            # Perform infrastructure check
            infrastructure_status = await checker.check_web_infrastructure(
                domain=domain,
                check_subdomains=True,
                subdomain_list=[
                    "www", "api", "app", "admin", "blog", "shop", "mail", "ftp",
                    "cdn", "static", "assets", "dev", "test", "staging", "docs",
                    "support", "portal", "dashboard", "secure", "mobile", "m"
                ]
            )
            
            # Generate and display report
            report = checker.generate_report(infrastructure_status)
            print("\n" + report)
            
            # Save report to file
            report_filename = f"web-report-{domain.replace('.', '-')}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"\n📄 Report saved to: {report_filename}")
            
            # Summary
            print(f"\n📊 SUMMARY FOR {domain.upper()}")
            print("-" * 40)
            print(f"🏷️  Domain Status: {infrastructure_status.domain_info.status.value}")
            print(f"🌐 Website Status: {infrastructure_status.main_website.status.value if infrastructure_status.main_website else 'N/A'}")
            print(f"🔍 Subdomains Found: {len(infrastructure_status.subdomains)}")
            print(f"📈 Health Score: {infrastructure_status.overall_health_score:.1f}/100")
            print(f"🎯 Readiness Level: {infrastructure_status.readiness_level.value}")
            print(f"⏱️  Scan Duration: {infrastructure_status.scan_duration_seconds:.1f}s")
            
            if infrastructure_status.recommendations:
                print(f"\n💡 Recommendations: {len(infrastructure_status.recommendations)} items")
                for i, rec in enumerate(infrastructure_status.recommendations[:3], 1):
                    print(f"   {i}. {rec}")
                if len(infrastructure_status.recommendations) > 3:
                    print(f"   ... and {len(infrastructure_status.recommendations) - 3} more (see full report)")
            
        finally:
            await checker.shutdown()
            
    except Exception as e:
        print(f"💥 Check failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

async def run_dashboard(host: str = "localhost", port: int = 8080):
    """Run the web dashboard"""
    try:
        print("🌐 SPIRITUAL WEB DASHBOARD")
        print("=" * 60)
        print(SPIRITUAL_DASHBOARD_BLESSING)
        print("=" * 60)
        
        # Create and initialize dashboard
        dashboard = SpiritualWebDashboard(host=host, port=port)
        await dashboard.initialize()
        
        try:
            dashboard.start_server()
            
            print(f"\n🌐 Dashboard is running at: http://{host}:{port}")
            print("🎯 Features available:")
            print("   • Real-time domain and subdomain checking")
            print("   • SSL certificate monitoring")
            print("   • DNS records analysis")
            print("   • Performance metrics")
            print("   • Health scoring and recommendations")
            print("\n⌨️  Press Ctrl+C to stop the server")
            
            # Keep running
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🔄 Stopping dashboard...")
        finally:
            await dashboard.shutdown()
            
    except Exception as e:
        print(f"💥 Dashboard failed: {e}")
        return 1
    
    return 0

async def run_batch_check(domains_file: str, output_dir: str = "reports"):
    """Run batch check for multiple domains"""
    try:
        print("🌐 SPIRITUAL WEB CHECKER - Batch Mode")
        print("=" * 60)
        
        # Read domains from file
        domains = []
        with open(domains_file, 'r') as f:
            for line in f:
                domain = line.strip()
                if domain and not domain.startswith('#'):
                    domains.append(domain)
        
        if not domains:
            print("❌ No domains found in file")
            return 1
        
        print(f"📋 Found {len(domains)} domains to check")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create checker
        checker = SpiritualWebChecker({
            "timeout": 30.0,
            "max_concurrent": 5  # Lower concurrency for batch
        })
        
        await checker.initialize()
        
        try:
            results = []
            
            for i, domain in enumerate(domains, 1):
                print(f"\n🔍 [{i}/{len(domains)}] Checking: {domain}")
                
                try:
                    infrastructure_status = await checker.check_web_infrastructure(
                        domain=domain,
                        check_subdomains=True,
                        subdomain_list=["www", "api", "app", "admin", "blog", "mail", "cdn"]
                    )
                    
                    results.append(infrastructure_status)
                    
                    # Save individual report
                    report = checker.generate_report(infrastructure_status)
                    report_file = os.path.join(output_dir, f"{domain.replace('.', '-')}-report.txt")
                    with open(report_file, 'w', encoding='utf-8') as f:
                        f.write(report)
                    
                    print(f"   ✅ Health: {infrastructure_status.overall_health_score:.1f}/100")
                    print(f"   📄 Report: {report_file}")
                    
                except Exception as e:
                    print(f"   ❌ Failed: {e}")
                    continue
            
            # Generate summary report
            summary_file = os.path.join(output_dir, f"batch-summary-{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt")
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("🌐 SPIRITUAL WEB CHECKER - BATCH SUMMARY\n")
                f.write("=" * 60 + "\n")
                f.write(f"Total Domains Checked: {len(results)}\n")
                f.write(f"Check Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for result in results:
                    f.write(f"Domain: {result.domain_info.domain}\n")
                    f.write(f"Health Score: {result.overall_health_score:.1f}/100\n")
                    f.write(f"Readiness: {result.readiness_level.value}\n")
                    f.write(f"Subdomains: {len(result.subdomains)}\n")
                    f.write("-" * 40 + "\n")
            
            print(f"\n📊 Batch check completed!")
            print(f"📄 Summary report: {summary_file}")
            
        finally:
            await checker.shutdown()
            
    except Exception as e:
        print(f"💥 Batch check failed: {e}")
        return 1
    
    return 0

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="🌐 Spiritual Web Checker - Ladang Berkah Digital",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check single domain
  python run-web-checker.py check google.com
  
  # Run web dashboard
  python run-web-checker.py dashboard
  
  # Run dashboard on custom host/port
  python run-web-checker.py dashboard --host 0.0.0.0 --port 8080
  
  # Batch check from file
  python run-web-checker.py batch domains.txt
  
  # Verbose output
  python run-web-checker.py check google.com --verbose
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Single check command
    check_parser = subparsers.add_parser('check', help='Check single domain')
    check_parser.add_argument('domain', help='Domain to check (e.g., google.com)')
    check_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Run web dashboard')
    dashboard_parser.add_argument('--host', default='localhost', help='Dashboard host (default: localhost)')
    dashboard_parser.add_argument('--port', type=int, default=8080, help='Dashboard port (default: 8080)')
    
    # Batch check command
    batch_parser = subparsers.add_parser('batch', help='Batch check multiple domains')
    batch_parser.add_argument('domains_file', help='File containing domains (one per line)')
    batch_parser.add_argument('--output-dir', default='reports', help='Output directory for reports')
    
    # Global options
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Handle commands
    if args.command == 'check':
        if not args.domain:
            print("❌ Domain is required for check command")
            return 1
        return asyncio.run(run_single_check(args.domain, args.verbose))
    
    elif args.command == 'dashboard':
        return asyncio.run(run_dashboard(args.host, args.port))
    
    elif args.command == 'batch':
        if not os.path.exists(args.domains_file):
            print(f"❌ Domains file not found: {args.domains_file}")
            return 1
        return asyncio.run(run_batch_check(args.domains_file, args.output_dir))
    
    else:
        # No command specified, show help
        parser.print_help()
        return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🔄 Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        sys.exit(1)