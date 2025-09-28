#!/usr/bin/env python3
"""
🌟 SPIRITUAL SYSTEM STATUS CHECKER 🌟
ZeroLight Orbit - Status Monitor untuk 6993 Static Bots
Monitor independen untuk mengecek status sistem bot

Author: ZeroLight Orbit Team
Version: 1.0.0
License: Spiritual Open Source License
"""

import os
import sys
import json
import time
import psutil
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import argparse

@dataclass
class SystemStatus:
    """Status sistem spiritual"""
    timestamp: datetime
    total_bots: int = 0
    active_bots: int = 0
    idle_bots: int = 0
    error_bots: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_connections: int = 0
    uptime: float = 0.0
    spiritual_energy: float = 100.0
    system_health: str = "UNKNOWN"

class SpiritualSystemStatusChecker:
    """Checker untuk status sistem spiritual"""
    
    def __init__(self):
        self.db_path = "spiritual_bot_registry.db"
        self.log_file = "spiritual_system_status.log"
        self.start_time = datetime.now()
    
    def check_database_status(self) -> Dict[str, Any]:
        """Check status database bot registry"""
        try:
            if not os.path.exists(self.db_path):
                return {
                    'status': 'NOT_FOUND',
                    'message': 'Database bot registry tidak ditemukan',
                    'bots': {}
                }
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            if 'bots' not in tables:
                conn.close()
                return {
                    'status': 'NO_TABLES',
                    'message': 'Tabel bots tidak ditemukan dalam database',
                    'bots': {}
                }
            
            # Get bot statistics
            cursor.execute("SELECT status, COUNT(*) FROM bots GROUP BY status")
            status_counts = dict(cursor.fetchall())
            
            cursor.execute("SELECT COUNT(*) FROM bots")
            total_bots = cursor.fetchone()[0]
            
            cursor.execute("SELECT category, COUNT(*) FROM bots GROUP BY category")
            category_counts = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                'status': 'ACTIVE',
                'message': 'Database bot registry aktif',
                'total_bots': total_bots,
                'status_counts': status_counts,
                'category_counts': category_counts,
                'bots': {
                    'total': total_bots,
                    'active': status_counts.get('active', 0),
                    'idle': status_counts.get('idle', 0),
                    'error': status_counts.get('error', 0)
                }
            }
            
        except Exception as e:
            return {
                'status': 'ERROR',
                'message': f'Error mengakses database: {str(e)}',
                'bots': {}
            }
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check penggunaan resource sistem"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Network connections
            connections = len(psutil.net_connections())
            
            # Process count
            process_count = len(psutil.pids())
            
            return {
                'cpu_usage': cpu_percent,
                'memory_usage': memory_percent,
                'memory_total_gb': memory.total / (1024**3),
                'memory_used_gb': memory.used / (1024**3),
                'disk_usage': disk_percent,
                'disk_total_gb': disk.total / (1024**3),
                'disk_used_gb': disk.used / (1024**3),
                'network_connections': connections,
                'process_count': process_count,
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            }
            
        except Exception as e:
            return {
                'error': f'Error mengecek resource sistem: {str(e)}'
            }
    
    def check_spiritual_processes(self) -> Dict[str, Any]:
        """Check proses spiritual yang berjalan"""
        try:
            spiritual_processes = []
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if any(keyword in cmdline.lower() for keyword in ['spiritual', 'bot', 'zerolight']):
                        spiritual_processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cmdline': cmdline,
                            'cpu_percent': proc.info['cpu_percent'],
                            'memory_percent': proc.info['memory_percent']
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return {
                'count': len(spiritual_processes),
                'processes': spiritual_processes
            }
            
        except Exception as e:
            return {
                'error': f'Error mengecek proses spiritual: {str(e)}'
            }
    
    def check_log_files(self) -> Dict[str, Any]:
        """Check file log sistem"""
        log_files = [
            'spiritual_master_launcher.log',
            'spiritual_bot_registry.log',
            'spiritual_performance.log',
            'spiritual_resource_manager.log',
            'spiritual_deployment.log'
        ]
        
        log_status = {}
        
        for log_file in log_files:
            if os.path.exists(log_file):
                stat = os.stat(log_file)
                log_status[log_file] = {
                    'exists': True,
                    'size_mb': stat.st_size / (1024*1024),
                    'modified': datetime.fromtimestamp(stat.st_mtime),
                    'age_hours': (datetime.now() - datetime.fromtimestamp(stat.st_mtime)).total_seconds() / 3600
                }
            else:
                log_status[log_file] = {
                    'exists': False
                }
        
        return log_status
    
    def calculate_spiritual_energy(self, db_status: Dict, resource_status: Dict, process_status: Dict) -> float:
        """Hitung spiritual energy berdasarkan status sistem"""
        try:
            energy = 100.0
            
            # Penalti berdasarkan CPU usage
            if resource_status.get('cpu_usage', 0) > 80:
                energy -= 20
            elif resource_status.get('cpu_usage', 0) > 60:
                energy -= 10
            
            # Penalti berdasarkan memory usage
            if resource_status.get('memory_usage', 0) > 80:
                energy -= 20
            elif resource_status.get('memory_usage', 0) > 60:
                energy -= 10
            
            # Bonus berdasarkan bot aktif
            if db_status.get('status') == 'ACTIVE':
                active_bots = db_status.get('bots', {}).get('active', 0)
                if active_bots > 1000:
                    energy += 10
                elif active_bots > 500:
                    energy += 5
            
            # Penalti berdasarkan error bots
            error_bots = db_status.get('bots', {}).get('error', 0)
            if error_bots > 100:
                energy -= 15
            elif error_bots > 50:
                energy -= 10
            
            # Bonus berdasarkan proses spiritual
            spiritual_count = process_status.get('count', 0)
            if spiritual_count > 0:
                energy += min(spiritual_count * 2, 20)
            
            return max(0.0, min(100.0, energy))
            
        except Exception:
            return 50.0  # Default energy jika error
    
    def determine_system_health(self, db_status: Dict, resource_status: Dict, spiritual_energy: float) -> str:
        """Tentukan kesehatan sistem"""
        try:
            if db_status.get('status') == 'ERROR':
                return "CRITICAL"
            
            if spiritual_energy < 20:
                return "CRITICAL"
            
            if resource_status.get('cpu_usage', 0) > 90 or resource_status.get('memory_usage', 0) > 90:
                return "CRITICAL"
            
            if spiritual_energy < 50:
                return "WARNING"
            
            if resource_status.get('cpu_usage', 0) > 70 or resource_status.get('memory_usage', 0) > 70:
                return "WARNING"
            
            error_bots = db_status.get('bots', {}).get('error', 0)
            if error_bots > 50:
                return "WARNING"
            
            if spiritual_energy > 80 and db_status.get('status') == 'ACTIVE':
                return "EXCELLENT"
            
            return "GOOD"
            
        except Exception:
            return "UNKNOWN"
    
    def get_system_status(self) -> SystemStatus:
        """Get status sistem lengkap"""
        # Check database
        db_status = self.check_database_status()
        
        # Check system resources
        resource_status = self.check_system_resources()
        
        # Check spiritual processes
        process_status = self.check_spiritual_processes()
        
        # Calculate spiritual energy
        spiritual_energy = self.calculate_spiritual_energy(db_status, resource_status, process_status)
        
        # Determine system health
        system_health = self.determine_system_health(db_status, resource_status, spiritual_energy)
        
        # Create status object
        status = SystemStatus(
            timestamp=datetime.now(),
            total_bots=db_status.get('bots', {}).get('total', 0),
            active_bots=db_status.get('bots', {}).get('active', 0),
            idle_bots=db_status.get('bots', {}).get('idle', 0),
            error_bots=db_status.get('bots', {}).get('error', 0),
            cpu_usage=resource_status.get('cpu_usage', 0),
            memory_usage=resource_status.get('memory_usage', 0),
            disk_usage=resource_status.get('disk_usage', 0),
            network_connections=resource_status.get('network_connections', 0),
            uptime=(datetime.now() - self.start_time).total_seconds(),
            spiritual_energy=spiritual_energy,
            system_health=system_health
        )
        
        return status
    
    def print_status_report(self, detailed: bool = False):
        """Print laporan status sistem"""
        print("🌟" + "="*58 + "🌟")
        print("    SPIRITUAL STATIC BOTS - SYSTEM STATUS REPORT")
        print("    ZeroLight Orbit - 6993 Bots Monitoring System")
        print("🌟" + "="*58 + "🌟")
        
        status = self.get_system_status()
        
        # Health status dengan warna
        health_colors = {
            'EXCELLENT': '🟢',
            'GOOD': '🟡',
            'WARNING': '🟠',
            'CRITICAL': '🔴',
            'UNKNOWN': '⚪'
        }
        
        print(f"\n📊 SYSTEM OVERVIEW:")
        print(f"   Status: {health_colors.get(status.system_health, '⚪')} {status.system_health}")
        print(f"   Timestamp: {status.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Uptime: {status.uptime:.2f} seconds")
        print(f"   Spiritual Energy: {status.spiritual_energy:.1f}% {'🙏' if status.spiritual_energy > 80 else '⚡' if status.spiritual_energy > 50 else '⚠️'}")
        
        print(f"\n🤖 BOT STATISTICS:")
        print(f"   Total Bots: {status.total_bots:,}")
        print(f"   Active Bots: {status.active_bots:,} {'✅' if status.active_bots > 0 else '❌'}")
        print(f"   Idle Bots: {status.idle_bots:,}")
        print(f"   Error Bots: {status.error_bots:,} {'⚠️' if status.error_bots > 0 else '✅'}")
        
        print(f"\n💻 SYSTEM RESOURCES:")
        print(f"   CPU Usage: {status.cpu_usage:.1f}% {'🔥' if status.cpu_usage > 80 else '⚡' if status.cpu_usage > 50 else '✅'}")
        print(f"   Memory Usage: {status.memory_usage:.1f}% {'🔥' if status.memory_usage > 80 else '⚡' if status.memory_usage > 50 else '✅'}")
        print(f"   Disk Usage: {status.disk_usage:.1f}% {'🔥' if status.disk_usage > 80 else '⚡' if status.disk_usage > 50 else '✅'}")
        print(f"   Network Connections: {status.network_connections}")
        
        if detailed:
            print(f"\n📋 DETAILED INFORMATION:")
            
            # Database status
            db_status = self.check_database_status()
            print(f"   Database Status: {db_status.get('status', 'UNKNOWN')}")
            if 'category_counts' in db_status:
                print(f"   Bot Categories:")
                for category, count in db_status['category_counts'].items():
                    print(f"     - {category}: {count:,} bots")
            
            # Spiritual processes
            process_status = self.check_spiritual_processes()
            print(f"   Spiritual Processes: {process_status.get('count', 0)}")
            
            # Log files
            log_status = self.check_log_files()
            print(f"   Log Files:")
            for log_file, info in log_status.items():
                if info.get('exists'):
                    print(f"     - {log_file}: {info['size_mb']:.2f} MB (modified {info['age_hours']:.1f}h ago)")
                else:
                    print(f"     - {log_file}: Not found ❌")
        
        print("\n" + "🌟" + "="*58 + "🌟")
    
    def save_status_to_file(self, filename: Optional[str] = None):
        """Simpan status ke file JSON"""
        if filename is None:
            filename = f"spiritual_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        status = self.get_system_status()
        db_status = self.check_database_status()
        resource_status = self.check_system_resources()
        process_status = self.check_spiritual_processes()
        log_status = self.check_log_files()
        
        report = {
            'timestamp': status.timestamp.isoformat(),
            'system_status': {
                'health': status.system_health,
                'uptime': status.uptime,
                'spiritual_energy': status.spiritual_energy
            },
            'bot_statistics': {
                'total_bots': status.total_bots,
                'active_bots': status.active_bots,
                'idle_bots': status.idle_bots,
                'error_bots': status.error_bots
            },
            'system_resources': resource_status,
            'database_status': db_status,
            'spiritual_processes': process_status,
            'log_files': log_status
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"📄 Status report saved to: {filename}")
        return filename

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Spiritual System Status Checker')
    parser.add_argument('--detailed', '-d', action='store_true', help='Show detailed information')
    parser.add_argument('--save', '-s', action='store_true', help='Save status to JSON file')
    parser.add_argument('--watch', '-w', type=int, metavar='SECONDS', help='Watch mode - refresh every N seconds')
    parser.add_argument('--output', '-o', type=str, help='Output filename for saved report')
    
    args = parser.parse_args()
    
    checker = SpiritualSystemStatusChecker()
    
    try:
        if args.watch:
            print("👁️ Entering watch mode... Press Ctrl+C to exit")
            while True:
                os.system('cls' if os.name == 'nt' else 'clear')  # Clear screen
                checker.print_status_report(detailed=args.detailed)
                print(f"\n🔄 Refreshing every {args.watch} seconds... Press Ctrl+C to exit")
                time.sleep(args.watch)
        else:
            checker.print_status_report(detailed=args.detailed)
            
            if args.save:
                checker.save_status_to_file(args.output)
    
    except KeyboardInterrupt:
        print("\n🛑 Status checker stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())