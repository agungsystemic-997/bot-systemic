#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌟 ORBIT TRAFFIC AUDITOR
Ladang Berkah Digital - ZeroLight Orbit System
Audit Trafik Spiritual untuk Monitoring Bandwidth & Compliance
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

class SpiritualTrafficAuditor:
    """Auditor trafik spiritual untuk compliance dan monitoring"""
    
    def __init__(self, db_path: str = "./spiritual_orbit_system.db"):
        self.db_path = db_path
        self.audit_dir = Path("./traffic_audit_reports")
        self.audit_dir.mkdir(exist_ok=True)
        
        # Compliance thresholds
        self.compliance_limits = {
            'max_requests_per_hour_per_domain': 60,
            'max_requests_per_day_per_domain': 500,
            'max_concurrent_requests': 3,
            'min_interval_seconds': 300,  # 5 menit
            'max_bandwidth_mb_per_day': 100,
            'max_response_time_threshold': 10.0  # 10 detik
        }
        
        # Spiritual monitoring principles
        self.monitoring_principles = {
            'purpose': 'spiritual_health_monitoring',
            'method': 'minimal_impact_head_requests',
            'respect_robots_txt': True,
            'honor_rate_limits': True,
            'no_data_scraping': True,
            'transparent_user_agent': True,
            'graceful_error_handling': True
        }
    
    def init_audit_database(self):
        """Inisialisasi database audit"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabel audit trafik harian
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_traffic_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audit_date TEXT UNIQUE,
                total_requests INTEGER,
                total_domains INTEGER,
                total_bandwidth_mb REAL,
                avg_response_time REAL,
                compliance_score REAL,
                violations TEXT,
                recommendations TEXT,
                created_at TEXT
            )
        ''')
        
        # Tabel detail audit per domain
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS domain_traffic_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audit_date TEXT,
                domain TEXT,
                total_requests INTEGER,
                requests_per_hour REAL,
                avg_response_time REAL,
                total_bandwidth_mb REAL,
                ssl_usage_rate REAL,
                cache_hit_rate REAL,
                error_rate REAL,
                compliance_status TEXT,
                violations TEXT,
                created_at TEXT
            )
        ''')
        
        # Tabel compliance violations
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS compliance_violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                violation_date TEXT,
                domain TEXT,
                violation_type TEXT,
                violation_details TEXT,
                severity TEXT,
                resolved BOOLEAN DEFAULT 0,
                resolution_action TEXT,
                created_at TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def analyze_daily_traffic(self, audit_date: str = None) -> Dict:
        """Analisis trafik harian"""
        if not audit_date:
            audit_date = datetime.now().strftime('%Y-%m-%d')
        
        if not os.path.exists(self.db_path):
            return {'error': 'Database tidak ditemukan'}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Ambil semua transaksi hari ini
            cursor.execute('''
                SELECT tl.*, sa.original_name, sa.spiritual_name, sa.category
                FROM transaction_logs tl
                JOIN spiritual_assets sa ON tl.asset_id = sa.id
                WHERE DATE(tl.timestamp) = ?
                ORDER BY tl.timestamp
            ''', (audit_date,))
            
            transactions = []
            for row in cursor.fetchall():
                transactions.append({
                    'id': row[0],
                    'asset_id': row[1],
                    'transaction_type': row[2],
                    'status_code': row[3],
                    'response_time': row[4] or 0,
                    'is_alive': bool(row[5]),
                    'ssl_valid': bool(row[6]),
                    'error': row[7],
                    'timestamp': row[8],
                    'from_cache': bool(row[9]) if row[9] is not None else False,
                    'monitoring_metadata': row[10],
                    'domain': row[11],
                    'spiritual_name': row[12],
                    'category': row[13]
                })
            
            conn.close()
            
            if not transactions:
                return {'error': f'Tidak ada transaksi pada {audit_date}'}
            
            # Analisis per domain
            domain_stats = defaultdict(lambda: {
                'requests': 0,
                'successful_requests': 0,
                'ssl_requests': 0,
                'cached_requests': 0,
                'total_response_time': 0,
                'errors': 0,
                'timestamps': [],
                'spiritual_name': '',
                'category': ''
            })
            
            for tx in transactions:
                domain = tx['domain']
                domain_stats[domain]['requests'] += 1
                domain_stats[domain]['spiritual_name'] = tx['spiritual_name']
                domain_stats[domain]['category'] = tx['category']
                domain_stats[domain]['timestamps'].append(datetime.fromisoformat(tx['timestamp']))
                
                if tx['is_alive']:
                    domain_stats[domain]['successful_requests'] += 1
                
                if tx['ssl_valid']:
                    domain_stats[domain]['ssl_requests'] += 1
                
                if tx['from_cache']:
                    domain_stats[domain]['cached_requests'] += 1
                
                if tx['response_time']:
                    domain_stats[domain]['total_response_time'] += tx['response_time']
                
                if not tx['is_alive'] or tx['error']:
                    domain_stats[domain]['errors'] += 1
            
            # Hitung statistik dan compliance
            audit_results = {
                'audit_date': audit_date,
                'total_requests': len(transactions),
                'total_domains': len(domain_stats),
                'domain_analysis': {},
                'compliance_summary': {
                    'compliant_domains': 0,
                    'violations': [],
                    'overall_score': 0
                },
                'recommendations': []
            }
            
            total_compliance_score = 0
            
            for domain, stats in domain_stats.items():
                # Hitung metrics per domain
                success_rate = (stats['successful_requests'] / stats['requests'] * 100) if stats['requests'] > 0 else 0
                ssl_rate = (stats['ssl_requests'] / stats['requests'] * 100) if stats['requests'] > 0 else 0
                cache_hit_rate = (stats['cached_requests'] / stats['requests'] * 100) if stats['requests'] > 0 else 0
                error_rate = (stats['errors'] / stats['requests'] * 100) if stats['requests'] > 0 else 0
                avg_response_time = (stats['total_response_time'] / stats['requests']) if stats['requests'] > 0 else 0
                
                # Hitung requests per jam
                if stats['timestamps']:
                    time_span = (max(stats['timestamps']) - min(stats['timestamps'])).total_seconds() / 3600
                    requests_per_hour = stats['requests'] / max(time_span, 1)
                else:
                    requests_per_hour = 0
                
                # Estimasi bandwidth (asumsi HEAD request ~1KB per request)
                estimated_bandwidth_mb = (stats['requests'] * 1) / 1024  # KB to MB
                
                # Compliance check
                compliance_issues = []
                compliance_score = 100
                
                # Check rate limits
                if requests_per_hour > self.compliance_limits['max_requests_per_hour_per_domain']:
                    compliance_issues.append(f"Rate limit exceeded: {requests_per_hour:.1f} req/hour")
                    compliance_score -= 20
                
                if stats['requests'] > self.compliance_limits['max_requests_per_day_per_domain']:
                    compliance_issues.append(f"Daily limit exceeded: {stats['requests']} requests")
                    compliance_score -= 30
                
                if avg_response_time > self.compliance_limits['max_response_time_threshold']:
                    compliance_issues.append(f"High response time: {avg_response_time:.2f}s")
                    compliance_score -= 10
                
                # Check interval compliance
                if len(stats['timestamps']) > 1:
                    intervals = []
                    sorted_times = sorted(stats['timestamps'])
                    for i in range(1, len(sorted_times)):
                        interval = (sorted_times[i] - sorted_times[i-1]).total_seconds()
                        intervals.append(interval)
                    
                    min_interval = min(intervals) if intervals else 0
                    if min_interval < self.compliance_limits['min_interval_seconds']:
                        compliance_issues.append(f"Interval too short: {min_interval:.0f}s")
                        compliance_score -= 15
                
                compliance_status = 'COMPLIANT' if compliance_score >= 80 else ('WARNING' if compliance_score >= 60 else 'VIOLATION')
                
                if compliance_status == 'COMPLIANT':
                    audit_results['compliance_summary']['compliant_domains'] += 1
                else:
                    audit_results['compliance_summary']['violations'].extend([
                        {
                            'domain': domain,
                            'spiritual_name': stats['spiritual_name'],
                            'issues': compliance_issues,
                            'score': compliance_score,
                            'status': compliance_status
                        }
                    ])
                
                total_compliance_score += compliance_score
                
                # Simpan analisis domain
                audit_results['domain_analysis'][domain] = {
                    'spiritual_name': stats['spiritual_name'],
                    'category': stats['category'],
                    'total_requests': stats['requests'],
                    'requests_per_hour': round(requests_per_hour, 2),
                    'success_rate': round(success_rate, 1),
                    'ssl_rate': round(ssl_rate, 1),
                    'cache_hit_rate': round(cache_hit_rate, 1),
                    'error_rate': round(error_rate, 1),
                    'avg_response_time': round(avg_response_time, 3),
                    'estimated_bandwidth_mb': round(estimated_bandwidth_mb, 3),
                    'compliance_score': compliance_score,
                    'compliance_status': compliance_status,
                    'compliance_issues': compliance_issues
                }
            
            # Overall compliance score
            audit_results['compliance_summary']['overall_score'] = round(total_compliance_score / len(domain_stats), 1) if domain_stats else 0
            
            # Generate recommendations
            audit_results['recommendations'] = self.generate_recommendations(audit_results)
            
            return audit_results
            
        except Exception as e:
            conn.close()
            return {'error': f'Error analyzing traffic: {str(e)}'}
    
    def generate_recommendations(self, audit_results: Dict) -> List[str]:
        """Generate rekomendasi berdasarkan audit"""
        recommendations = []
        
        overall_score = audit_results['compliance_summary']['overall_score']
        violations = audit_results['compliance_summary']['violations']
        
        if overall_score < 70:
            recommendations.append("🚨 URGENT: Compliance score rendah, perlu optimisasi segera")
        
        if violations:
            recommendations.append(f"⚠️ Ditemukan {len(violations)} domain dengan pelanggaran")
            
            # Analisis jenis pelanggaran
            rate_limit_violations = [v for v in violations if any('Rate limit' in issue for issue in v['issues'])]
            if rate_limit_violations:
                recommendations.append("🐌 Perlambat interval monitoring untuk menghindari rate limiting")
            
            interval_violations = [v for v in violations if any('Interval too short' in issue for issue in v['issues'])]
            if interval_violations:
                recommendations.append("⏰ Tingkatkan interval minimum antar request (min 5 menit)")
            
            response_time_violations = [v for v in violations if any('High response time' in issue for issue in v['issues'])]
            if response_time_violations:
                recommendations.append("🚀 Optimisasi timeout dan connection handling")
        
        # Rekomendasi umum
        total_requests = audit_results['total_requests']
        if total_requests > 1000:
            recommendations.append("📊 Pertimbangkan mengurangi frekuensi monitoring untuk efisiensi")
        
        # Rekomendasi cache
        cache_domains = [d for d in audit_results['domain_analysis'].values() if d['cache_hit_rate'] < 30]
        if cache_domains:
            recommendations.append("📋 Tingkatkan cache duration untuk mengurangi request aktual")
        
        # Rekomendasi spiritual
        recommendations.extend([
            "🙏 Pastikan monitoring dilakukan dengan adab dan hormat",
            "✨ Gunakan HEAD request untuk meminimalkan impact",
            "🌸 Monitoring spiritual harus membawa berkah, bukan beban"
        ])
        
        return recommendations
    
    def save_audit_report(self, audit_results: Dict) -> str:
        """Simpan laporan audit"""
        audit_date = audit_results['audit_date']
        
        # Simpan ke database
        self.save_audit_to_database(audit_results)
        
        # Generate laporan text
        report_file = self.audit_dir / f"traffic_audit_{audit_date}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"""
🌟 LAPORAN AUDIT TRAFIK SPIRITUAL
Tanggal: {audit_date}
Generated: {datetime.now().isoformat()}

📊 RINGKASAN AUDIT:
- Total Requests: {audit_results['total_requests']}
- Total Domains: {audit_results['total_domains']}
- Compliance Score: {audit_results['compliance_summary']['overall_score']}/100
- Compliant Domains: {audit_results['compliance_summary']['compliant_domains']}/{audit_results['total_domains']}

🚦 STATUS COMPLIANCE:
""")
            
            if audit_results['compliance_summary']['violations']:
                f.write("⚠️ PELANGGARAN DITEMUKAN:\n")
                for violation in audit_results['compliance_summary']['violations']:
                    f.write(f"\n🔴 {violation['domain']} ({violation['spiritual_name']})\n")
                    f.write(f"   Score: {violation['score']}/100 - {violation['status']}\n")
                    for issue in violation['issues']:
                        f.write(f"   - {issue}\n")
            else:
                f.write("✅ Semua domain compliant!\n")
            
            f.write(f"\n📋 DETAIL ANALISIS PER DOMAIN:\n")
            for domain, analysis in audit_results['domain_analysis'].items():
                f.write(f"\n🌍 {domain} ({analysis['spiritual_name']})\n")
                f.write(f"   Kategori: {analysis['category']}\n")
                f.write(f"   Total Requests: {analysis['total_requests']}\n")
                f.write(f"   Requests/Hour: {analysis['requests_per_hour']}\n")
                f.write(f"   Success Rate: {analysis['success_rate']}%\n")
                f.write(f"   SSL Rate: {analysis['ssl_rate']}%\n")
                f.write(f"   Cache Hit Rate: {analysis['cache_hit_rate']}%\n")
                f.write(f"   Avg Response Time: {analysis['avg_response_time']}s\n")
                f.write(f"   Estimated Bandwidth: {analysis['estimated_bandwidth_mb']} MB\n")
                f.write(f"   Compliance: {analysis['compliance_status']} ({analysis['compliance_score']}/100)\n")
            
            f.write(f"\n💡 REKOMENDASI:\n")
            for i, rec in enumerate(audit_results['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
            
            f.write(f"\n🙏 PRINSIP MONITORING SPIRITUAL:\n")
            for principle, value in self.monitoring_principles.items():
                f.write(f"- {principle}: {value}\n")
            
            f.write(f"\n✨ Alhamdulillahi rabbil alamiin\n")
            f.write(f"🌸 Audit dilakukan dengan adab dan berkah\n")
        
        # Generate CSV report
        csv_file = self.audit_dir / f"traffic_audit_{audit_date}.csv"
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Domain', 'Spiritual_Name', 'Category', 'Total_Requests', 
                'Requests_Per_Hour', 'Success_Rate', 'SSL_Rate', 'Cache_Hit_Rate',
                'Error_Rate', 'Avg_Response_Time', 'Bandwidth_MB', 
                'Compliance_Score', 'Compliance_Status'
            ])
            
            for domain, analysis in audit_results['domain_analysis'].items():
                writer.writerow([
                    domain, analysis['spiritual_name'], analysis['category'],
                    analysis['total_requests'], analysis['requests_per_hour'],
                    analysis['success_rate'], analysis['ssl_rate'], analysis['cache_hit_rate'],
                    analysis['error_rate'], analysis['avg_response_time'], analysis['estimated_bandwidth_mb'],
                    analysis['compliance_score'], analysis['compliance_status']
                ])
        
        print(f"📄 Laporan audit tersimpan:")
        print(f"   📝 Text: {report_file}")
        print(f"   📊 CSV: {csv_file}")
        
        return str(report_file)
    
    def save_audit_to_database(self, audit_results: Dict):
        """Simpan hasil audit ke database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        audit_date = audit_results['audit_date']
        
        # Hitung total bandwidth
        total_bandwidth = sum(
            analysis['estimated_bandwidth_mb'] 
            for analysis in audit_results['domain_analysis'].values()
        )
        
        # Hitung rata-rata response time
        response_times = [
            analysis['avg_response_time'] 
            for analysis in audit_results['domain_analysis'].values()
            if analysis['avg_response_time'] > 0
        ]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Simpan audit harian
        cursor.execute('''
            INSERT OR REPLACE INTO daily_traffic_audit 
            (audit_date, total_requests, total_domains, total_bandwidth_mb, 
             avg_response_time, compliance_score, violations, recommendations, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            audit_date, audit_results['total_requests'], audit_results['total_domains'],
            total_bandwidth, avg_response_time, audit_results['compliance_summary']['overall_score'],
            json.dumps(audit_results['compliance_summary']['violations']),
            json.dumps(audit_results['recommendations']),
            datetime.now().isoformat()
        ))
        
        # Simpan detail per domain
        for domain, analysis in audit_results['domain_analysis'].items():
            cursor.execute('''
                INSERT INTO domain_traffic_audit 
                (audit_date, domain, total_requests, requests_per_hour, avg_response_time,
                 total_bandwidth_mb, ssl_usage_rate, cache_hit_rate, error_rate,
                 compliance_status, violations, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                audit_date, domain, analysis['total_requests'], analysis['requests_per_hour'],
                analysis['avg_response_time'], analysis['estimated_bandwidth_mb'],
                analysis['ssl_rate'], analysis['cache_hit_rate'], analysis['error_rate'],
                analysis['compliance_status'], json.dumps(analysis['compliance_issues']),
                datetime.now().isoformat()
            ))
        
        # Simpan violations
        for violation in audit_results['compliance_summary']['violations']:
            for issue in violation['issues']:
                severity = 'HIGH' if violation['score'] < 60 else ('MEDIUM' if violation['score'] < 80 else 'LOW')
                
                cursor.execute('''
                    INSERT INTO compliance_violations 
                    (violation_date, domain, violation_type, violation_details, severity, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    audit_date, violation['domain'], 'compliance_violation',
                    issue, severity, datetime.now().isoformat()
                ))
        
        conn.commit()
        conn.close()
    
    def generate_compliance_dashboard(self, days: int = 7) -> str:
        """Generate dashboard compliance untuk beberapa hari"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        conn = sqlite3.connect(self.db_path)
        
        # Ambil data audit harian
        df_daily = pd.read_sql_query('''
            SELECT audit_date, total_requests, total_domains, compliance_score, total_bandwidth_mb
            FROM daily_traffic_audit 
            WHERE audit_date >= ? AND audit_date <= ?
            ORDER BY audit_date
        ''', conn, params=[start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')])
        
        if df_daily.empty:
            conn.close()
            return "Tidak ada data audit untuk periode ini"
        
        # Create compliance dashboard
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('🌟 Spiritual Traffic Compliance Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Compliance Score Trend
        ax1.plot(df_daily['audit_date'], df_daily['compliance_score'], marker='o', linewidth=2, markersize=6)
        ax1.set_title('📊 Compliance Score Trend')
        ax1.set_ylabel('Compliance Score')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Total Requests per Day
        ax2.bar(df_daily['audit_date'], df_daily['total_requests'], color='skyblue', alpha=0.7)
        ax2.set_title('📈 Total Requests per Day')
        ax2.set_ylabel('Total Requests')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Bandwidth Usage
        ax3.bar(df_daily['audit_date'], df_daily['total_bandwidth_mb'], color='lightgreen', alpha=0.7)
        ax3.set_title('📊 Bandwidth Usage (MB)')
        ax3.set_ylabel('Bandwidth (MB)')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Domains Monitored
        ax4.bar(df_daily['audit_date'], df_daily['total_domains'], color='orange', alpha=0.7)
        ax4.set_title('🌍 Domains Monitored')
        ax4.set_ylabel('Total Domains')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Simpan dashboard
        dashboard_file = self.audit_dir / f"compliance_dashboard_{datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        conn.close()
        
        print(f"📊 Compliance dashboard: {dashboard_file}")
        return str(dashboard_file)

def main():
    """Fungsi utama traffic auditor"""
    print("🌟 ORBIT TRAFFIC AUDITOR")
    print("=" * 50)
    print("🔍 Memulai audit trafik spiritual...")
    print("🙏 Bismillahirrahmanirrahim")
    print()
    
    # Inisialisasi auditor
    auditor = SpiritualTrafficAuditor()
    auditor.init_audit_database()
    
    # Audit hari ini
    print("📊 TAHAP 1: Audit Trafik Hari Ini")
    audit_results = auditor.analyze_daily_traffic()
    
    if 'error' in audit_results:
        print(f"❌ Error: {audit_results['error']}")
        return
    
    # Tampilkan ringkasan
    print(f"📋 Total Requests: {audit_results['total_requests']}")
    print(f"🌍 Total Domains: {audit_results['total_domains']}")
    print(f"🎯 Compliance Score: {audit_results['compliance_summary']['overall_score']}/100")
    print(f"✅ Compliant Domains: {audit_results['compliance_summary']['compliant_domains']}")
    
    if audit_results['compliance_summary']['violations']:
        print(f"⚠️ Violations: {len(audit_results['compliance_summary']['violations'])}")
    
    # Simpan laporan
    print("\n📄 TAHAP 2: Generate Laporan Audit")
    report_file = auditor.save_audit_report(audit_results)
    
    # Generate dashboard
    print("\n📊 TAHAP 3: Generate Compliance Dashboard")
    try:
        dashboard_file = auditor.generate_compliance_dashboard(days=7)
        print(f"✅ Dashboard: {dashboard_file}")
    except Exception as e:
        print(f"⚠️ Error generating dashboard: {e}")
    
    # Tampilkan rekomendasi
    print("\n💡 REKOMENDASI:")
    for i, rec in enumerate(audit_results['recommendations'], 1):
        print(f"{i}. {rec}")
    
    print("\n✨ RINGKASAN AUDIT:")
    print("=" * 30)
    print(f"📄 Laporan: {report_file}")
    print(f"📁 Audit Directory: {auditor.audit_dir}")
    print(f"💾 Database: {auditor.db_path}")
    
    compliance_status = "EXCELLENT" if audit_results['compliance_summary']['overall_score'] >= 90 else \
                       "GOOD" if audit_results['compliance_summary']['overall_score'] >= 80 else \
                       "NEEDS_IMPROVEMENT" if audit_results['compliance_summary']['overall_score'] >= 60 else \
                       "CRITICAL"
    
    print(f"🎯 Status: {compliance_status}")
    print(f"🙏 Monitoring dengan adab dan berkah")
    print("✨ Alhamdulillahi rabbil alamiin")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🌸 Audit dihentikan dengan lembut...")
    except Exception as e:
        print(f"\n⚠️ Terjadi kesalahan: {e}")