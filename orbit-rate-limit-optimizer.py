#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌟 ORBIT RATE LIMIT OPTIMIZER
Ladang Berkah Digital - ZeroLight Orbit System
Optimisasi Rate Limiting & Delay Batching untuk Spiritual Monitoring
"""

import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time
import asyncio
from collections import defaultdict, deque
import math

class SpiritualRateLimitOptimizer:
    """Optimizer untuk rate limiting dan delay batching spiritual"""
    
    def __init__(self, db_path: str = "./spiritual_orbit_system.db"):
        self.db_path = db_path
        self.config_dir = Path("./rate_limit_configs")
        self.config_dir.mkdir(exist_ok=True)
        
        # Rate limiting configurations
        self.base_limits = {
            'min_interval_seconds': 300,  # 5 menit minimum
            'max_requests_per_hour': 12,  # 12 request per jam
            'max_requests_per_day': 200,  # 200 request per hari
            'batch_size': 3,              # 3 domain per batch
            'batch_delay_seconds': 60,    # 1 menit antar batch
            'error_backoff_multiplier': 2, # Exponential backoff
            'max_backoff_seconds': 3600,  # 1 jam maksimum backoff
            'compliance_buffer': 1.5      # 50% buffer untuk compliance
        }
        
        # Domain-specific adjustments
        self.domain_adjustments = {
            'high_traffic': {
                'multiplier': 2.0,
                'min_interval': 600,  # 10 menit
                'max_hourly': 6
            },
            'medium_traffic': {
                'multiplier': 1.5,
                'min_interval': 450,  # 7.5 menit
                'max_hourly': 8
            },
            'low_traffic': {
                'multiplier': 1.0,
                'min_interval': 300,  # 5 menit
                'max_hourly': 12
            }
        }
        
        # Spiritual principles for rate limiting
        self.spiritual_principles = {
            'respect': 'Honor platform resources with gratitude',
            'patience': 'Wait with sabr (patience) between requests',
            'moderation': 'Take only what is needed for monitoring',
            'gratitude': 'Each successful check is a blessing',
            'responsibility': 'Minimize impact on server resources'
        }
    
    def analyze_current_rates(self) -> Dict:
        """Analisis rate limiting saat ini"""
        if not os.path.exists(self.db_path):
            return {'error': 'Database tidak ditemukan'}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Ambil data transaksi 24 jam terakhir
            yesterday = (datetime.now() - timedelta(days=1)).isoformat()
            
            cursor.execute('''
                SELECT sa.original_name, sa.category, tl.timestamp, tl.status_code, tl.error_message
                FROM transaction_logs tl
                JOIN spiritual_assets sa ON tl.asset_id = sa.id
                WHERE tl.timestamp >= ?
                ORDER BY sa.original_name, tl.timestamp
            ''', (yesterday,))
            
            transactions = cursor.fetchall()
            conn.close()
            
            if not transactions:
                return {'error': 'Tidak ada data transaksi dalam 24 jam terakhir'}
            
            # Analisis per domain
            domain_analysis = defaultdict(lambda: {
                'requests': [],
                'errors': 0,
                'success': 0,
                'intervals': [],
                'category': 'unknown'
            })
            
            for row in transactions:
                domain, category, timestamp, status_code, error_message = row
                domain_analysis[domain]['category'] = category
                domain_analysis[domain]['requests'].append(datetime.fromisoformat(timestamp))
                
                if error_message or (status_code and status_code >= 400):
                    domain_analysis[domain]['errors'] += 1
                else:
                    domain_analysis[domain]['success'] += 1
            
            # Hitung statistik dan rekomendasi
            analysis_results = {
                'analysis_timestamp': datetime.now().isoformat(),
                'total_domains': len(domain_analysis),
                'domain_stats': {},
                'recommendations': {
                    'rate_adjustments': {},
                    'batching_strategy': {},
                    'spiritual_guidance': []
                }
            }
            
            for domain, stats in domain_analysis.items():
                requests = sorted(stats['requests'])
                
                # Hitung interval antar request
                intervals = []
                for i in range(1, len(requests)):
                    interval = (requests[i] - requests[i-1]).total_seconds()
                    intervals.append(interval)
                
                # Statistik domain
                total_requests = len(requests)
                error_rate = (stats['errors'] / total_requests * 100) if total_requests > 0 else 0
                success_rate = (stats['success'] / total_requests * 100) if total_requests > 0 else 0
                
                # Hitung requests per jam
                if len(requests) > 1:
                    time_span = (requests[-1] - requests[0]).total_seconds() / 3600
                    requests_per_hour = total_requests / max(time_span, 1)
                else:
                    requests_per_hour = 0
                
                # Interval statistics
                avg_interval = sum(intervals) / len(intervals) if intervals else 0
                min_interval = min(intervals) if intervals else 0
                
                # Klasifikasi traffic level
                traffic_level = self.classify_traffic_level(requests_per_hour, error_rate)
                
                domain_stats = {
                    'category': stats['category'],
                    'total_requests': total_requests,
                    'requests_per_hour': round(requests_per_hour, 2),
                    'error_rate': round(error_rate, 1),
                    'success_rate': round(success_rate, 1),
                    'avg_interval_seconds': round(avg_interval, 0),
                    'min_interval_seconds': round(min_interval, 0),
                    'traffic_level': traffic_level,
                    'compliance_status': self.check_compliance_status(requests_per_hour, min_interval, error_rate)
                }
                
                analysis_results['domain_stats'][domain] = domain_stats
                
                # Generate recommendations
                recommendations = self.generate_rate_recommendations(domain, domain_stats)
                analysis_results['recommendations']['rate_adjustments'][domain] = recommendations
            
            # Generate batching strategy
            analysis_results['recommendations']['batching_strategy'] = self.generate_batching_strategy(analysis_results['domain_stats'])
            
            # Spiritual guidance
            analysis_results['recommendations']['spiritual_guidance'] = self.generate_spiritual_guidance(analysis_results)
            
            return analysis_results
            
        except Exception as e:
            conn.close()
            return {'error': f'Error analyzing rates: {str(e)}'}
    
    def classify_traffic_level(self, requests_per_hour: float, error_rate: float) -> str:
        """Klasifikasi level traffic berdasarkan metrics"""
        if error_rate > 20 or requests_per_hour > 15:
            return 'high_traffic'
        elif error_rate > 10 or requests_per_hour > 10:
            return 'medium_traffic'
        else:
            return 'low_traffic'
    
    def check_compliance_status(self, requests_per_hour: float, min_interval: float, error_rate: float) -> str:
        """Check status compliance"""
        violations = []
        
        if requests_per_hour > self.base_limits['max_requests_per_hour']:
            violations.append('hourly_limit_exceeded')
        
        if min_interval < self.base_limits['min_interval_seconds']:
            violations.append('interval_too_short')
        
        if error_rate > 15:
            violations.append('high_error_rate')
        
        if not violations:
            return 'COMPLIANT'
        elif len(violations) == 1:
            return 'WARNING'
        else:
            return 'VIOLATION'
    
    def generate_rate_recommendations(self, domain: str, stats: Dict) -> Dict:
        """Generate rekomendasi rate limiting untuk domain"""
        traffic_level = stats['traffic_level']
        adjustments = self.domain_adjustments[traffic_level]
        
        # Hitung interval yang direkomendasikan
        current_interval = stats['avg_interval_seconds']
        recommended_interval = max(
            adjustments['min_interval'],
            current_interval * adjustments['multiplier']
        )
        
        # Hitung max requests per hour yang direkomendasikan
        recommended_max_hourly = min(
            adjustments['max_hourly'],
            int(3600 / recommended_interval)
        )
        
        recommendations = {
            'current_interval': current_interval,
            'recommended_interval': recommended_interval,
            'current_hourly_rate': stats['requests_per_hour'],
            'recommended_max_hourly': recommended_max_hourly,
            'traffic_level': traffic_level,
            'priority': 'HIGH' if stats['compliance_status'] == 'VIOLATION' else 'MEDIUM' if stats['compliance_status'] == 'WARNING' else 'LOW',
            'actions': []
        }
        
        # Generate specific actions
        if stats['compliance_status'] in ['VIOLATION', 'WARNING']:
            if stats['requests_per_hour'] > self.base_limits['max_requests_per_hour']:
                recommendations['actions'].append(f"Reduce hourly rate from {stats['requests_per_hour']:.1f} to {recommended_max_hourly}")
            
            if stats['min_interval_seconds'] < self.base_limits['min_interval_seconds']:
                recommendations['actions'].append(f"Increase minimum interval from {stats['min_interval_seconds']:.0f}s to {recommended_interval:.0f}s")
            
            if stats['error_rate'] > 10:
                recommendations['actions'].append(f"Implement exponential backoff for {stats['error_rate']:.1f}% error rate")
        
        if not recommendations['actions']:
            recommendations['actions'].append("Maintain current rate - compliant")
        
        return recommendations
    
    def generate_batching_strategy(self, domain_stats: Dict) -> Dict:
        """Generate strategi batching untuk optimisasi"""
        total_domains = len(domain_stats)
        
        # Kelompokkan domain berdasarkan traffic level
        traffic_groups = defaultdict(list)
        for domain, stats in domain_stats.items():
            traffic_groups[stats['traffic_level']].append(domain)
        
        # Hitung batch configuration
        batch_config = {
            'total_domains': total_domains,
            'batch_strategy': {},
            'execution_schedule': {},
            'estimated_completion_time': 0
        }
        
        current_time_offset = 0
        
        for traffic_level, domains in traffic_groups.items():
            adjustments = self.domain_adjustments[traffic_level]
            
            # Batch size berdasarkan traffic level
            if traffic_level == 'high_traffic':
                batch_size = 1  # Satu per satu untuk high traffic
                batch_delay = 300  # 5 menit antar batch
            elif traffic_level == 'medium_traffic':
                batch_size = 2
                batch_delay = 180  # 3 menit antar batch
            else:
                batch_size = 3
                batch_delay = 120  # 2 menit antar batch
            
            # Buat batches
            batches = [domains[i:i + batch_size] for i in range(0, len(domains), batch_size)]
            
            batch_config['batch_strategy'][traffic_level] = {
                'domains': domains,
                'batch_size': batch_size,
                'batch_count': len(batches),
                'batch_delay_seconds': batch_delay,
                'batches': batches
            }
            
            # Schedule execution
            for i, batch in enumerate(batches):
                execution_time = current_time_offset + (i * batch_delay)
                batch_config['execution_schedule'][f"{traffic_level}_batch_{i+1}"] = {
                    'domains': batch,
                    'execution_offset_seconds': execution_time,
                    'execution_time': (datetime.now() + timedelta(seconds=execution_time)).isoformat()
                }
            
            current_time_offset += len(batches) * batch_delay
        
        batch_config['estimated_completion_time'] = current_time_offset
        
        return batch_config
    
    def generate_spiritual_guidance(self, analysis_results: Dict) -> List[str]:
        """Generate panduan spiritual untuk rate limiting"""
        guidance = []
        
        total_domains = analysis_results['total_domains']
        violation_count = sum(1 for stats in analysis_results['domain_stats'].values() 
                            if stats['compliance_status'] == 'VIOLATION')
        warning_count = sum(1 for stats in analysis_results['domain_stats'].values() 
                          if stats['compliance_status'] == 'WARNING')
        
        # Spiritual guidance based on compliance
        if violation_count > 0:
            guidance.append(f"🚨 {violation_count} domain melanggar batas - perlukan taubat dan perbaikan segera")
            guidance.append("🙏 Istighfar: 'Astaghfirullah, kami telah melampaui batas yang wajar'")
        
        if warning_count > 0:
            guidance.append(f"⚠️ {warning_count} domain perlu perhatian - praktikkan sabr dan moderasi")
        
        # General spiritual principles
        guidance.extend([
            "🕰️ Sabr (Kesabaran): Tunggu dengan sabar antar request",
            "🤲 Syukur: Setiap response sukses adalah nikmat yang patut disyukuri",
            "⚖️ Adil: Berikan kesempatan yang sama untuk semua domain",
            "🌸 Ihsan: Lakukan monitoring dengan cara yang terbaik dan terindah",
            "🤝 Akhlaq: Hormati resources server seperti menghormati tetangga"
        ])
        
        # Specific recommendations
        if total_domains > 20:
            guidance.append("📊 Dengan banyak domain, praktikkan batch processing untuk efisiensi")
        
        guidance.append("🌟 Ingat: Monitoring spiritual adalah amanah, bukan hak")
        guidance.append("✨ Doa: 'Rabbana atina fi'd-dunya hasanatan wa fi'l-akhirati hasanatan'")
        
        return guidance
    
    def create_optimized_config(self, analysis_results: Dict) -> str:
        """Buat konfigurasi yang dioptimasi"""
        config_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_file = self.config_dir / f"optimized_rate_config_{config_timestamp}.json"
        
        # Build optimized configuration
        optimized_config = {
            'config_id': f"spiritual_rate_config_{config_timestamp}",
            'created_at': datetime.now().isoformat(),
            'base_configuration': self.base_limits,
            'domain_configurations': {},
            'batching_configuration': analysis_results['recommendations']['batching_strategy'],
            'spiritual_principles': self.spiritual_principles,
            'compliance_summary': {
                'total_domains': analysis_results['total_domains'],
                'compliant_domains': sum(1 for stats in analysis_results['domain_stats'].values() 
                                       if stats['compliance_status'] == 'COMPLIANT'),
                'warning_domains': sum(1 for stats in analysis_results['domain_stats'].values() 
                                     if stats['compliance_status'] == 'WARNING'),
                'violation_domains': sum(1 for stats in analysis_results['domain_stats'].values() 
                                       if stats['compliance_status'] == 'VIOLATION')
            }
        }
        
        # Add domain-specific configurations
        for domain, recommendations in analysis_results['recommendations']['rate_adjustments'].items():
            optimized_config['domain_configurations'][domain] = {
                'interval_seconds': recommendations['recommended_interval'],
                'max_requests_per_hour': recommendations['recommended_max_hourly'],
                'traffic_level': recommendations['traffic_level'],
                'priority': recommendations['priority'],
                'actions_required': recommendations['actions']
            }
        
        # Save configuration
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(optimized_config, f, indent=2, ensure_ascii=False)
        
        # Create human-readable version
        txt_file = self.config_dir / f"optimized_rate_config_{config_timestamp}.txt"
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"""
🌟 KONFIGURASI RATE LIMITING SPIRITUAL YANG DIOPTIMASI
Config ID: {optimized_config['config_id']}
Generated: {optimized_config['created_at']}

📊 RINGKASAN COMPLIANCE:
- Total Domains: {optimized_config['compliance_summary']['total_domains']}
- Compliant: {optimized_config['compliance_summary']['compliant_domains']}
- Warning: {optimized_config['compliance_summary']['warning_domains']}
- Violations: {optimized_config['compliance_summary']['violation_domains']}

⚙️ KONFIGURASI DASAR:
- Min Interval: {self.base_limits['min_interval_seconds']} detik
- Max Requests/Hour: {self.base_limits['max_requests_per_hour']}
- Max Requests/Day: {self.base_limits['max_requests_per_day']}
- Batch Size: {self.base_limits['batch_size']}
- Batch Delay: {self.base_limits['batch_delay_seconds']} detik

🌍 KONFIGURASI PER DOMAIN:
""")
            
            for domain, config in optimized_config['domain_configurations'].items():
                f.write(f"\n📍 {domain}:\n")
                f.write(f"   Interval: {config['interval_seconds']} detik\n")
                f.write(f"   Max/Hour: {config['max_requests_per_hour']}\n")
                f.write(f"   Traffic Level: {config['traffic_level']}\n")
                f.write(f"   Priority: {config['priority']}\n")
                f.write(f"   Actions: {', '.join(config['actions_required'])}\n")
            
            f.write(f"\n📦 STRATEGI BATCHING:\n")
            batching = optimized_config['batching_configuration']
            f.write(f"Total Completion Time: {batching['estimated_completion_time']} detik\n")
            
            for traffic_level, strategy in batching['batch_strategy'].items():
                f.write(f"\n🚦 {traffic_level.upper()}:\n")
                f.write(f"   Domains: {len(strategy['domains'])}\n")
                f.write(f"   Batch Size: {strategy['batch_size']}\n")
                f.write(f"   Batch Count: {strategy['batch_count']}\n")
                f.write(f"   Batch Delay: {strategy['batch_delay_seconds']} detik\n")
            
            f.write(f"\n🙏 PRINSIP SPIRITUAL:\n")
            for principle, description in self.spiritual_principles.items():
                f.write(f"- {principle.title()}: {description}\n")
            
            f.write(f"\n💡 PANDUAN SPIRITUAL:\n")
            for i, guidance in enumerate(analysis_results['recommendations']['spiritual_guidance'], 1):
                f.write(f"{i}. {guidance}\n")
            
            f.write(f"\n✨ Alhamdulillahi rabbil alamiin\n")
            f.write(f"🌸 Konfigurasi dibuat dengan berkah dan hikmah\n")
        
        print(f"📄 Optimized configuration saved:")
        print(f"   📊 JSON: {config_file}")
        print(f"   📝 Text: {txt_file}")
        
        return str(config_file)
    
    def apply_rate_limits(self, config_file: str) -> bool:
        """Apply rate limits dari konfigurasi"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            print(f"🔧 Applying rate limits dari {config['config_id']}")
            
            # Update database dengan konfigurasi baru
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create atau update rate_limit_config table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rate_limit_config (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain TEXT UNIQUE,
                    interval_seconds INTEGER,
                    max_requests_per_hour INTEGER,
                    traffic_level TEXT,
                    priority TEXT,
                    config_applied_at TEXT,
                    config_id TEXT
                )
            ''')
            
            # Apply configurations
            applied_count = 0
            for domain, domain_config in config['domain_configurations'].items():
                cursor.execute('''
                    INSERT OR REPLACE INTO rate_limit_config 
                    (domain, interval_seconds, max_requests_per_hour, traffic_level, 
                     priority, config_applied_at, config_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    domain, domain_config['interval_seconds'], 
                    domain_config['max_requests_per_hour'],
                    domain_config['traffic_level'], domain_config['priority'],
                    datetime.now().isoformat(), config['config_id']
                ))
                applied_count += 1
            
            conn.commit()
            conn.close()
            
            print(f"✅ Applied rate limits untuk {applied_count} domains")
            return True
            
        except Exception as e:
            print(f"❌ Error applying rate limits: {e}")
            return False

def main():
    """Fungsi utama rate limit optimizer"""
    print("🌟 ORBIT RATE LIMIT OPTIMIZER")
    print("=" * 50)
    print("⚙️ Mengoptimasi rate limiting untuk spiritual monitoring...")
    print("🙏 Bismillahirrahmanirrahim")
    print()
    
    # Inisialisasi optimizer
    optimizer = SpiritualRateLimitOptimizer()
    
    # Analisis rate limiting saat ini
    print("📊 TAHAP 1: Analisis Rate Limiting Saat Ini")
    analysis_results = optimizer.analyze_current_rates()
    
    if 'error' in analysis_results:
        print(f"❌ Error: {analysis_results['error']}")
        return
    
    # Tampilkan ringkasan
    print(f"📋 Total Domains: {analysis_results['total_domains']}")
    
    compliance_summary = {
        'COMPLIANT': 0,
        'WARNING': 0,
        'VIOLATION': 0
    }
    
    for stats in analysis_results['domain_stats'].values():
        compliance_summary[stats['compliance_status']] += 1
    
    print(f"✅ Compliant: {compliance_summary['COMPLIANT']}")
    print(f"⚠️ Warning: {compliance_summary['WARNING']}")
    print(f"🚨 Violations: {compliance_summary['VIOLATION']}")
    
    # Create optimized configuration
    print("\n⚙️ TAHAP 2: Buat Konfigurasi Optimal")
    config_file = optimizer.create_optimized_config(analysis_results)
    
    # Apply rate limits
    print("\n🔧 TAHAP 3: Apply Rate Limits")
    success = optimizer.apply_rate_limits(config_file)
    
    if success:
        print("✅ Rate limits berhasil diterapkan")
    else:
        print("❌ Gagal menerapkan rate limits")
    
    # Tampilkan panduan spiritual
    print("\n🙏 PANDUAN SPIRITUAL:")
    for guidance in analysis_results['recommendations']['spiritual_guidance']:
        print(f"   {guidance}")
    
    # Ringkasan
    print("\n✨ RINGKASAN OPTIMISASI:")
    print("=" * 30)
    print(f"📄 Config File: {config_file}")
    print(f"📁 Config Directory: {optimizer.config_dir}")
    print(f"💾 Database: {optimizer.db_path}")
    
    batching = analysis_results['recommendations']['batching_strategy']
    print(f"📦 Estimated Completion: {batching['estimated_completion_time']} detik")
    
    print(f"\n🎯 PRINSIP OPTIMISASI:")
    for principle, description in optimizer.spiritual_principles.items():
        print(f"✅ {principle.title()}: {description}")
    
    print(f"\n🙏 Optimisasi dilakukan dengan hikmah dan berkah")
    print("✨ Alhamdulillahi rabbil alamiin")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🌸 Optimisasi dihentikan dengan lembut...")
    except Exception as e:
        print(f"\n⚠️ Terjadi kesalahan: {e}")