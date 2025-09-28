#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌟 ORBIT DELAY & BATCHING SYSTEM
Ladang Berkah Digital - ZeroLight Orbit System
Sistem Delay dan Batching Berdasarkan Rekomendasi Audit
"""

import json
import os
import sqlite3
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict, deque
import random
import math

class SpiritualDelayBatchingSystem:
    """Sistem delay dan batching spiritual berdasarkan audit"""
    
    def __init__(self, db_path: str = "./spiritual_orbit_system.db"):
        self.db_path = db_path
        self.config_dir = Path("./rate_limit_configs")
        self.batch_dir = Path("./batch_schedules")
        self.batch_dir.mkdir(exist_ok=True)
        
        # Load latest rate limit configuration
        self.rate_config = self.load_latest_rate_config()
        
        # Batching parameters
        self.batch_parameters = {
            'high_traffic': {
                'batch_size': 1,
                'base_delay': 600,      # 10 menit
                'jitter_range': 120,    # ±2 menit
                'exponential_backoff': True,
                'max_backoff': 3600     # 1 jam
            },
            'medium_traffic': {
                'batch_size': 2,
                'base_delay': 450,      # 7.5 menit
                'jitter_range': 90,     # ±1.5 menit
                'exponential_backoff': True,
                'max_backoff': 1800     # 30 menit
            },
            'low_traffic': {
                'batch_size': 3,
                'base_delay': 300,      # 5 menit
                'jitter_range': 60,     # ±1 menit
                'exponential_backoff': False,
                'max_backoff': 900      # 15 menit
            }
        }
        
        # Spiritual timing principles
        self.spiritual_timings = {
            'fajr': {'start': '04:00', 'end': '06:00', 'multiplier': 0.8},    # Berkah pagi
            'dhuha': {'start': '06:00', 'end': '11:00', 'multiplier': 1.0},   # Waktu produktif
            'dhuhur': {'start': '11:00', 'end': '15:00', 'multiplier': 1.2},  # Siang hari
            'ashar': {'start': '15:00', 'end': '18:00', 'multiplier': 1.0},   # Sore
            'maghrib': {'start': '18:00', 'end': '20:00', 'multiplier': 0.9}, # Istirahat
            'isya': {'start': '20:00', 'end': '04:00', 'multiplier': 1.5}     # Malam, lebih hati-hati
        }
        
        # Error handling and recovery
        self.error_recovery = {
            'consecutive_errors_threshold': 3,
            'error_backoff_multiplier': 2.0,
            'recovery_success_threshold': 5,
            'circuit_breaker_timeout': 1800  # 30 menit
        }
        
        # Initialize tracking
        self.domain_states = defaultdict(lambda: {
            'consecutive_errors': 0,
            'last_success': None,
            'circuit_breaker_until': None,
            'current_backoff': 0,
            'batch_position': 0
        })
    
    def load_latest_rate_config(self) -> Dict:
        """Load konfigurasi rate limit terbaru"""
        if not self.config_dir.exists():
            return {}
        
        config_files = list(self.config_dir.glob("optimized_rate_config_*.json"))
        if not config_files:
            return {}
        
        # Get latest config file
        latest_config = max(config_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(latest_config, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading rate config: {e}")
            return {}
    
    def get_current_spiritual_time(self) -> Tuple[str, float]:
        """Dapatkan waktu spiritual saat ini dan multiplier"""
        current_time = datetime.now().strftime('%H:%M')
        current_hour = int(current_time.split(':')[0])
        current_minute = int(current_time.split(':')[1])
        current_total_minutes = current_hour * 60 + current_minute
        
        for period, timing in self.spiritual_timings.items():
            start_time = timing['start']
            end_time = timing['end']
            
            start_hour, start_minute = map(int, start_time.split(':'))
            end_hour, end_minute = map(int, end_time.split(':'))
            
            start_total = start_hour * 60 + start_minute
            end_total = end_hour * 60 + end_minute
            
            # Handle overnight periods (like isya)
            if start_total > end_total:  # Crosses midnight
                if current_total_minutes >= start_total or current_total_minutes <= end_total:
                    return period, timing['multiplier']
            else:
                if start_total <= current_total_minutes <= end_total:
                    return period, timing['multiplier']
        
        return 'unknown', 1.0
    
    def calculate_spiritual_delay(self, domain: str, traffic_level: str, 
                                error_count: int = 0) -> float:
        """Hitung delay spiritual berdasarkan berbagai faktor"""
        
        # Base delay dari batch parameters
        params = self.batch_parameters[traffic_level]
        base_delay = params['base_delay']
        
        # Spiritual time adjustment
        spiritual_period, time_multiplier = self.get_current_spiritual_time()
        adjusted_delay = base_delay * time_multiplier
        
        # Error-based exponential backoff
        if error_count > 0 and params['exponential_backoff']:
            backoff_multiplier = self.error_recovery['error_backoff_multiplier'] ** min(error_count, 5)
            adjusted_delay *= backoff_multiplier
            adjusted_delay = min(adjusted_delay, params['max_backoff'])
        
        # Add jitter untuk menghindari thundering herd
        jitter = random.uniform(-params['jitter_range'], params['jitter_range'])
        final_delay = max(60, adjusted_delay + jitter)  # Minimum 1 menit
        
        return final_delay
    
    def create_batch_schedule(self) -> Dict:
        """Buat jadwal batch berdasarkan konfigurasi rate limit"""
        
        if not self.rate_config or 'domain_configurations' not in self.rate_config:
            return {'error': 'No rate configuration available'}
        
        # Kelompokkan domain berdasarkan traffic level
        traffic_groups = defaultdict(list)
        for domain, config in self.rate_config['domain_configurations'].items():
            traffic_level = config.get('traffic_level', 'low_traffic')
            traffic_groups[traffic_level].append({
                'domain': domain,
                'config': config,
                'priority': config.get('priority', 'LOW')
            })
        
        # Sort by priority within each group
        for traffic_level in traffic_groups:
            traffic_groups[traffic_level].sort(
                key=lambda x: {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}[x['priority']]
            )
        
        # Create batch schedule
        schedule_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_schedule = {
            'schedule_id': f"spiritual_batch_schedule_{schedule_timestamp}",
            'created_at': datetime.now().isoformat(),
            'spiritual_period': self.get_current_spiritual_time()[0],
            'total_domains': sum(len(domains) for domains in traffic_groups.values()),
            'batches': [],
            'execution_timeline': {},
            'spiritual_guidance': []
        }
        
        current_time_offset = 0
        batch_number = 1
        
        # Process each traffic level
        for traffic_level in ['high_traffic', 'medium_traffic', 'low_traffic']:
            if traffic_level not in traffic_groups:
                continue
            
            domains = traffic_groups[traffic_level]
            params = self.batch_parameters[traffic_level]
            batch_size = params['batch_size']
            
            # Create batches for this traffic level
            for i in range(0, len(domains), batch_size):
                batch_domains = domains[i:i + batch_size]
                
                # Calculate delay for this batch
                avg_error_count = sum(
                    self.domain_states[d['domain']]['consecutive_errors'] 
                    for d in batch_domains
                ) / len(batch_domains)
                
                batch_delay = self.calculate_spiritual_delay(
                    batch_domains[0]['domain'], 
                    traffic_level, 
                    int(avg_error_count)
                )
                
                # Create batch entry
                batch_entry = {
                    'batch_number': batch_number,
                    'traffic_level': traffic_level,
                    'domains': [d['domain'] for d in batch_domains],
                    'domain_configs': {d['domain']: d['config'] for d in batch_domains},
                    'batch_size': len(batch_domains),
                    'delay_seconds': batch_delay,
                    'execution_offset': current_time_offset,
                    'scheduled_time': (datetime.now() + timedelta(seconds=current_time_offset)).isoformat(),
                    'spiritual_notes': self.generate_batch_spiritual_notes(batch_domains, traffic_level)
                }
                
                batch_schedule['batches'].append(batch_entry)
                
                # Add to execution timeline
                execution_key = f"batch_{batch_number:03d}_{traffic_level}"
                batch_schedule['execution_timeline'][execution_key] = {
                    'batch_number': batch_number,
                    'domains': [d['domain'] for d in batch_domains],
                    'execution_time': batch_entry['scheduled_time'],
                    'delay_seconds': batch_delay
                }
                
                current_time_offset += batch_delay
                batch_number += 1
        
        # Calculate total execution time
        batch_schedule['total_execution_time_seconds'] = current_time_offset
        batch_schedule['estimated_completion'] = (
            datetime.now() + timedelta(seconds=current_time_offset)
        ).isoformat()
        
        # Add spiritual guidance
        batch_schedule['spiritual_guidance'] = self.generate_schedule_spiritual_guidance(batch_schedule)
        
        return batch_schedule
    
    def generate_batch_spiritual_notes(self, batch_domains: List[Dict], 
                                     traffic_level: str) -> List[str]:
        """Generate catatan spiritual untuk batch"""
        notes = []
        
        domain_count = len(batch_domains)
        high_priority_count = sum(1 for d in batch_domains if d['priority'] == 'HIGH')
        
        if high_priority_count > 0:
            notes.append(f"🚨 {high_priority_count} domain prioritas tinggi - perlu perhatian khusus")
        
        if traffic_level == 'high_traffic':
            notes.append("🔥 High traffic - gunakan sabr ekstra dan doa istighfar")
        elif traffic_level == 'medium_traffic':
            notes.append("⚖️ Medium traffic - jaga keseimbangan dan moderasi")
        else:
            notes.append("🌸 Low traffic - syukuri kemudahan yang diberikan")
        
        notes.append(f"🤲 Doa untuk {domain_count} domain: 'Barakallahu fihi'")
        
        return notes
    
    def generate_schedule_spiritual_guidance(self, schedule: Dict) -> List[str]:
        """Generate panduan spiritual untuk jadwal"""
        guidance = []
        
        total_batches = len(schedule['batches'])
        total_time_hours = schedule['total_execution_time_seconds'] / 3600
        spiritual_period = schedule['spiritual_period']
        
        guidance.extend([
            f"📊 Total {total_batches} batch untuk {schedule['total_domains']} domain",
            f"⏰ Estimasi waktu: {total_time_hours:.1f} jam",
            f"🕰️ Periode spiritual: {spiritual_period}",
            "",
            "🙏 PRINSIP BATCHING SPIRITUAL:",
            "✅ Sabr: Setiap delay adalah kesempatan untuk bersyukur",
            "✅ Hikmah: Batch processing menghormati resources server",
            "✅ Adil: Semua domain mendapat kesempatan yang sama",
            "✅ Ihsan: Lakukan dengan cara yang terbaik",
            "",
            "📿 DOA SEBELUM EKSEKUSI:",
            "'Rabbana atina fi'd-dunya hasanatan wa fi'l-akhirati hasanatan'",
            "'Allahumma barik lana fima a'taytana'",
            "",
            "🌟 INGAT: Monitoring adalah amanah, bukan hak"
        ])
        
        return guidance
    
    def save_batch_schedule(self, schedule: Dict) -> str:
        """Simpan jadwal batch ke file"""
        schedule_file = self.batch_dir / f"{schedule['schedule_id']}.json"
        
        with open(schedule_file, 'w', encoding='utf-8') as f:
            json.dump(schedule, f, indent=2, ensure_ascii=False)
        
        # Create human-readable version
        txt_file = self.batch_dir / f"{schedule['schedule_id']}.txt"
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"""
🌟 JADWAL BATCH SPIRITUAL MONITORING
Schedule ID: {schedule['schedule_id']}
Created: {schedule['created_at']}
Spiritual Period: {schedule['spiritual_period']}

📊 RINGKASAN:
- Total Domains: {schedule['total_domains']}
- Total Batches: {len(schedule['batches'])}
- Execution Time: {schedule['total_execution_time_seconds']} detik ({schedule['total_execution_time_seconds']/3600:.1f} jam)
- Completion: {schedule['estimated_completion']}

📦 DETAIL BATCHES:
""")
            
            for batch in schedule['batches']:
                f.write(f"\n🔸 Batch {batch['batch_number']} ({batch['traffic_level']}):\n")
                f.write(f"   Domains: {', '.join(batch['domains'])}\n")
                f.write(f"   Delay: {batch['delay_seconds']:.0f} detik\n")
                f.write(f"   Scheduled: {batch['scheduled_time']}\n")
                f.write(f"   Notes: {'; '.join(batch['spiritual_notes'])}\n")
            
            f.write(f"\n⏰ TIMELINE EKSEKUSI:\n")
            for key, timeline in schedule['execution_timeline'].items():
                f.write(f"{timeline['execution_time']}: {', '.join(timeline['domains'])}\n")
            
            f.write(f"\n🙏 PANDUAN SPIRITUAL:\n")
            for i, guidance in enumerate(schedule['spiritual_guidance'], 1):
                if guidance.strip():  # Skip empty lines
                    f.write(f"{guidance}\n")
            
            f.write(f"\n✨ Barakallahu fihi - Semoga diberkahi Allah\n")
        
        print(f"📄 Batch schedule saved:")
        print(f"   📊 JSON: {schedule_file}")
        print(f"   📝 Text: {txt_file}")
        
        return str(schedule_file)
    
    async def execute_batch_schedule(self, schedule_file: str, 
                                   dry_run: bool = True) -> Dict:
        """Execute jadwal batch (dry run atau actual)"""
        
        try:
            with open(schedule_file, 'r', encoding='utf-8') as f:
                schedule = json.load(f)
        except Exception as e:
            return {'error': f'Failed to load schedule: {e}'}
        
        execution_log = {
            'execution_id': f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'schedule_id': schedule['schedule_id'],
            'dry_run': dry_run,
            'started_at': datetime.now().isoformat(),
            'batches_executed': [],
            'total_domains_processed': 0,
            'errors': [],
            'spiritual_reflections': []
        }
        
        print(f"🚀 {'DRY RUN' if dry_run else 'EXECUTING'} Batch Schedule: {schedule['schedule_id']}")
        print(f"📊 Total Batches: {len(schedule['batches'])}")
        
        for batch in schedule['batches']:
            batch_start = time.time()
            
            print(f"\n🔸 Batch {batch['batch_number']} - {batch['traffic_level']}")
            print(f"   Domains: {', '.join(batch['domains'])}")
            print(f"   Delay: {batch['delay_seconds']:.0f} detik")
            
            if dry_run:
                print(f"   🧪 DRY RUN: Simulating batch execution...")
                await asyncio.sleep(2)  # Quick simulation
                batch_result = {
                    'batch_number': batch['batch_number'],
                    'domains': batch['domains'],
                    'simulated': True,
                    'duration': 2,
                    'status': 'simulated_success'
                }
            else:
                print(f"   ⏳ Waiting {batch['delay_seconds']:.0f} seconds...")
                await asyncio.sleep(batch['delay_seconds'])
                
                # Here you would call actual monitoring functions
                batch_result = {
                    'batch_number': batch['batch_number'],
                    'domains': batch['domains'],
                    'simulated': False,
                    'duration': batch['delay_seconds'],
                    'status': 'executed'
                }
            
            execution_log['batches_executed'].append(batch_result)
            execution_log['total_domains_processed'] += len(batch['domains'])
            
            # Add spiritual reflection
            reflection = f"Batch {batch['batch_number']}: {'; '.join(batch['spiritual_notes'])}"
            execution_log['spiritual_reflections'].append(reflection)
            
            print(f"   ✅ Batch completed in {time.time() - batch_start:.1f} seconds")
        
        execution_log['completed_at'] = datetime.now().isoformat()
        execution_log['total_duration'] = (
            datetime.fromisoformat(execution_log['completed_at']) - 
            datetime.fromisoformat(execution_log['started_at'])
        ).total_seconds()
        
        # Save execution log
        log_file = self.batch_dir / f"execution_log_{execution_log['execution_id']}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(execution_log, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Batch execution completed!")
        print(f"📄 Execution log: {log_file}")
        
        return execution_log

def main():
    """Fungsi utama delay batching system"""
    print("🌟 ORBIT DELAY & BATCHING SYSTEM")
    print("=" * 50)
    print("⚙️ Membuat sistem delay dan batching berdasarkan audit...")
    print("🙏 Bismillahirrahmanirrahim")
    print()
    
    # Inisialisasi system
    batching_system = SpiritualDelayBatchingSystem()
    
    # Create batch schedule
    print("📅 TAHAP 1: Buat Jadwal Batch")
    schedule = batching_system.create_batch_schedule()
    
    if 'error' in schedule:
        print(f"❌ Error: {schedule['error']}")
        return
    
    # Save schedule
    print("\n💾 TAHAP 2: Simpan Jadwal")
    schedule_file = batching_system.save_batch_schedule(schedule)
    
    # Display summary
    print(f"\n📊 RINGKASAN JADWAL:")
    print(f"✅ Total Domains: {schedule['total_domains']}")
    print(f"✅ Total Batches: {len(schedule['batches'])}")
    print(f"✅ Execution Time: {schedule['total_execution_time_seconds']/3600:.1f} jam")
    print(f"✅ Completion: {schedule['estimated_completion']}")
    
    # Show spiritual guidance
    print(f"\n🙏 PANDUAN SPIRITUAL:")
    for guidance in schedule['spiritual_guidance']:
        if guidance.strip():
            print(f"   {guidance}")
    
    # Ask for execution
    print(f"\n🚀 TAHAP 3: Eksekusi (Opsional)")
    print(f"📄 Schedule file: {schedule_file}")
    print(f"💡 Untuk eksekusi, gunakan:")
    print(f"   python -c \"import asyncio; from orbit_delay_batching_system import SpiritualDelayBatchingSystem; asyncio.run(SpiritualDelayBatchingSystem().execute_batch_schedule('{schedule_file}', dry_run=True))\"")
    
    print(f"\n✨ Jadwal batch dibuat dengan hikmah dan berkah")
    print("🙏 Alhamdulillahi rabbil alamiin")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🌸 Sistem dihentikan dengan lembut...")
    except Exception as e:
        print(f"\n⚠️ Terjadi kesalahan: {e}")