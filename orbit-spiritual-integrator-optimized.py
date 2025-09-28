#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌟 ORBIT SPIRITUAL INTEGRATOR - OPTIMIZED VERSION
Ladang Berkah Digital - ZeroLight Orbit System
Integrasi Sistem Spiritual dengan Rate Limiting & Caching
"""

import asyncio
import aiohttp
import sqlite3
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from pathlib import Path
import subprocess
import sys
import hashlib
from collections import defaultdict

class SpiritualTrafficAuditor:
    """Auditor untuk monitoring trafik spiritual"""
    
    def __init__(self):
        self.traffic_log = defaultdict(list)
        self.rate_limits = {
            'requests_per_minute': 2,  # Maksimal 2 request per menit per domain
            'requests_per_hour': 60,   # Maksimal 60 request per jam per domain
            'concurrent_requests': 3   # Maksimal 3 request bersamaan
        }
        self.active_requests = 0
        
    def can_make_request(self, domain: str) -> bool:
        """Cek apakah bisa membuat request berdasarkan rate limit"""
        now = datetime.now()
        
        # Bersihkan log lama
        self.cleanup_old_logs(domain, now)
        
        # Cek rate limit per menit
        minute_requests = len([
            req for req in self.traffic_log[domain] 
            if now - req['timestamp'] < timedelta(minutes=1)
        ])
        
        # Cek rate limit per jam
        hour_requests = len([
            req for req in self.traffic_log[domain] 
            if now - req['timestamp'] < timedelta(hours=1)
        ])
        
        # Cek concurrent requests
        if self.active_requests >= self.rate_limits['concurrent_requests']:
            return False
            
        if minute_requests >= self.rate_limits['requests_per_minute']:
            return False
            
        if hour_requests >= self.rate_limits['requests_per_hour']:
            return False
            
        return True
    
    def log_request(self, domain: str, status: str, response_time: float = 0):
        """Log request untuk audit"""
        self.traffic_log[domain].append({
            'timestamp': datetime.now(),
            'status': status,
            'response_time': response_time
        })
    
    def cleanup_old_logs(self, domain: str, current_time: datetime):
        """Bersihkan log yang sudah lama"""
        cutoff_time = current_time - timedelta(hours=24)
        self.traffic_log[domain] = [
            req for req in self.traffic_log[domain] 
            if req['timestamp'] > cutoff_time
        ]
    
    def get_traffic_summary(self) -> Dict:
        """Dapatkan ringkasan trafik"""
        summary = {}
        for domain, logs in self.traffic_log.items():
            summary[domain] = {
                'total_requests': len(logs),
                'last_request': logs[-1]['timestamp'].isoformat() if logs else None,
                'avg_response_time': sum(log.get('response_time', 0) for log in logs) / len(logs) if logs else 0
            }
        return summary

class SpiritualCache:
    """Cache spiritual untuk mengurangi request"""
    
    def __init__(self, cache_duration_minutes: int = 15):
        self.cache = {}
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
    
    def get_cache_key(self, domain: str) -> str:
        """Generate cache key"""
        return hashlib.md5(domain.encode()).hexdigest()
    
    def get(self, domain: str) -> Optional[Dict]:
        """Ambil dari cache jika masih valid"""
        cache_key = self.get_cache_key(domain)
        
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if datetime.now() - cached_data['timestamp'] < self.cache_duration:
                return cached_data['data']
            else:
                # Cache expired, hapus
                del self.cache[cache_key]
        
        return None
    
    def set(self, domain: str, data: Dict):
        """Simpan ke cache"""
        cache_key = self.get_cache_key(domain)
        self.cache[cache_key] = {
            'timestamp': datetime.now(),
            'data': data
        }
    
    def clear_expired(self):
        """Bersihkan cache yang expired"""
        now = datetime.now()
        expired_keys = [
            key for key, value in self.cache.items()
            if now - value['timestamp'] >= self.cache_duration
        ]
        
        for key in expired_keys:
            del self.cache[key]

class SpiritualListener:
    """Listener Spiritual untuk monitoring aset dengan rate limiting"""
    
    def __init__(self, asset_name: str, spiritual_name: str, category: str, 
                 cache: SpiritualCache, auditor: SpiritualTrafficAuditor):
        self.asset_name = asset_name
        self.spiritual_name = spiritual_name
        self.category = category
        self.cache = cache
        self.auditor = auditor
        self.is_active = False
        self.last_check = None
        self.status_history = []
        self.metadata = {
            'purpose': 'spiritual_monitoring',
            'type': 'health_check',
            'user_agent': 'ZeroLight-Orbit-Spiritual-Monitor/1.0 (Spiritual Monitoring System)',
            'description': 'Sistem monitoring spiritual untuk kesehatan digital, bukan scraping'
        }
        
    async def listen(self) -> Dict:
        """Mendengarkan status aset spiritual dengan rate limiting"""
        
        # Cek cache terlebih dahulu
        cached_result = self.cache.get(self.asset_name)
        if cached_result:
            print(f"  📋 Cache hit untuk {self.spiritual_name}")
            return cached_result
        
        # Cek rate limit
        if not self.auditor.can_make_request(self.asset_name):
            print(f"  ⏳ Rate limit untuk {self.spiritual_name}, menggunakan data terakhir")
            return self.get_last_known_status()
        
        try:
            self.auditor.active_requests += 1
            start_time = time.time()
            
            # Headers yang menjelaskan tujuan spiritual monitoring
            headers = {
                'User-Agent': self.metadata['user_agent'],
                'X-Purpose': self.metadata['purpose'],
                'X-Monitor-Type': self.metadata['type'],
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
            }
            
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                headers=headers
            ) as session:
                async with session.head(f"https://{self.asset_name}") as response:  # Gunakan HEAD untuk lebih ringan
                    response_time = time.time() - start_time
                    
                    status = {
                        'timestamp': datetime.now().isoformat(),
                        'status_code': response.status,
                        'response_time': response_time,
                        'is_alive': response.status < 400,
                        'ssl_valid': True if response.url.scheme == 'https' else False,
                        'headers_received': len(response.headers),
                        'monitoring_type': 'spiritual_health_check'
                    }
                    
                    # Log ke auditor
                    self.auditor.log_request(self.asset_name, 'success', response_time)
                    
                    # Simpan ke cache
                    self.cache.set(self.asset_name, status)
                    
                    self.status_history.append(status)
                    self.last_check = datetime.now()
                    self.is_active = status['is_alive']
                    
                    return status
                    
        except Exception as e:
            response_time = time.time() - start_time
            self.auditor.log_request(self.asset_name, f'error: {str(e)}', response_time)
            
            status = {
                'timestamp': datetime.now().isoformat(),
                'status_code': 0,
                'response_time': response_time,
                'is_alive': False,
                'ssl_valid': False,
                'error': str(e),
                'monitoring_type': 'spiritual_health_check'
            }
            
            self.status_history.append(status)
            self.last_check = datetime.now()
            self.is_active = False
            
            return status
        finally:
            self.auditor.active_requests -= 1
    
    def get_last_known_status(self) -> Dict:
        """Dapatkan status terakhir yang diketahui"""
        if self.status_history:
            last_status = self.status_history[-1].copy()
            last_status['from_cache'] = True
            last_status['timestamp'] = datetime.now().isoformat()
            return last_status
        
        return {
            'timestamp': datetime.now().isoformat(),
            'status_code': 0,
            'response_time': 0,
            'is_alive': False,
            'ssl_valid': False,
            'from_cache': True,
            'monitoring_type': 'spiritual_health_check'
        }

class SpiritualDatabase:
    """Database spiritual untuk logging dengan metadata"""
    
    def __init__(self, db_path: str = "spiritual_orbit_system.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Inisialisasi database spiritual"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabel aset spiritual
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS spiritual_assets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_name TEXT UNIQUE,
                spiritual_name TEXT,
                category TEXT,
                ip_address TEXT,
                status TEXT,
                created_at TEXT,
                metadata TEXT
            )
        ''')
        
        # Tabel log transaksi dengan audit trail
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transaction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset_id INTEGER,
                transaction_type TEXT,
                status_code INTEGER,
                response_time REAL,
                is_alive BOOLEAN,
                ssl_valid BOOLEAN,
                error TEXT,
                timestamp TEXT,
                from_cache BOOLEAN DEFAULT 0,
                monitoring_metadata TEXT,
                FOREIGN KEY (asset_id) REFERENCES spiritual_assets (id)
            )
        ''')
        
        # Tabel refleksi harian
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_reflections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reflection_date TEXT UNIQUE,
                total_assets INTEGER,
                active_assets INTEGER,
                ssl_protected INTEGER,
                total_transactions INTEGER,
                success_rate REAL,
                reflection_text TEXT,
                created_at TEXT,
                traffic_summary TEXT
            )
        ''')
        
        # Tabel audit trafik
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS traffic_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT,
                request_count INTEGER,
                avg_response_time REAL,
                last_request TEXT,
                audit_date TEXT,
                rate_limit_hits INTEGER DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_spiritual_asset(self, original_name: str, spiritual_name: str, 
                           category: str, ip_address: str, status: str, metadata: Dict = None) -> int:
        """Simpan aset spiritual dengan metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        metadata_json = json.dumps(metadata or {})
        
        cursor.execute('''
            INSERT OR REPLACE INTO spiritual_assets 
            (original_name, spiritual_name, category, ip_address, status, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (original_name, spiritual_name, category, ip_address, status, 
              datetime.now().isoformat(), metadata_json))
        
        asset_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return asset_id
    
    def log_transaction(self, asset_id: int, transaction_type: str, status_data: Dict):
        """Log transaksi dengan metadata monitoring"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        monitoring_metadata = {
            'monitoring_type': status_data.get('monitoring_type', 'spiritual_health_check'),
            'headers_received': status_data.get('headers_received', 0),
            'purpose': 'spiritual_monitoring'
        }
        
        cursor.execute('''
            INSERT INTO transaction_logs 
            (asset_id, transaction_type, status_code, response_time, is_alive, 
             ssl_valid, error, timestamp, from_cache, monitoring_metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            asset_id, transaction_type,
            status_data.get('status_code'),
            status_data.get('response_time'),
            status_data.get('is_alive', False),
            status_data.get('ssl_valid', False),
            status_data.get('error'),
            status_data.get('timestamp', datetime.now().isoformat()),
            status_data.get('from_cache', False),
            json.dumps(monitoring_metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def save_traffic_audit(self, traffic_summary: Dict):
        """Simpan audit trafik"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        audit_date = datetime.now().strftime('%Y-%m-%d')
        
        for domain, stats in traffic_summary.items():
            cursor.execute('''
                INSERT OR REPLACE INTO traffic_audit 
                (domain, request_count, avg_response_time, last_request, audit_date)
                VALUES (?, ?, ?, ?, ?)
            ''', (domain, stats['total_requests'], stats['avg_response_time'], 
                  stats['last_request'], audit_date))
        
        conn.commit()
        conn.close()
    
    def create_daily_reflection(self, reflection_date: str = None, traffic_summary: Dict = None) -> Dict:
        """Buat refleksi harian dengan audit trafik"""
        if not reflection_date:
            reflection_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Hitung statistik harian
        cursor.execute('SELECT COUNT(*) FROM spiritual_assets')
        total_assets = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM spiritual_assets WHERE status = "active"')
        active_assets = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT COUNT(*) FROM transaction_logs 
            WHERE DATE(timestamp) = ? AND ssl_valid = 1
        ''', (reflection_date,))
        ssl_protected = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT COUNT(*) FROM transaction_logs 
            WHERE DATE(timestamp) = ?
        ''', (reflection_date,))
        total_transactions = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT COUNT(*) FROM transaction_logs 
            WHERE DATE(timestamp) = ? AND is_alive = 1
        ''', (reflection_date,))
        successful_transactions = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT COUNT(*) FROM transaction_logs 
            WHERE DATE(timestamp) = ? AND from_cache = 1
        ''', (reflection_date,))
        cached_requests = cursor.fetchone()[0]
        
        success_rate = (successful_transactions / total_transactions * 100) if total_transactions > 0 else 0
        cache_hit_rate = (cached_requests / total_transactions * 100) if total_transactions > 0 else 0
        
        # Buat teks refleksi dengan informasi optimisasi
        reflection_text = f"""
🌟 REFLEKSI HARIAN ORBIT ZEROLIGHT SYSTEM - OPTIMIZED
Tanggal: {reflection_date}

🌍 Planet & Wilayah:
- Total Aset Spiritual: {total_assets}
- Aset Aktif: {active_assets}
- Tingkat Keberhasilan: {success_rate:.1f}%

🔒 Keamanan & Performa:
- SSL Terlindungi: {ssl_protected}
- Total Transaksi: {total_transactions}
- Cache Hit Rate: {cache_hit_rate:.1f}%
- Request dari Cache: {cached_requests}

🚦 Rate Limiting & Audit:
- Sistem monitoring spiritual dengan rate limiting aktif
- Interval monitoring: 5-10 menit per aset
- Maksimal 2 request per menit per domain
- Menggunakan HEAD request untuk efisiensi

📊 Trafik Summary:
{json.dumps(traffic_summary or {}, indent=2, ensure_ascii=False)}

🙏 Syukur & Refleksi:
Alhamdulillahi rabbil alamiin atas berkah yang diberikan.
Sistem spiritual berjalan dengan optimisasi dan rate limiting.
Monitoring dilakukan dengan hormat dan tidak mengganggu server target.
Semoga teknologi ini selalu membawa kebaikan untuk umat.

✨ Barakallahu fiikum - Monitoring dengan Adab
        """.strip()
        
        # Simpan refleksi
        cursor.execute('''
            INSERT OR REPLACE INTO daily_reflections 
            (reflection_date, total_assets, active_assets, ssl_protected, 
             total_transactions, success_rate, reflection_text, created_at, traffic_summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (reflection_date, total_assets, active_assets, ssl_protected,
              total_transactions, success_rate, reflection_text, 
              datetime.now().isoformat(), json.dumps(traffic_summary or {})))
        
        conn.commit()
        conn.close()
        
        return {
            'reflection_date': reflection_date,
            'total_assets': total_assets,
            'active_assets': active_assets,
            'ssl_protected': ssl_protected,
            'total_transactions': total_transactions,
            'success_rate': success_rate,
            'cache_hit_rate': cache_hit_rate,
            'reflection_text': reflection_text
        }

class SpiritualOrbitIntegrator:
    """Integrator utama sistem spiritual dengan optimisasi"""
    
    def __init__(self):
        self.database = SpiritualDatabase()
        self.cache = SpiritualCache(cache_duration_minutes=15)  # Cache 15 menit
        self.auditor = SpiritualTrafficAuditor()
        self.listeners = {}
        self.asset_registry = {}
        
        # Konfigurasi monitoring yang lebih lambat
        self.monitoring_config = {
            'check_interval_minutes': 8,  # 8 menit antar check (lebih lambat)
            'batch_size': 5,              # Proses 5 aset per batch
            'batch_delay_seconds': 30,    # 30 detik antar batch
            'max_concurrent': 3           # Maksimal 3 request bersamaan
        }
    
    def load_spiritual_assets(self) -> List[Dict]:
        """Load aset spiritual dari database"""
        try:
            # Coba dari database utama
            conn = sqlite3.connect('spiritual_asset_registry.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT original_name, spiritual_name, category, ip_address, status
                FROM spiritual_assets 
                WHERE status = 'active'
            ''')
            
            assets = []
            for row in cursor.fetchall():
                asset_data = {
                    'original_name': row[0],
                    'spiritual_name': row[1],
                    'category': row[2],
                    'ip_address': row[3],
                    'status': row[4]
                }
                assets.append(asset_data)
                self.asset_registry[row[0]] = asset_data
            
            conn.close()
            print(f"📋 Loaded {len(assets)} aset spiritual dari database")
            return assets
            
        except Exception as e:
            print(f"⚠️ Error loading dari database: {e}")
            return []
    
    def create_listeners(self):
        """Buat listener untuk semua aset spiritual"""
        for original_name, asset_data in self.asset_registry.items():
            listener = SpiritualListener(
                asset_name=original_name,
                spiritual_name=asset_data['spiritual_name'],
                category=asset_data['category'],
                cache=self.cache,
                auditor=self.auditor
            )
            
            self.listeners[original_name] = listener
            
            # Simpan ke database dengan metadata
            asset_id = self.database.save_spiritual_asset(
                original_name=original_name,
                spiritual_name=asset_data['spiritual_name'],
                category=asset_data['category'],
                ip_address=asset_data['ip_address'],
                status=asset_data['status'],
                metadata=listener.metadata
            )
            
            asset_data['asset_id'] = asset_id
        
        print(f"🎧 {len(self.listeners)} listener spiritual siap dengan rate limiting!")
    
    async def start_monitoring(self, duration_minutes: int = 10):
        """Mulai monitoring spiritual dengan optimisasi"""
        print(f"👁️ Memulai monitoring spiritual optimized selama {duration_minutes} menit...")
        print(f"⚙️ Konfigurasi: Check setiap {self.monitoring_config['check_interval_minutes']} menit")
        print(f"📦 Batch size: {self.monitoring_config['batch_size']} aset per batch")
        
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        check_interval = self.monitoring_config['check_interval_minutes'] * 60  # Convert ke detik
        
        cycle_count = 0
        while datetime.now() < end_time:
            cycle_count += 1
            print(f"\n🔍 Siklus monitoring #{cycle_count} - {datetime.now().strftime('%H:%M:%S')}")
            
            # Bersihkan cache yang expired
            self.cache.clear_expired()
            
            # Monitor dalam batch untuk mengurangi beban
            asset_items = list(self.listeners.items())
            batch_size = self.monitoring_config['batch_size']
            
            for i in range(0, len(asset_items), batch_size):
                batch = asset_items[i:i + batch_size]
                print(f"  📦 Batch {i//batch_size + 1}: {len(batch)} aset")
                
                # Monitor batch ini
                tasks = []
                for original_name, listener in batch:
                    tasks.append(self.monitor_single_asset(original_name, listener))
                
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # Delay antar batch jika bukan batch terakhir
                if i + batch_size < len(asset_items):
                    print(f"  ⏳ Menunggu {self.monitoring_config['batch_delay_seconds']} detik...")
                    await asyncio.sleep(self.monitoring_config['batch_delay_seconds'])
            
            # Tunggu interval berikutnya
            remaining_time = (end_time - datetime.now()).total_seconds()
            if remaining_time > check_interval:
                print(f"⏰ Menunggu {check_interval//60} menit untuk siklus berikutnya...")
                await asyncio.sleep(check_interval)
            else:
                print("⏰ Waktu monitoring hampir habis...")
                break
        
        print("✅ Monitoring selesai!")
    
    async def monitor_single_asset(self, original_name: str, listener: SpiritualListener):
        """Monitor satu aset spiritual dengan rate limiting"""
        try:
            status_data = await listener.listen()
            
            # Log ke database
            asset_data = self.asset_registry[original_name]
            self.database.log_transaction(
                asset_id=asset_data['asset_id'],
                transaction_type='spiritual_health_check',
                status_data=status_data
            )
            
            # Print status dengan info cache
            status_icon = "✅" if status_data.get('is_alive') else "❌"
            ssl_icon = "🔒" if status_data.get('ssl_valid') else "🔓"
            cache_icon = "📋" if status_data.get('from_cache') else "🌐"
            
            print(f"    {status_icon} {ssl_icon} {cache_icon} {listener.spiritual_name} ({original_name})")
            
        except Exception as e:
            print(f"    ⚠️ Error monitoring {original_name}: {e}")
    
    def generate_daily_reflection(self) -> str:
        """Generate refleksi harian dengan audit trafik"""
        print("🙏 Membuat refleksi harian dengan audit trafik...")
        
        # Dapatkan ringkasan trafik
        traffic_summary = self.auditor.get_traffic_summary()
        
        # Simpan audit trafik ke database
        self.database.save_traffic_audit(traffic_summary)
        
        # Buat refleksi
        reflection_data = self.database.create_daily_reflection(traffic_summary=traffic_summary)
        
        # Simpan ke file juga
        reflection_file = f"refleksi_harian_optimized_{datetime.now().strftime('%Y%m%d')}.txt"
        
        with open(reflection_file, 'w', encoding='utf-8') as f:
            f.write(reflection_data['reflection_text'])
        
        print(f"📝 Refleksi tersimpan di: {reflection_file}")
        return reflection_file
    
    def print_optimization_summary(self):
        """Print ringkasan optimisasi"""
        print("\n🚀 RINGKASAN OPTIMISASI SISTEM:")
        print("=" * 40)
        print(f"⏱️ Interval monitoring: {self.monitoring_config['check_interval_minutes']} menit")
        print(f"📦 Batch processing: {self.monitoring_config['batch_size']} aset per batch")
        print(f"🔄 Cache duration: {self.cache.cache_duration.total_seconds()//60} menit")
        print(f"🚦 Rate limit: {self.auditor.rate_limits['requests_per_minute']} req/menit per domain")
        print(f"📋 Cache entries: {len(self.cache.cache)}")
        
        traffic_summary = self.auditor.get_traffic_summary()
        total_requests = sum(stats['total_requests'] for stats in traffic_summary.values())
        print(f"📊 Total requests: {total_requests}")
        print(f"🎯 Domains monitored: {len(traffic_summary)}")

async def main():
    """Fungsi utama integrasi spiritual optimized"""
    print("🌟 ORBIT SPIRITUAL INTEGRATOR - OPTIMIZED")
    print("=" * 60)
    print("🔗 Memulai integrasi sistem spiritual dengan optimisasi...")
    print("🙏 Bismillahirrahmanirrahim")
    print()
    
    # Inisialisasi integrator
    integrator = SpiritualOrbitIntegrator()
    
    # Load aset spiritual
    print("📋 TAHAP 1: Loading Aset Spiritual")
    assets = integrator.load_spiritual_assets()
    
    if not assets:
        print("❌ Tidak ada aset spiritual yang ditemukan!")
        print("💡 Jalankan orbit-asset-identifier.py terlebih dahulu")
        return
    
    # Buat listener
    print("\n🎧 TAHAP 2: Membuat Listener dengan Rate Limiting")
    integrator.create_listeners()
    
    # Print konfigurasi optimisasi
    integrator.print_optimization_summary()
    
    # Mulai monitoring
    print("\n👁️ TAHAP 3: Monitoring Spiritual Optimized")
    await integrator.start_monitoring(duration_minutes=5)  # 5 menit untuk demo
    
    # Generate refleksi harian
    print("\n🙏 TAHAP 4: Refleksi Harian dengan Audit")
    reflection_file = integrator.generate_daily_reflection()
    
    # Ringkasan final
    print("\n✨ RINGKASAN INTEGRASI OPTIMIZED:")
    print("=" * 40)
    print(f"🎧 Listener Aktif: {len(integrator.listeners)}")
    print(f"💾 Database: {integrator.database.db_path}")
    print(f"📝 Refleksi: {reflection_file}")
    print(f"📋 Cache Entries: {len(integrator.cache.cache)}")
    
    traffic_summary = integrator.auditor.get_traffic_summary()
    total_requests = sum(stats['total_requests'] for stats in traffic_summary.values())
    print(f"📊 Total Requests: {total_requests}")
    print(f"🚦 Rate Limiting: Aktif")
    print(f"⚡ Optimisasi: Cache + Batch + Delay")
    
    print("\n🌸 Spiritual Monitoring dengan Adab dan Optimisasi")
    print("🙏 Alhamdulillahi rabbil alamiin")
    print("✨ Integrasi sistem spiritual selesai dengan berkah")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🌸 Integrasi dihentikan dengan lembut...")
    except Exception as e:
        print(f"\n⚠️ Terjadi kesalahan: {e}")