#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌟 ORBIT SPIRITUAL INTEGRATOR
Ladang Berkah Digital - ZeroLight Orbit System
Integrasi Sistem Spiritual dengan Listener & Database
"""

import asyncio
import aiohttp
import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from pathlib import Path
import subprocess
import sys

class SpiritualListener:
    """Listener Spiritual untuk monitoring aset"""
    
    def __init__(self, asset_name: str, spiritual_name: str, category: str):
        self.asset_name = asset_name
        self.spiritual_name = spiritual_name
        self.category = category
        self.is_active = False
        self.last_check = None
        self.status_history = []
        
    async def listen(self) -> Dict:
        """Mendengarkan status aset spiritual"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"https://{self.asset_name}") as response:
                    status = {
                        'timestamp': datetime.now().isoformat(),
                        'status_code': response.status,
                        'response_time': response.headers.get('X-Response-Time', 'unknown'),
                        'is_alive': response.status < 400,
                        'ssl_valid': True if response.url.scheme == 'https' else False
                    }
                    
                    self.status_history.append(status)
                    self.last_check = datetime.now()
                    self.is_active = status['is_alive']
                    
                    return status
        except Exception as e:
            error_status = {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'is_alive': False,
                'ssl_valid': False
            }
            self.status_history.append(error_status)
            self.is_active = False
            return error_status

class SpiritualDatabase:
    """Database Spiritual untuk logging transaksi dan refleksi"""
    
    def __init__(self, db_path: str = "./spiritual_orbit_system.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Inisialisasi database spiritual"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabel untuk aset spiritual
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS spiritual_assets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_name TEXT UNIQUE,
                spiritual_name TEXT,
                category TEXT,
                ip_address TEXT,
                status TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        ''')
        
        # Tabel untuk log transaksi
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transaction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset_id INTEGER,
                transaction_type TEXT,
                status_code INTEGER,
                response_time TEXT,
                is_alive BOOLEAN,
                ssl_valid BOOLEAN,
                error_message TEXT,
                timestamp TEXT,
                FOREIGN KEY (asset_id) REFERENCES spiritual_assets (id)
            )
        ''')
        
        # Tabel untuk refleksi harian
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
                created_at TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def insert_spiritual_asset(self, original_name: str, spiritual_name: str, 
                             category: str, ip_address: str, status: str) -> int:
        """Insert aset spiritual ke database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO spiritual_assets 
            (original_name, spiritual_name, category, ip_address, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (original_name, spiritual_name, category, ip_address, status, 
              datetime.now().isoformat(), datetime.now().isoformat()))
        
        asset_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return asset_id
    
    def log_transaction(self, asset_id: int, transaction_type: str, 
                       status_data: Dict):
        """Log transaksi ke database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO transaction_logs 
            (asset_id, transaction_type, status_code, response_time, is_alive, 
             ssl_valid, error_message, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            asset_id, transaction_type,
            status_data.get('status_code'),
            status_data.get('response_time'),
            status_data.get('is_alive', False),
            status_data.get('ssl_valid', False),
            status_data.get('error'),
            status_data.get('timestamp', datetime.now().isoformat())
        ))
        
        conn.commit()
        conn.close()
    
    def create_daily_reflection(self, reflection_date: str = None) -> Dict:
        """Buat refleksi harian"""
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
        
        success_rate = (successful_transactions / total_transactions * 100) if total_transactions > 0 else 0
        
        # Buat teks refleksi
        reflection_text = f"""
🌟 REFLEKSI HARIAN ORBIT ZEROLIGHT SYSTEM
Tanggal: {reflection_date}

🌍 Planet & Wilayah:
- Total Aset Spiritual: {total_assets}
- Aset Aktif: {active_assets}
- Tingkat Keberhasilan: {success_rate:.1f}%

🔒 Keamanan:
- SSL Terlindungi: {ssl_protected}
- Total Transaksi: {total_transactions}

🙏 Syukur & Refleksi:
Alhamdulillahi rabbil alamiin atas berkah yang diberikan.
Sistem spiritual berjalan dengan lancar dan memberikan manfaat.
Semoga teknologi ini selalu membawa kebaikan untuk umat.

✨ Barakallahu fiikum
        """.strip()
        
        # Simpan refleksi
        cursor.execute('''
            INSERT OR REPLACE INTO daily_reflections 
            (reflection_date, total_assets, active_assets, ssl_protected, 
             total_transactions, success_rate, reflection_text, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (reflection_date, total_assets, active_assets, ssl_protected,
              total_transactions, success_rate, reflection_text, 
              datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        return {
            'reflection_date': reflection_date,
            'total_assets': total_assets,
            'active_assets': active_assets,
            'ssl_protected': ssl_protected,
            'total_transactions': total_transactions,
            'success_rate': success_rate,
            'reflection_text': reflection_text
        }

class SpiritualOrbitIntegrator:
    """Integrator utama sistem spiritual"""
    
    def __init__(self):
        self.database = SpiritualDatabase()
        self.listeners = {}
        self.asset_registry = {}
        
    def load_spiritual_assets(self) -> Dict:
        """Load aset spiritual dari registry"""
        registry_file = "./spiritual_asset_registry.db"
        
        if not os.path.exists(registry_file):
            print("⚠️ Registry aset spiritual tidak ditemukan!")
            return {}
        
        conn = sqlite3.connect(registry_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT original_name, spiritual_name, category, ip_address, status 
            FROM spiritual_assets WHERE status = 'active'
        ''')
        
        assets = {}
        for row in cursor.fetchall():
            original_name, spiritual_name, category, ip_address, status = row
            assets[original_name] = {
                'spiritual_name': spiritual_name,
                'category': category,
                'ip_address': ip_address,
                'status': status
            }
        
        conn.close()
        self.asset_registry = assets
        
        print(f"📋 Loaded {len(assets)} aset spiritual aktif")
        return assets
    
    def create_listeners(self):
        """Buat listener untuk semua aset aktif"""
        print("🎧 Membuat listener spiritual...")
        
        for original_name, asset_data in self.asset_registry.items():
            listener = SpiritualListener(
                asset_name=original_name,
                spiritual_name=asset_data['spiritual_name'],
                category=asset_data['category']
            )
            
            self.listeners[original_name] = listener
            
            # Insert ke database utama
            asset_id = self.database.insert_spiritual_asset(
                original_name=original_name,
                spiritual_name=asset_data['spiritual_name'],
                category=asset_data['category'],
                ip_address=asset_data['ip_address'],
                status=asset_data['status']
            )
            
            asset_data['asset_id'] = asset_id
        
        print(f"🎧 {len(self.listeners)} listener spiritual siap!")
    
    async def start_monitoring(self, duration_minutes: int = 5):
        """Mulai monitoring spiritual"""
        print(f"👁️ Memulai monitoring spiritual selama {duration_minutes} menit...")
        
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        check_interval = 30  # 30 detik
        
        while datetime.now() < end_time:
            print(f"🔍 Checking semua aset spiritual... ({datetime.now().strftime('%H:%M:%S')})")
            
            # Monitor semua listener
            tasks = []
            for original_name, listener in self.listeners.items():
                tasks.append(self.monitor_single_asset(original_name, listener))
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Tunggu interval berikutnya
            await asyncio.sleep(check_interval)
        
        print("✅ Monitoring selesai!")
    
    async def monitor_single_asset(self, original_name: str, listener: SpiritualListener):
        """Monitor satu aset spiritual"""
        try:
            status_data = await listener.listen()
            
            # Log ke database
            asset_data = self.asset_registry[original_name]
            self.database.log_transaction(
                asset_id=asset_data['asset_id'],
                transaction_type='health_check',
                status_data=status_data
            )
            
            # Print status
            status_icon = "✅" if status_data.get('is_alive') else "❌"
            ssl_icon = "🔒" if status_data.get('ssl_valid') else "🔓"
            
            print(f"  {status_icon} {ssl_icon} {listener.spiritual_name} ({original_name})")
            
        except Exception as e:
            print(f"  ⚠️ Error monitoring {original_name}: {e}")
    
    def generate_daily_reflection(self) -> str:
        """Generate refleksi harian"""
        print("🙏 Membuat refleksi harian...")
        
        reflection_data = self.database.create_daily_reflection()
        
        # Simpan ke file juga
        reflection_file = f"refleksi_harian_{datetime.now().strftime('%Y%m%d')}.txt"
        
        with open(reflection_file, 'w', encoding='utf-8') as f:
            f.write(reflection_data['reflection_text'])
        
        print(f"📝 Refleksi tersimpan di: {reflection_file}")
        return reflection_file

async def main():
    """Fungsi utama integrasi spiritual"""
    print("🌟 ORBIT SPIRITUAL INTEGRATOR")
    print("=" * 50)
    print("🔗 Memulai integrasi sistem spiritual...")
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
    print("\n🎧 TAHAP 2: Membuat Listener")
    integrator.create_listeners()
    
    # Mulai monitoring
    print("\n👁️ TAHAP 3: Monitoring Spiritual")
    await integrator.start_monitoring(duration_minutes=2)  # 2 menit untuk demo
    
    # Generate refleksi harian
    print("\n🙏 TAHAP 4: Refleksi Harian")
    reflection_file = integrator.generate_daily_reflection()
    
    # Ringkasan
    print("\n✨ RINGKASAN INTEGRASI:")
    print("=" * 30)
    print(f"🎧 Listener Aktif: {len(integrator.listeners)}")
    print(f"💾 Database: {integrator.database.db_path}")
    print(f"📝 Refleksi: {reflection_file}")
    
    print("\n🙏 Alhamdulillahi rabbil alamiin")
    print("✨ Integrasi sistem spiritual selesai dengan berkah")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🌸 Integrasi dihentikan dengan lembut...")
    except Exception as e:
        print(f"\n⚠️ Terjadi kesalahan: {e}")