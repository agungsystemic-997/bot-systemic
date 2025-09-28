#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌟 ORBIT ASSET IDENTIFIER
Ladang Berkah Digital - ZeroLight Orbit System
Identifikasi & Validasi Aset Web Spiritual
"""

import asyncio
import aiohttp
import socket
import ssl
import dns.resolver
import csv
import json
import sqlite3
import whois
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlparse
import subprocess
import sys

class SpiritualAssetIdentifier:
    """Pengidentifikasi Aset Spiritual untuk Planet, Wilayah, dan Makhluk"""
    
    def __init__(self, db_path: str = "./spiritual_web_discovery.db"):
        self.db_path = db_path
        self.discovered_assets = {
            'planets': [],      # Domain utama
            'wilayah': [],      # Subdomain
            'makhluk': []       # Sub-subdomain atau endpoint spesifik
        }
        self.validation_results = {}
        
    async def identifikasi_aset_dari_database(self) -> Dict[str, List]:
        """Mengidentifikasi aset dari database discovery yang sudah ada"""
        print("🔍 Mengidentifikasi aset dari database discovery...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Ambil semua domain yang ditemukan
            cursor.execute("SELECT DISTINCT domain FROM domains WHERE is_active = 1")
            planets = [row[0] for row in cursor.fetchall()]
            
            # Ambil semua subdomain yang ditemukan
            cursor.execute("SELECT DISTINCT subdomain FROM subdomains WHERE is_active = 1")
            wilayah = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            
            self.discovered_assets['planets'] = planets
            self.discovered_assets['wilayah'] = wilayah
            
            print(f"🌍 Planet ditemukan: {len(planets)}")
            print(f"🌊 Wilayah ditemukan: {len(wilayah)}")
            
        except Exception as e:
            print(f"⚠️ Error membaca database: {e}")
            # Fallback ke CSV jika database tidak tersedia
            await self.identifikasi_aset_dari_csv()
        
        return self.discovered_assets
    
    async def identifikasi_aset_dari_csv(self) -> Dict[str, List]:
        """Fallback: Identifikasi aset dari file CSV"""
        print("📄 Mengidentifikasi aset dari file CSV...")
        
        csv_path = "./log/sea_water.csv"
        try:
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row in reader:
                    if row['Jenis'] == 'Planet_Earth' and row['Status_Hidup'] == 'True':
                        self.discovered_assets['planets'].append(row['Nama'])
                    elif row['Jenis'] == 'Wilayah_Sea' and row['Status_Hidup'] == 'True':
                        self.discovered_assets['wilayah'].append(row['Nama'])
            
            print(f"🌍 Planet dari CSV: {len(self.discovered_assets['planets'])}")
            print(f"🌊 Wilayah dari CSV: {len(self.discovered_assets['wilayah'])}")
            
        except Exception as e:
            print(f"⚠️ Error membaca CSV: {e}")
            # Fallback ke daftar default
            self.discovered_assets['planets'] = ['google.com', 'github.com', 'stackoverflow.com']
            self.discovered_assets['wilayah'] = ['www.google.com', 'api.google.com', 'blog.github.com']
        
        return self.discovered_assets
    
    async def validasi_dns_aktif(self, domain: str) -> Tuple[bool, Optional[str]]:
        """Validasi apakah domain masih aktif secara DNS"""
        try:
            ip_address = socket.gethostbyname(domain)
            return True, ip_address
        except Exception:
            return False, None
    
    async def validasi_http_https(self, domain: str) -> Dict[str, bool]:
        """Validasi akses HTTP dan HTTPS"""
        results = {'http': False, 'https': False, 'redirect': False, 'error_404': False}
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            # Test HTTP
            try:
                async with session.get(f"http://{domain}", allow_redirects=False) as response:
                    results['http'] = response.status < 400
                    if response.status in [301, 302, 303, 307, 308]:
                        results['redirect'] = True
                    if response.status == 404:
                        results['error_404'] = True
            except:
                pass
            
            # Test HTTPS
            try:
                async with session.get(f"https://{domain}", allow_redirects=False) as response:
                    results['https'] = response.status < 400
                    if response.status in [301, 302, 303, 307, 308]:
                        results['redirect'] = True
                    if response.status == 404:
                        results['error_404'] = True
            except:
                pass
        
        return results
    
    async def validasi_ssl(self, domain: str) -> Dict[str, any]:
        """Validasi SSL certificate"""
        ssl_info = {'valid': False, 'issuer': None, 'expiry': None, 'error': None}
        
        try:
            context = ssl.create_default_context()
            with socket.create_connection((domain, 443), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert = ssock.getpeercert()
                    ssl_info['valid'] = True
                    ssl_info['issuer'] = cert.get('issuer', [{}])[0].get('organizationName', 'Unknown')
                    ssl_info['expiry'] = cert.get('notAfter', 'Unknown')
        except Exception as e:
            ssl_info['error'] = str(e)
        
        return ssl_info
    
    async def pembersihan_dan_validasi_lengkap(self) -> Dict[str, Dict]:
        """Pembersihan dan validasi lengkap semua aset"""
        print("\n🧼 Memulai pembersihan dan validasi lengkap...")
        
        all_domains = self.discovered_assets['planets'] + self.discovered_assets['wilayah']
        validated_assets = {}
        
        for domain in all_domains:
            print(f"🔍 Memvalidasi: {domain}")
            
            # DNS validation
            dns_active, ip_address = await self.validasi_dns_aktif(domain)
            
            # HTTP/HTTPS validation
            http_results = await self.validasi_http_https(domain) if dns_active else {}
            
            # SSL validation
            ssl_results = await self.validasi_ssl(domain) if dns_active else {}
            
            validated_assets[domain] = {
                'dns_active': dns_active,
                'ip_address': ip_address,
                'http_accessible': http_results.get('http', False),
                'https_accessible': http_results.get('https', False),
                'has_redirect': http_results.get('redirect', False),
                'has_404_error': http_results.get('error_404', False),
                'ssl_valid': ssl_results.get('valid', False),
                'ssl_issuer': ssl_results.get('issuer'),
                'ssl_expiry': ssl_results.get('expiry'),
                'ssl_error': ssl_results.get('error'),
                'validation_time': datetime.now().isoformat(),
                'status': 'active' if dns_active and (http_results.get('http') or http_results.get('https')) else 'inactive'
            }
            
            # Jeda lembut
            await asyncio.sleep(0.5)
        
        self.validation_results = validated_assets
        return validated_assets
    
    def generate_spiritual_naming_scheme(self) -> Dict[str, Dict]:
        """Generate skema penamaan spiritual untuk aset"""
        print("\n🧭 Menggenerate skema penamaan spiritual...")
        
        spiritual_names = {}
        planet_counter = 1
        wilayah_counter = 1
        
        for domain, validation in self.validation_results.items():
            if validation['status'] == 'active':
                # Tentukan apakah ini planet atau wilayah
                if '.' in domain and domain.count('.') == 1:
                    # Domain utama = Planet
                    spiritual_name = f"planet{planet_counter}.earth.orbit.system"
                    category = "planet"
                    planet_counter += 1
                else:
                    # Subdomain = Wilayah
                    spiritual_name = f"wilayah{wilayah_counter}.sea.earth.orbit.system"
                    category = "wilayah"
                    wilayah_counter += 1
                
                spiritual_names[domain] = {
                    'original_name': domain,
                    'spiritual_name': spiritual_name,
                    'category': category,
                    'validation_data': validation,
                    'assigned_at': datetime.now().isoformat()
                }
        
        return spiritual_names
    
    def save_results_to_database(self, spiritual_names: Dict) -> str:
        """Simpan hasil ke database spiritual"""
        db_file = "./spiritual_asset_registry.db"
        
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # Buat tabel jika belum ada
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS spiritual_assets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_name TEXT UNIQUE,
                    spiritual_name TEXT,
                    category TEXT,
                    dns_active BOOLEAN,
                    ip_address TEXT,
                    http_accessible BOOLEAN,
                    https_accessible BOOLEAN,
                    ssl_valid BOOLEAN,
                    ssl_issuer TEXT,
                    status TEXT,
                    assigned_at TEXT,
                    validation_time TEXT
                )
            ''')
            
            # Insert data
            for domain, data in spiritual_names.items():
                validation = data['validation_data']
                cursor.execute('''
                    INSERT OR REPLACE INTO spiritual_assets 
                    (original_name, spiritual_name, category, dns_active, ip_address, 
                     http_accessible, https_accessible, ssl_valid, ssl_issuer, status, 
                     assigned_at, validation_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    domain, data['spiritual_name'], data['category'],
                    validation['dns_active'], validation['ip_address'],
                    validation['http_accessible'], validation['https_accessible'],
                    validation['ssl_valid'], validation['ssl_issuer'],
                    validation['status'], data['assigned_at'], validation['validation_time']
                ))
            
            conn.commit()
            conn.close()
            
            print(f"💾 Data tersimpan di: {db_file}")
            return db_file
            
        except Exception as e:
            print(f"⚠️ Error menyimpan ke database: {e}")
            return ""
    
    def generate_report(self, spiritual_names: Dict) -> str:
        """Generate laporan lengkap"""
        report_file = f"spiritual_asset_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report_data = {
            'generated_at': datetime.now().isoformat(),
            'total_assets_identified': len(self.discovered_assets['planets']) + len(self.discovered_assets['wilayah']),
            'total_active_assets': len([a for a in self.validation_results.values() if a['status'] == 'active']),
            'total_spiritual_names': len(spiritual_names),
            'summary': {
                'planets': len([s for s in spiritual_names.values() if s['category'] == 'planet']),
                'wilayah': len([s for s in spiritual_names.values() if s['category'] == 'wilayah']),
                'ssl_enabled': len([v for v in self.validation_results.values() if v['ssl_valid']]),
                'https_accessible': len([v for v in self.validation_results.values() if v['https_accessible']])
            },
            'spiritual_assets': spiritual_names,
            'validation_results': self.validation_results
        }
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            print(f"📊 Laporan tersimpan di: {report_file}")
            return report_file
            
        except Exception as e:
            print(f"⚠️ Error menyimpan laporan: {e}")
            return ""

async def main():
    """Fungsi utama untuk identifikasi aset spiritual"""
    print("🌟 ORBIT ASSET IDENTIFIER - ZeroLight System")
    print("=" * 50)
    print("🔍 Memulai identifikasi dan validasi aset spiritual...")
    print()
    
    # Inisialisasi identifier
    identifier = SpiritualAssetIdentifier()
    
    # 1. Identifikasi aset dari database/CSV
    print("📋 TAHAP 1: Identifikasi Aset")
    discovered_assets = await identifier.identifikasi_aset_dari_database()
    
    # 2. Pembersihan dan validasi
    print("\n🧼 TAHAP 2: Pembersihan & Validasi")
    validation_results = await identifier.pembersihan_dan_validasi_lengkap()
    
    # 3. Penamaan spiritual
    print("\n🧭 TAHAP 3: Penamaan Spiritual")
    spiritual_names = identifier.generate_spiritual_naming_scheme()
    
    # 4. Simpan hasil
    print("\n💾 TAHAP 4: Penyimpanan Hasil")
    db_file = identifier.save_results_to_database(spiritual_names)
    report_file = identifier.generate_report(spiritual_names)
    
    # 5. Tampilkan ringkasan
    print("\n✨ RINGKASAN HASIL:")
    print("=" * 30)
    
    active_count = len([a for a in validation_results.values() if a['status'] == 'active'])
    ssl_count = len([v for v in validation_results.values() if v['ssl_valid']])
    
    print(f"🌍 Total Planet: {len([s for s in spiritual_names.values() if s['category'] == 'planet'])}")
    print(f"🌊 Total Wilayah: {len([s for s in spiritual_names.values() if s['category'] == 'wilayah'])}")
    print(f"⚡ Aset Aktif: {active_count}")
    print(f"🔒 SSL Terlindungi: {ssl_count}")
    print(f"💾 Database: {db_file}")
    print(f"📊 Laporan: {report_file}")
    
    print("\n🙏 Alhamdulillahi rabbil alamiin")
    print("✨ Identifikasi aset selesai dengan berkah")
    
    return spiritual_names, validation_results

if __name__ == "__main__":
    try:
        spiritual_names, validation_results = asyncio.run(main())
        print(f"\n🎯 Proses selesai dengan {len(spiritual_names)} aset spiritual!")
    except KeyboardInterrupt:
        print("\n🌸 Proses dihentikan dengan lembut...")
    except Exception as e:
        print(f"\n⚠️ Terjadi kesalahan: {e}")