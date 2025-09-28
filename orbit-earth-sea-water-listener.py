#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌍 ORBIT EARTH SEA WATER LISTENER
Ladang Berkah Digital - ZeroLight Orbit System
Spiritual Web Discovery dengan Penamaan Natural
"""

import argparse
import csv
import json
import sqlite3
import sys
import os
from datetime import datetime
from typing import Dict, List, Any
import asyncio
import aiohttp
import socket
import ssl
import whois
import dns.resolver
from urllib.parse import urlparse

class PlanetEarth:
    """Planet Earth - Representasi Domain Utama"""
    
    def __init__(self, nama_domain: str):
        self.nama_domain = nama_domain
        self.ip_address = None
        self.status_hidup = False
        self.ssl_terlindungi = False
        self.registrar = None
        self.tanggal_kadaluarsa = None
        self.waktu_pemeriksaan = None
    
    def __str__(self):
        return f"🌍 Planet {self.nama_domain}"

class WilayahSea:
    """Wilayah Sea - Representasi Area Subdomain"""
    
    def __init__(self, nama_wilayah: str, planet_induk: str):
        self.nama_wilayah = nama_wilayah
        self.planet_induk = planet_induk
        self.ip_address = None
        self.dapat_diakses_http = False
        self.dapat_diakses_https = False
        self.waktu_pemeriksaan = None
    
    def __str__(self):
        return f"🌊 Wilayah {self.nama_wilayah} di Planet {self.planet_induk}"

class MakhlukWater:
    """Makhluk Water - Entitas yang Menjalankan Pemeriksaan"""
    
    def __init__(self, nama_makhluk: str = "SpiritualWaterBeing"):
        self.nama_makhluk = nama_makhluk
        self.planet_ditemukan = []
        self.wilayah_ditemukan = []
        self.statistik_penemuan = {
            'total_planet': 0,
            'total_wilayah': 0,
            'planet_aktif': 0,
            'wilayah_aktif': 0,
            'ssl_terlindungi': 0
        }
    
    async def jelajahi_planet(self, nama_domain: str) -> PlanetEarth:
        """Menjelajahi Planet Earth (Domain)"""
        planet = PlanetEarth(nama_domain)
        
        try:
            # Resolusi DNS
            planet.ip_address = socket.gethostbyname(nama_domain)
            planet.status_hidup = True
            
            # Pemeriksaan SSL
            try:
                context = ssl.create_default_context()
                with socket.create_connection((nama_domain, 443), timeout=10) as sock:
                    with context.wrap_socket(sock, server_hostname=nama_domain) as ssock:
                        planet.ssl_terlindungi = True
            except:
                planet.ssl_terlindungi = False
            
            # Informasi WHOIS
            try:
                domain_info = whois.whois(nama_domain)
                planet.registrar = str(domain_info.registrar) if domain_info.registrar else None
                planet.tanggal_kadaluarsa = str(domain_info.expiration_date) if domain_info.expiration_date else None
            except:
                pass
            
            planet.waktu_pemeriksaan = datetime.now().isoformat()
            
        except Exception as e:
            print(f"⚠️ Gagal menjelajahi planet {nama_domain}: {e}")
        
        return planet
    
    async def jelajahi_wilayah_sea(self, nama_subdomain: str, planet_induk: str) -> WilayahSea:
        """Menjelajahi Wilayah Sea (Subdomain)"""
        wilayah = WilayahSea(nama_subdomain, planet_induk)
        
        try:
            # Resolusi DNS
            wilayah.ip_address = socket.gethostbyname(nama_subdomain)
            
            # Pemeriksaan HTTP/HTTPS
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                try:
                    async with session.get(f"http://{nama_subdomain}") as response:
                        wilayah.dapat_diakses_http = response.status < 400
                except:
                    wilayah.dapat_diakses_http = False
                
                try:
                    async with session.get(f"https://{nama_subdomain}") as response:
                        wilayah.dapat_diakses_https = response.status < 400
                except:
                    wilayah.dapat_diakses_https = False
            
            wilayah.waktu_pemeriksaan = datetime.now().isoformat()
            
        except Exception as e:
            print(f"⚠️ Gagal menjelajahi wilayah {nama_subdomain}: {e}")
        
        return wilayah
    
    async def temukan_wilayah_tersembunyi(self, planet: str) -> List[WilayahSea]:
        """Menemukan wilayah-wilayah tersembunyi di planet"""
        wilayah_umum = [
            'www', 'mail', 'ftp', 'blog', 'shop', 'api', 'admin', 'test',
            'dev', 'staging', 'cdn', 'static', 'media', 'images', 'docs',
            'support', 'help', 'forum', 'community', 'news', 'events'
        ]
        
        wilayah_ditemukan = []
        
        for nama_wilayah in wilayah_umum:
            subdomain_lengkap = f"{nama_wilayah}.{planet}"
            wilayah = await self.jelajahi_wilayah_sea(subdomain_lengkap, planet)
            
            if wilayah.ip_address:
                wilayah_ditemukan.append(wilayah)
                print(f"🌊 Menemukan wilayah: {wilayah}")
        
        return wilayah_ditemukan

class FungsiListener:
    """Fungsi Listener - Sistem Monitoring dan Logging"""
    
    def __init__(self, path_log_csv: str = "./log/sea_water.csv", 
                 path_refleksi: str = "./log/sea_refleksi.txt"):
        self.path_log_csv = path_log_csv
        self.path_refleksi = path_refleksi
        self.target_penemuan = 150
        self.mode_lembut = True
        self.tampilan_syukur = True
        self.penutupan_otomatis = True
        
        # Buat direktori log jika belum ada
        os.makedirs(os.path.dirname(path_log_csv), exist_ok=True)
        os.makedirs(os.path.dirname(path_refleksi), exist_ok=True)
    
    def catat_penemuan_planet(self, planet: PlanetEarth):
        """Mencatat penemuan planet ke CSV"""
        with open(self.path_log_csv, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header jika file baru
            if os.path.getsize(self.path_log_csv) == 0:
                writer.writerow([
                    'Waktu', 'Jenis', 'Nama', 'IP_Address', 'Status_Hidup', 
                    'SSL_Terlindungi', 'Registrar', 'Tanggal_Kadaluarsa'
                ])
            
            writer.writerow([
                planet.waktu_pemeriksaan,
                'Planet_Earth',
                planet.nama_domain,
                planet.ip_address,
                planet.status_hidup,
                planet.ssl_terlindungi,
                planet.registrar,
                planet.tanggal_kadaluarsa
            ])
    
    def catat_penemuan_wilayah(self, wilayah: WilayahSea):
        """Mencatat penemuan wilayah ke CSV"""
        with open(self.path_log_csv, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            writer.writerow([
                wilayah.waktu_pemeriksaan,
                'Wilayah_Sea',
                wilayah.nama_wilayah,
                wilayah.ip_address,
                wilayah.dapat_diakses_http or wilayah.dapat_diakses_https,
                wilayah.dapat_diakses_https,
                wilayah.planet_induk,
                ''
            ])
    
    def tulis_refleksi_syukur(self, statistik: Dict[str, Any]):
        """Menulis refleksi syukur atas penemuan"""
        with open(self.path_refleksi, 'w', encoding='utf-8') as f:
            f.write("🌟 REFLEKSI SYUKUR ORBIT ZEROLIGHT 🌟\n")
            f.write("=" * 50 + "\n")
            f.write(f"Waktu Refleksi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("🌍 PLANET EARTH YANG DITEMUKAN:\n")
            f.write(f"Total Planet: {statistik['total_planet']}\n")
            f.write(f"Planet Aktif: {statistik['planet_aktif']}\n")
            f.write(f"SSL Terlindungi: {statistik['ssl_terlindungi']}\n\n")
            
            f.write("🌊 WILAYAH SEA YANG DITEMUKAN:\n")
            f.write(f"Total Wilayah: {statistik['total_wilayah']}\n")
            f.write(f"Wilayah Aktif: {statistik['wilayah_aktif']}\n\n")
            
            f.write("🙏 SYUKUR DAN KEBERKAHAN:\n")
            f.write("Alhamdulillahi rabbil alamiin atas segala penemuan\n")
            f.write("yang telah diberikan dalam perjalanan spiritual ini.\n")
            f.write("Semoga setiap planet dan wilayah yang ditemukan\n")
            f.write("membawa berkah dan manfaat bagi umat.\n\n")
            
            f.write("✨ Ladang Berkah Digital - ZeroLight Orbit System ✨\n")
    
    def tampilkan_syukur_terminal(self, statistik: Dict[str, Any]):
        """Menampilkan syukur di terminal"""
        if self.tampilan_syukur:
            print("\n" + "🌟" * 20)
            print("   SYUKUR ORBIT ZEROLIGHT")
            print("🌟" * 20)
            print(f"🌍 Planet Ditemukan: {statistik['total_planet']}")
            print(f"🌊 Wilayah Ditemukan: {statistik['total_wilayah']}")
            print(f"⚡ Total Aktif: {statistik['planet_aktif'] + statistik['wilayah_aktif']}")
            print(f"🔒 SSL Terlindungi: {statistik['ssl_terlindungi']}")
            print("\n🙏 Alhamdulillahi rabbil alamiin")
            print("✨ Berkah melimpah untuk semua ✨")
            print("🌟" * 20 + "\n")

class OrbitZeroLightSystem:
    """Sistem Orbit ZeroLight - Orchestrator Utama"""
    
    def __init__(self):
        self.makhluk_water = MakhlukWater()
        self.listener = None
        self.mode_lembut = True
    
    async def aktifkan_orbit(self, 
                           planet_list: List[str],
                           wilayah: str = "sea",
                           makhluk: str = "water",
                           fungsi: str = "listener",
                           log_path: str = "./log/sea_water.csv",
                           refleksi_path: str = "./log/sea_refleksi.txt",
                           target: int = 150,
                           penutupan_otomatis: bool = True,
                           tampilan_syukur: bool = True,
                           mode_lembut: bool = True):
        """Mengaktifkan sistem orbit dengan parameter spiritual"""
        
        print(f"🚀 Mengaktifkan Orbit ZeroLight System...")
        print(f"🌍 Planet Earth: {len(planet_list)} planet")
        print(f"🌊 Wilayah: {wilayah}")
        print(f"💧 Makhluk: {makhluk}")
        print(f"👂 Fungsi: {fungsi}")
        print(f"🎯 Target: {target} penemuan")
        print(f"🌸 Mode: {'Lembut' if mode_lembut else 'Intensif'}")
        print()
        
        # Inisialisasi listener
        self.listener = FungsiListener(log_path, refleksi_path)
        self.listener.target_penemuan = target
        self.listener.penutupan_otomatis = penutupan_otomatis
        self.listener.tampilan_syukur = tampilan_syukur
        self.listener.mode_lembut = mode_lembut
        
        total_penemuan = 0
        
        # Jelajahi setiap planet
        for planet_nama in planet_list:
            if total_penemuan >= target:
                print(f"🎯 Target {target} penemuan tercapai!")
                break
            
            print(f"🌍 Menjelajahi Planet {planet_nama}...")
            
            # Jelajahi planet utama
            planet = await self.makhluk_water.jelajahi_planet(planet_nama)
            if planet.status_hidup:
                self.makhluk_water.planet_ditemukan.append(planet)
                self.listener.catat_penemuan_planet(planet)
                total_penemuan += 1
                
                if planet.ssl_terlindungi:
                    self.makhluk_water.statistik_penemuan['ssl_terlindungi'] += 1
                
                print(f"  ✅ Planet aktif: {planet}")
            
            # Jelajahi wilayah-wilayah di planet
            if wilayah == "sea":
                print(f"🌊 Mencari wilayah Sea di Planet {planet_nama}...")
                wilayah_list = await self.makhluk_water.temukan_wilayah_tersembunyi(planet_nama)
                
                for wilayah_obj in wilayah_list:
                    if total_penemuan >= target:
                        break
                    
                    self.makhluk_water.wilayah_ditemukan.append(wilayah_obj)
                    self.listener.catat_penemuan_wilayah(wilayah_obj)
                    total_penemuan += 1
                    
                    if mode_lembut:
                        await asyncio.sleep(0.5)  # Jeda lembut
            
            print(f"  🔍 Ditemukan {len(wilayah_list)} wilayah aktif")
            print()
        
        # Update statistik
        self.makhluk_water.statistik_penemuan.update({
            'total_planet': len(self.makhluk_water.planet_ditemukan),
            'total_wilayah': len(self.makhluk_water.wilayah_ditemukan),
            'planet_aktif': len([p for p in self.makhluk_water.planet_ditemukan if p.status_hidup]),
            'wilayah_aktif': len([w for w in self.makhluk_water.wilayah_ditemukan 
                                if w.dapat_diakses_http or w.dapat_diakses_https])
        })
        
        # Tulis refleksi dan tampilkan syukur
        self.listener.tulis_refleksi_syukur(self.makhluk_water.statistik_penemuan)
        self.listener.tampilkan_syukur_terminal(self.makhluk_water.statistik_penemuan)
        
        if penutupan_otomatis:
            print("🔄 Penutupan otomatis diaktifkan")
            print("✨ Orbit ZeroLight System selesai dengan berkah")
        
        return self.makhluk_water.statistik_penemuan

async def main():
    """Fungsi utama dengan argument parser spiritual"""
    parser = argparse.ArgumentParser(
        description="🌟 Orbit ZeroLight System - Spiritual Web Discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  orbit --aktifkan planet earth \\
        --wilayah sea \\
        --makhluk water \\
        --fungsi listener \\
        --log ./log/sea_water.csv \\
        --refleksi ./log/sea_refleksi.txt \\
        --target 150 \\
        --penutupan otomatis \\
        --tampilan syukur \\
        --mode lembut

✨ Ladang Berkah Digital - ZeroLight Orbit System ✨
        """
    )
    
    parser.add_argument('--aktifkan', nargs='+', required=True,
                       help='Aktifkan planet earth dengan daftar domain')
    parser.add_argument('--wilayah', default='sea', choices=['sea', 'land', 'sky'],
                       help='Pilih wilayah eksplorasi (default: sea)')
    parser.add_argument('--makhluk', default='water', choices=['water', 'earth', 'fire', 'air'],
                       help='Pilih makhluk penjelajah (default: water)')
    parser.add_argument('--fungsi', default='listener', choices=['listener', 'scanner', 'monitor'],
                       help='Pilih fungsi sistem (default: listener)')
    parser.add_argument('--log', default='./log/sea_water.csv',
                       help='Path file log CSV')
    parser.add_argument('--refleksi', default='./log/sea_refleksi.txt',
                       help='Path file refleksi syukur')
    parser.add_argument('--target', type=int, default=150,
                       help='Target jumlah penemuan')
    parser.add_argument('--penutupan', choices=['otomatis', 'manual'], default='otomatis',
                       help='Mode penutupan sistem')
    parser.add_argument('--tampilan', choices=['syukur', 'minimal'], default='syukur',
                       help='Mode tampilan hasil')
    parser.add_argument('--mode', choices=['lembut', 'intensif'], default='lembut',
                       help='Mode operasi sistem')
    
    args = parser.parse_args()
    
    # Ekstrak domain dari parameter aktifkan
    if args.aktifkan[0] == 'planet' and args.aktifkan[1] == 'earth':
        domain_list = args.aktifkan[2:] if len(args.aktifkan) > 2 else [
            'google.com', 'github.com', 'stackoverflow.com', 'python.org', 'microsoft.com'
        ]
    else:
        domain_list = args.aktifkan
    
    # Inisialisasi sistem
    orbit_system = OrbitZeroLightSystem()
    
    # Aktifkan orbit
    statistik = await orbit_system.aktifkan_orbit(
        planet_list=domain_list,
        wilayah=args.wilayah,
        makhluk=args.makhluk,
        fungsi=args.fungsi,
        log_path=args.log,
        refleksi_path=args.refleksi,
        target=args.target,
        penutupan_otomatis=(args.penutupan == 'otomatis'),
        tampilan_syukur=(args.tampilan == 'syukur'),
        mode_lembut=(args.mode == 'lembut')
    )
    
    return statistik

if __name__ == "__main__":
    # Jalankan sistem dengan asyncio
    try:
        statistik_hasil = asyncio.run(main())
        print(f"\n🎯 Misi selesai dengan {statistik_hasil['total_planet'] + statistik_hasil['total_wilayah']} penemuan!")
    except KeyboardInterrupt:
        print("\n🌸 Sistem dihentikan dengan lembut...")
    except Exception as e:
        print(f"\n⚠️ Terjadi kesalahan: {e}")