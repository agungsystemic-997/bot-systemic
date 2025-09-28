#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌟 ORBIT DASHBOARD REFLECTION - OPTIMIZED
Ladang Berkah Digital - ZeroLight Orbit System
Dashboard Optimized dengan Cache, Delay & Metadata Stabil
"""

import asyncio
import sqlite3
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List
from pathlib import Path
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs
import threading
import webbrowser
import hashlib
from collections import defaultdict

class SpiritualDashboardCache:
    """Cache untuk dashboard spiritual"""
    
    def __init__(self, cache_duration_minutes: int = 10):
        self.cache = {}
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self.last_db_check = {}
    
    def get_cache_key(self, query_type: str, params: str = "") -> str:
        """Generate cache key"""
        return hashlib.md5(f"{query_type}_{params}".encode()).hexdigest()
    
    def get(self, query_type: str, params: str = "") -> Dict:
        """Ambil dari cache jika masih valid"""
        cache_key = self.get_cache_key(query_type, params)
        
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if datetime.now() - cached_data['timestamp'] < self.cache_duration:
                return cached_data['data']
            else:
                # Cache expired
                del self.cache[cache_key]
        
        return None
    
    def set(self, query_type: str, data: Dict, params: str = ""):
        """Simpan ke cache"""
        cache_key = self.get_cache_key(query_type, params)
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
    
    def get_cache_stats(self) -> Dict:
        """Statistik cache"""
        return {
            'total_entries': len(self.cache),
            'cache_duration_minutes': self.cache_duration.total_seconds() / 60,
            'oldest_entry': min([v['timestamp'] for v in self.cache.values()]) if self.cache else None,
            'newest_entry': max([v['timestamp'] for v in self.cache.values()]) if self.cache else None
        }

class SpiritualMetadataStabilizer:
    """Stabilizer untuk metadata sistem"""
    
    def __init__(self):
        self.stable_metadata = {
            'system_info': {
                'name': 'ZeroLight Orbit Spiritual System',
                'version': '2.0.0-optimized',
                'purpose': 'Spiritual Digital Asset Monitoring',
                'description': 'Sistem monitoring spiritual untuk kesehatan digital dengan adab',
                'monitoring_type': 'health_check_only',
                'data_collection': 'minimal_headers_only',
                'respect_robots': True,
                'rate_limited': True,
                'cache_enabled': True
            },
            'spiritual_context': {
                'philosophy': 'Monitoring dengan adab dan hormat',
                'intention': 'Menjaga kesehatan digital dengan spiritual',
                'ethics': 'Tidak mengganggu, tidak mengambil data pribadi',
                'gratitude': 'Alhamdulillahi rabbil alamiin'
            },
            'technical_specs': {
                'request_method': 'HEAD (minimal impact)',
                'user_agent': 'ZeroLight-Orbit-Spiritual-Monitor/2.0 (Health Check Only)',
                'max_requests_per_minute': 2,
                'max_concurrent_requests': 3,
                'cache_duration_minutes': 10,
                'monitoring_interval_minutes': 8
            }
        }
    
    def get_stable_metadata(self) -> Dict:
        """Dapatkan metadata yang stabil"""
        return self.stable_metadata.copy()
    
    def get_request_headers(self) -> Dict:
        """Headers yang menjelaskan tujuan monitoring"""
        return {
            'User-Agent': self.stable_metadata['technical_specs']['user_agent'],
            'X-Purpose': 'spiritual_monitoring',
            'X-Monitor-Type': 'health_check',
            'X-System': self.stable_metadata['system_info']['name'],
            'X-Version': self.stable_metadata['system_info']['version'],
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'id,en;q=0.9',
            'DNT': '1',  # Do Not Track
            'Connection': 'close'
        }

class SpiritualDashboardGenerator:
    """Generator Dashboard Spiritual dengan optimisasi"""
    
    def __init__(self, db_path: str = "./spiritual_orbit_system.db"):
        self.db_path = db_path
        self.dashboard_dir = Path("./spiritual_dashboard_optimized")
        self.dashboard_dir.mkdir(exist_ok=True)
        self.cache = SpiritualDashboardCache(cache_duration_minutes=10)
        self.metadata_stabilizer = SpiritualMetadataStabilizer()
        self.generation_delay = 2  # 2 detik delay antar generasi
        
    def get_spiritual_assets_data(self) -> Dict:
        """Ambil data aset spiritual dengan cache"""
        
        # Cek cache terlebih dahulu
        cached_data = self.cache.get('assets_data')
        if cached_data:
            print("📋 Menggunakan data dari cache")
            return cached_data
        
        print("🔄 Mengambil data fresh dari database...")
        time.sleep(self.generation_delay)  # Delay untuk stabilitas
        
        if not os.path.exists(self.db_path):
            empty_data = {'assets': [], 'transactions': [], 'reflections': []}
            self.cache.set('assets_data', empty_data)
            return empty_data
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Ambil data aset
            cursor.execute('''
                SELECT id, original_name, spiritual_name, category, ip_address, status, created_at
                FROM spiritual_assets
            ''')
            assets = []
            for row in cursor.fetchall():
                assets.append({
                    'id': row[0],
                    'original_name': row[1],
                    'spiritual_name': row[2],
                    'category': row[3],
                    'ip_address': row[4],
                    'status': row[5],
                    'created_at': row[6]
                })
            
            # Ambil transaksi terbaru (24 jam terakhir)
            cursor.execute('''
                SELECT asset_id, transaction_type, status_code, response_time, 
                       is_alive, ssl_valid, timestamp, from_cache
                FROM transaction_logs 
                WHERE datetime(timestamp) > datetime('now', '-24 hours')
                ORDER BY timestamp DESC
                LIMIT 100
            ''')
            transactions = []
            for row in cursor.fetchall():
                transactions.append({
                    'asset_id': row[0],
                    'transaction_type': row[1],
                    'status_code': row[2],
                    'response_time': row[3],
                    'is_alive': bool(row[4]),
                    'ssl_valid': bool(row[5]),
                    'timestamp': row[6],
                    'from_cache': bool(row[7]) if row[7] is not None else False
                })
            
            # Ambil refleksi terbaru
            cursor.execute('''
                SELECT reflection_date, total_assets, active_assets, ssl_protected,
                       total_transactions, success_rate, reflection_text, traffic_summary
                FROM daily_reflections 
                ORDER BY reflection_date DESC 
                LIMIT 5
            ''')
            reflections = []
            for row in cursor.fetchall():
                reflections.append({
                    'reflection_date': row[0],
                    'total_assets': row[1],
                    'active_assets': row[2],
                    'ssl_protected': row[3],
                    'total_transactions': row[4],
                    'success_rate': row[5],
                    'reflection_text': row[6],
                    'traffic_summary': json.loads(row[7]) if row[7] else {}
                })
            
            data = {
                'assets': assets,
                'transactions': transactions,
                'reflections': reflections,
                'metadata': self.metadata_stabilizer.get_stable_metadata(),
                'cache_stats': self.cache.get_cache_stats(),
                'generated_at': datetime.now().isoformat()
            }
            
            # Simpan ke cache
            self.cache.set('assets_data', data)
            
            conn.close()
            return data
            
        except Exception as e:
            print(f"⚠️ Error mengambil data: {e}")
            conn.close()
            return {'assets': [], 'transactions': [], 'reflections': []}
    
    def calculate_dashboard_stats(self, data: Dict) -> Dict:
        """Hitung statistik dashboard dengan cache"""
        
        cached_stats = self.cache.get('dashboard_stats')
        if cached_stats:
            return cached_stats
        
        assets = data['assets']
        transactions = data['transactions']
        
        # Hitung statistik
        total_assets = len(assets)
        active_assets = len([a for a in assets if a['status'] == 'active'])
        planets = len([a for a in assets if a['category'] == 'planet'])
        wilayah = len([a for a in assets if a['category'] == 'wilayah'])
        
        # Statistik transaksi
        recent_transactions = len(transactions)
        successful_transactions = len([t for t in transactions if t['is_alive']])
        ssl_protected = len([t for t in transactions if t['ssl_valid']])
        cached_requests = len([t for t in transactions if t.get('from_cache', False)])
        
        success_rate = (successful_transactions / recent_transactions * 100) if recent_transactions > 0 else 0
        ssl_rate = (ssl_protected / recent_transactions * 100) if recent_transactions > 0 else 0
        cache_hit_rate = (cached_requests / recent_transactions * 100) if recent_transactions > 0 else 0
        
        # Rata-rata response time
        response_times = [t['response_time'] for t in transactions if t['response_time'] and t['response_time'] > 0]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        stats = {
            'total_assets': total_assets,
            'active_assets': active_assets,
            'planets': planets,
            'wilayah': wilayah,
            'recent_transactions': recent_transactions,
            'successful_transactions': successful_transactions,
            'ssl_protected': ssl_protected,
            'cached_requests': cached_requests,
            'success_rate': round(success_rate, 1),
            'ssl_rate': round(ssl_rate, 1),
            'cache_hit_rate': round(cache_hit_rate, 1),
            'avg_response_time': round(avg_response_time, 3)
        }
        
        # Cache statistik
        self.cache.set('dashboard_stats', stats)
        
        return stats
    
    def generate_main_dashboard(self, data: Dict, stats: Dict) -> str:
        """Generate dashboard utama dengan optimisasi"""
        
        metadata = data.get('metadata', {})
        system_info = metadata.get('system_info', {})
        cache_stats = data.get('cache_stats', {})
        
        html_content = f"""
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌟 {system_info.get('name', 'Orbit Dashboard')} - Optimized</title>
    <meta name="description" content="{system_info.get('description', 'Spiritual monitoring system')}">
    <meta name="robots" content="noindex, nofollow">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 2px solid #e0e0e0;
        }}
        
        .header h1 {{
            color: #4a5568;
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }}
        
        .header .subtitle {{
            color: #718096;
            font-size: 1.2em;
            margin-bottom: 15px;
        }}
        
        .optimization-badge {{
            display: inline-block;
            background: linear-gradient(45deg, #48bb78, #38a169);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
            margin: 5px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #f7fafc, #edf2f7);
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            border-left: 5px solid #4299e1;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        }}
        
        .stat-card.success {{
            border-left-color: #48bb78;
        }}
        
        .stat-card.warning {{
            border-left-color: #ed8936;
        }}
        
        .stat-card.info {{
            border-left-color: #4299e1;
        }}
        
        .stat-card.cache {{
            border-left-color: #9f7aea;
        }}
        
        .stat-number {{
            font-size: 2.5em;
            font-weight: bold;
            color: #2d3748;
            margin-bottom: 10px;
        }}
        
        .stat-label {{
            color: #718096;
            font-size: 1.1em;
            font-weight: 500;
        }}
        
        .metadata-section {{
            background: #f8f9fa;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 30px;
        }}
        
        .metadata-section h3 {{
            color: #4a5568;
            margin-bottom: 15px;
            font-size: 1.3em;
        }}
        
        .metadata-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        
        .metadata-item {{
            background: white;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #4299e1;
        }}
        
        .metadata-item strong {{
            color: #2d3748;
            display: block;
            margin-bottom: 5px;
        }}
        
        .metadata-item span {{
            color: #718096;
        }}
        
        .assets-section {{
            margin-top: 30px;
        }}
        
        .assets-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .asset-card {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            transition: all 0.3s ease;
        }}
        
        .asset-card:hover {{
            border-color: #4299e1;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        .asset-card.planet {{
            border-left: 5px solid #48bb78;
        }}
        
        .asset-card.wilayah {{
            border-left: 5px solid #4299e1;
        }}
        
        .asset-name {{
            font-weight: bold;
            color: #2d3748;
            margin-bottom: 8px;
        }}
        
        .asset-spiritual {{
            color: #4299e1;
            font-style: italic;
            margin-bottom: 8px;
        }}
        
        .asset-status {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
        }}
        
        .asset-status.active {{
            background: #c6f6d5;
            color: #22543d;
        }}
        
        .asset-status.inactive {{
            background: #fed7d7;
            color: #742a2a;
        }}
        
        .refresh-info {{
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            background: #e6fffa;
            border-radius: 10px;
            border: 1px solid #81e6d9;
        }}
        
        .refresh-info p {{
            color: #234e52;
            margin-bottom: 10px;
        }}
        
        .cache-info {{
            background: #faf5ff;
            border: 1px solid #d6bcfa;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
        }}
        
        .cache-info h4 {{
            color: #553c9a;
            margin-bottom: 10px;
        }}
        
        .cache-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
        }}
        
        .cache-stat {{
            background: white;
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e2e8f0;
            color: #718096;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 20px;
                margin: 10px;
            }}
            
            .stats-grid {{
                grid-template-columns: 1fr;
            }}
            
            .header h1 {{
                font-size: 2em;
            }}
        }}
    </style>
    <script>
        // Auto refresh setiap 5 menit dengan delay
        setTimeout(function() {{
            setInterval(function() {{
                console.log('🔄 Refreshing dashboard...');
                window.location.reload();
            }}, 300000); // 5 menit
        }}, 30000); // Delay 30 detik sebelum mulai auto refresh
        
        // Update timestamp
        function updateTimestamp() {{
            const now = new Date();
            const timestamp = now.toLocaleString('id-ID');
            document.getElementById('current-time').textContent = timestamp;
        }}
        
        setInterval(updateTimestamp, 1000);
        window.onload = updateTimestamp;
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌟 {system_info.get('name', 'Orbit Dashboard')}</h1>
            <div class="subtitle">{system_info.get('description', 'Spiritual monitoring system')}</div>
            <div class="optimization-badge">⚡ Optimized v{system_info.get('version', '2.0')}</div>
            <div class="optimization-badge">📋 Cache Enabled</div>
            <div class="optimization-badge">🚦 Rate Limited</div>
            <div class="optimization-badge">🙏 Spiritual Monitoring</div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card info">
                <div class="stat-number">{stats['total_assets']}</div>
                <div class="stat-label">Total Aset Spiritual</div>
            </div>
            
            <div class="stat-card success">
                <div class="stat-number">{stats['active_assets']}</div>
                <div class="stat-label">Aset Aktif</div>
            </div>
            
            <div class="stat-card info">
                <div class="stat-number">{stats['planets']}</div>
                <div class="stat-label">Planet Earth</div>
            </div>
            
            <div class="stat-card info">
                <div class="stat-number">{stats['wilayah']}</div>
                <div class="stat-label">Wilayah Sea</div>
            </div>
            
            <div class="stat-card success">
                <div class="stat-number">{stats['success_rate']}%</div>
                <div class="stat-label">Tingkat Keberhasilan</div>
            </div>
            
            <div class="stat-card warning">
                <div class="stat-number">{stats['ssl_rate']}%</div>
                <div class="stat-label">SSL Protected</div>
            </div>
            
            <div class="stat-card cache">
                <div class="stat-number">{stats['cache_hit_rate']}%</div>
                <div class="stat-label">Cache Hit Rate</div>
            </div>
            
            <div class="stat-card info">
                <div class="stat-number">{stats['avg_response_time']}s</div>
                <div class="stat-label">Avg Response Time</div>
            </div>
        </div>
        
        <div class="metadata-section">
            <h3>🔧 Informasi Sistem & Optimisasi</h3>
            <div class="metadata-grid">
                <div class="metadata-item">
                    <strong>Tujuan Monitoring:</strong>
                    <span>{system_info.get('purpose', 'Spiritual monitoring')}</span>
                </div>
                <div class="metadata-item">
                    <strong>Jenis Monitoring:</strong>
                    <span>{metadata.get('technical_specs', {}).get('request_method', 'HEAD (minimal impact)')}</span>
                </div>
                <div class="metadata-item">
                    <strong>Rate Limit:</strong>
                    <span>{metadata.get('technical_specs', {}).get('max_requests_per_minute', 2)} req/menit per domain</span>
                </div>
                <div class="metadata-item">
                    <strong>Cache Duration:</strong>
                    <span>{metadata.get('technical_specs', {}).get('cache_duration_minutes', 10)} menit</span>
                </div>
                <div class="metadata-item">
                    <strong>Monitoring Interval:</strong>
                    <span>{metadata.get('technical_specs', {}).get('monitoring_interval_minutes', 8)} menit</span>
                </div>
                <div class="metadata-item">
                    <strong>Filosofi:</strong>
                    <span>{metadata.get('spiritual_context', {}).get('philosophy', 'Monitoring dengan adab')}</span>
                </div>
            </div>
        </div>
        
        <div class="cache-info">
            <h4>📋 Informasi Cache</h4>
            <div class="cache-stats">
                <div class="cache-stat">
                    <strong>{cache_stats.get('total_entries', 0)}</strong><br>
                    <small>Cache Entries</small>
                </div>
                <div class="cache-stat">
                    <strong>{cache_stats.get('cache_duration_minutes', 10)} min</strong><br>
                    <small>Cache Duration</small>
                </div>
                <div class="cache-stat">
                    <strong>{stats['cached_requests']}</strong><br>
                    <small>Cached Requests</small>
                </div>
            </div>
        </div>
        
        <div class="assets-section">
            <h3>🌍 Aset Spiritual Terdaftar</h3>
            <div class="assets-grid">
"""
        
        # Tambahkan kartu aset
        for asset in data['assets'][:12]:  # Batasi 12 aset untuk performa
            status_class = 'active' if asset['status'] == 'active' else 'inactive'
            category_class = asset['category']
            
            html_content += f"""
                <div class="asset-card {category_class}">
                    <div class="asset-name">{asset['original_name']}</div>
                    <div class="asset-spiritual">✨ {asset['spiritual_name']}</div>
                    <div class="asset-status {status_class}">{asset['status'].upper()}</div>
                </div>
"""
        
        html_content += f"""
            </div>
        </div>
        
        <div class="refresh-info">
            <p><strong>🕐 Waktu Saat Ini:</strong> <span id="current-time"></span></p>
            <p><strong>📊 Data Generated:</strong> {data.get('generated_at', datetime.now().isoformat())}</p>
            <p><strong>🔄 Auto Refresh:</strong> Setiap 5 menit (dengan delay 30 detik)</p>
            <p><strong>📋 Cache:</strong> Data di-cache selama 10 menit untuk optimisasi</p>
        </div>
        
        <div class="footer">
            <p>🙏 <strong>{metadata.get('spiritual_context', {}).get('gratitude', 'Alhamdulillahi rabbil alamiin')}</strong></p>
            <p>✨ Sistem monitoring spiritual dengan adab dan optimisasi</p>
            <p>🌸 Generated with love and respect • ZeroLight Orbit System</p>
        </div>
    </div>
</body>
</html>
"""
        
        return html_content
    
    def generate_all_dashboards(self):
        """Generate semua dashboard dengan delay"""
        print("🎨 Generating dashboard optimized...")
        
        # Bersihkan cache expired
        self.cache.clear_expired()
        
        # Ambil data dengan cache
        data = self.get_spiritual_assets_data()
        
        if not data['assets']:
            print("⚠️ Tidak ada data aset untuk dashboard")
            return
        
        # Hitung statistik
        stats = self.calculate_dashboard_stats(data)
        
        # Generate dashboard utama
        print("📊 Generating main dashboard...")
        main_html = self.generate_main_dashboard(data, stats)
        
        main_file = self.dashboard_dir / "index.html"
        with open(main_file, 'w', encoding='utf-8') as f:
            f.write(main_html)
        
        print(f"✅ Dashboard utama: {main_file}")
        
        # Delay sebelum generate individual pages
        time.sleep(self.generation_delay)
        
        # Generate halaman individual (batasi untuk performa)
        print("📄 Generating individual asset pages...")
        
        for i, asset in enumerate(data['assets'][:10]):  # Batasi 10 aset
            if i > 0 and i % 3 == 0:  # Delay setiap 3 aset
                time.sleep(1)
            
            self.generate_individual_asset_page(asset, data['transactions'])
        
        print(f"🎨 Dashboard generation selesai!")
        print(f"📁 Dashboard directory: {self.dashboard_dir}")
        print(f"📋 Cache entries: {len(self.cache.cache)}")
    
    def generate_individual_asset_page(self, asset: Dict, transactions: List[Dict]):
        """Generate halaman individual aset dengan optimisasi"""
        
        # Filter transaksi untuk aset ini
        asset_transactions = [t for t in transactions if t['asset_id'] == asset['id']]
        
        # Statistik aset
        total_checks = len(asset_transactions)
        successful_checks = len([t for t in asset_transactions if t['is_alive']])
        ssl_checks = len([t for t in asset_transactions if t['ssl_valid']])
        cached_checks = len([t for t in asset_transactions if t.get('from_cache', False)])
        
        success_rate = (successful_checks / total_checks * 100) if total_checks > 0 else 0
        
        html_content = f"""
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📊 {asset['spiritual_name']} - Asset Detail</title>
    <meta name="robots" content="noindex, nofollow">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            margin: 0;
            padding: 20px;
        }}
        
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #e0e0e0;
        }}
        
        .back-link {{
            display: inline-block;
            margin-bottom: 20px;
            color: #4299e1;
            text-decoration: none;
            font-weight: bold;
        }}
        
        .back-link:hover {{
            text-decoration: underline;
        }}
        
        .asset-info {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 30px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border-left: 4px solid #4299e1;
        }}
        
        .stat-number {{
            font-size: 1.8em;
            font-weight: bold;
            color: #2d3748;
        }}
        
        .stat-label {{
            color: #718096;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        
        .transactions {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 15px;
        }}
        
        .transaction-item {{
            background: white;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 8px;
            border-left: 4px solid #48bb78;
        }}
        
        .transaction-item.failed {{
            border-left-color: #f56565;
        }}
        
        .transaction-item.cached {{
            border-left-color: #9f7aea;
        }}
    </style>
</head>
<body>
    <div class="container">
        <a href="index.html" class="back-link">← Kembali ke Dashboard</a>
        
        <div class="header">
            <h1>📊 {asset['spiritual_name']}</h1>
            <p>{asset['original_name']}</p>
        </div>
        
        <div class="asset-info">
            <h3>ℹ️ Informasi Aset</h3>
            <p><strong>Kategori:</strong> {asset['category'].title()}</p>
            <p><strong>IP Address:</strong> {asset['ip_address']}</p>
            <p><strong>Status:</strong> {asset['status'].upper()}</p>
            <p><strong>Terdaftar:</strong> {asset['created_at']}</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{total_checks}</div>
                <div class="stat-label">Total Checks</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{success_rate:.1f}%</div>
                <div class="stat-label">Success Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{ssl_checks}</div>
                <div class="stat-label">SSL Checks</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{cached_checks}</div>
                <div class="stat-label">Cached</div>
            </div>
        </div>
        
        <div class="transactions">
            <h3>📋 Transaksi Terbaru (24 jam)</h3>
"""
        
        # Tambahkan transaksi terbaru (batasi 10)
        for transaction in asset_transactions[:10]:
            status_class = 'failed' if not transaction['is_alive'] else ('cached' if transaction.get('from_cache') else '')
            status_icon = '✅' if transaction['is_alive'] else '❌'
            cache_icon = '📋' if transaction.get('from_cache') else '🌐'
            
            html_content += f"""
            <div class="transaction-item {status_class}">
                <strong>{status_icon} {cache_icon} {transaction['timestamp']}</strong><br>
                Status: {transaction['status_code']} | 
                Response: {transaction['response_time']:.3f}s |
                SSL: {'✅' if transaction['ssl_valid'] else '❌'}
            </div>
"""
        
        html_content += """
        </div>
        
        <div style="text-align: center; margin-top: 30px; color: #718096;">
            <p>🙏 Monitoring dengan adab dan hormat</p>
            <p>✨ ZeroLight Orbit Spiritual System</p>
        </div>
    </div>
</body>
</html>
"""
        
        # Simpan file
        filename = f"{asset['spiritual_name'].replace(' ', '_').replace('.', '_')}.html"
        filepath = self.dashboard_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

class SpiritualDashboardServer:
    """Server dashboard spiritual dengan optimisasi"""
    
    def __init__(self, dashboard_dir: str, port: int = 9002):
        self.dashboard_dir = Path(dashboard_dir)
        self.port = port
        self.server = None
        self.server_thread = None
        
    def start_server(self):
        """Start server dashboard"""
        try:
            os.chdir(self.dashboard_dir)
            
            handler = http.server.SimpleHTTPRequestHandler
            self.server = socketserver.TCPServer(("", self.port), handler)
            
            print(f"🌐 Dashboard server starting pada http://localhost:{self.port}")
            
            # Jalankan server di thread terpisah
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            # Delay sebelum buka browser
            time.sleep(2)
            
            # Buka browser
            webbrowser.open(f"http://localhost:{self.port}")
            
            return f"http://localhost:{self.port}"
            
        except Exception as e:
            print(f"⚠️ Error starting server: {e}")
            return None
    
    def stop_server(self):
        """Stop server dashboard"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            print("🛑 Dashboard server stopped")

async def main():
    """Fungsi utama dashboard optimized"""
    print("🌟 ORBIT DASHBOARD REFLECTION - OPTIMIZED")
    print("=" * 60)
    print("🎨 Generating dashboard dengan cache & optimisasi...")
    print("🙏 Bismillahirrahmanirrahim")
    print()
    
    # Inisialisasi generator
    generator = SpiritualDashboardGenerator()
    
    # Generate dashboard
    print("📊 TAHAP 1: Generate Dashboard dengan Cache")
    generator.generate_all_dashboards()
    
    # Start server
    print("\n🌐 TAHAP 2: Start Dashboard Server")
    server = SpiritualDashboardServer(
        dashboard_dir=generator.dashboard_dir,
        port=9002
    )
    
    dashboard_url = server.start_server()
    
    if dashboard_url:
        print(f"✅ Dashboard tersedia di: {dashboard_url}")
        print(f"📁 Dashboard files: {generator.dashboard_dir}")
        print(f"📋 Cache entries: {len(generator.cache.cache)}")
        print(f"⚡ Optimisasi: Cache + Delay + Rate Limiting")
        
        print("\n🔄 Dashboard akan auto-refresh setiap 5 menit")
        print("📋 Data di-cache selama 10 menit untuk optimisasi")
        print("🚦 Rate limiting aktif untuk monitoring yang hormat")
        
        print("\n🙏 Dashboard spiritual optimized siap digunakan!")
        print("✨ Monitoring dengan adab dan berkah")
        
        # Keep server running
        try:
            while True:
                await asyncio.sleep(60)  # Check setiap menit
                
                # Bersihkan cache expired setiap 5 menit
                if datetime.now().minute % 5 == 0:
                    generator.cache.clear_expired()
                    print(f"🧹 Cache cleaned, entries: {len(generator.cache.cache)}")
                
        except KeyboardInterrupt:
            print("\n🌸 Dashboard dihentikan dengan lembut...")
            server.stop_server()
    else:
        print("❌ Gagal memulai dashboard server")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🌸 Dashboard dihentikan dengan lembut...")
    except Exception as e:
        print(f"\n⚠️ Terjadi kesalahan: {e}")