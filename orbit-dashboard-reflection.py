#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌟 ORBIT DASHBOARD REFLECTION
Ladang Berkah Digital - ZeroLight Orbit System
Dashboard & Refleksi HTML untuk Monitoring Status
"""

import asyncio
import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List
from pathlib import Path
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs
import threading
import webbrowser

class SpiritualDashboardGenerator:
    """Generator Dashboard Spiritual untuk monitoring status"""
    
    def __init__(self, db_path: str = "./spiritual_orbit_system.db"):
        self.db_path = db_path
        self.dashboard_dir = Path("./spiritual_dashboard")
        self.dashboard_dir.mkdir(exist_ok=True)
        
    def get_spiritual_assets_data(self) -> Dict:
        """Ambil data aset spiritual dari database"""
        if not os.path.exists(self.db_path):
            return {'assets': [], 'transactions': [], 'reflections': []}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
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
        
        # Ambil data transaksi terbaru
        cursor.execute('''
            SELECT asset_id, transaction_type, status_code, is_alive, ssl_valid, timestamp
            FROM transaction_logs
            ORDER BY timestamp DESC
            LIMIT 100
        ''')
        transactions = []
        for row in cursor.fetchall():
            transactions.append({
                'asset_id': row[0],
                'transaction_type': row[1],
                'status_code': row[2],
                'is_alive': bool(row[3]),
                'ssl_valid': bool(row[4]),
                'timestamp': row[5]
            })
        
        # Ambil refleksi harian
        cursor.execute('''
            SELECT reflection_date, total_assets, active_assets, success_rate, reflection_text
            FROM daily_reflections
            ORDER BY reflection_date DESC
            LIMIT 7
        ''')
        reflections = []
        for row in cursor.fetchall():
            reflections.append({
                'date': row[0],
                'total_assets': row[1],
                'active_assets': row[2],
                'success_rate': row[3],
                'reflection_text': row[4]
            })
        
        conn.close()
        
        return {
            'assets': assets,
            'transactions': transactions,
            'reflections': reflections
        }
    
    def generate_main_dashboard_html(self) -> str:
        """Generate HTML dashboard utama"""
        data = self.get_spiritual_assets_data()
        
        # Hitung statistik
        total_assets = len(data['assets'])
        active_assets = len([a for a in data['assets'] if a['status'] == 'active'])
        planets = len([a for a in data['assets'] if a['category'] == 'planet'])
        wilayah = len([a for a in data['assets'] if a['category'] == 'wilayah'])
        
        # Hitung status terbaru dari transaksi
        asset_status = {}
        for transaction in data['transactions']:
            asset_id = transaction['asset_id']
            if asset_id not in asset_status:
                asset_status[asset_id] = transaction
        
        html_content = f'''<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌟 Orbit ZeroLight Dashboard - Spiritual Monitoring</title>
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
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        
        .stat-card .icon {{
            font-size: 3em;
            margin-bottom: 15px;
        }}
        
        .stat-card .number {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 10px;
        }}
        
        .stat-card .label {{
            font-size: 1.1em;
            color: #666;
        }}
        
        .assets-section {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }}
        
        .section-title {{
            font-size: 1.8em;
            margin-bottom: 20px;
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .assets-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }}
        
        .asset-card {{
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px;
            transition: all 0.3s ease;
        }}
        
        .asset-card.active {{
            border-color: #4CAF50;
            background: #f8fff8;
        }}
        
        .asset-card.inactive {{
            border-color: #f44336;
            background: #fff8f8;
        }}
        
        .asset-name {{
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 5px;
        }}
        
        .spiritual-name {{
            color: #667eea;
            font-style: italic;
            margin-bottom: 10px;
        }}
        
        .asset-details {{
            font-size: 0.9em;
            color: #666;
        }}
        
        .status-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        
        .status-active {{
            background: #4CAF50;
        }}
        
        .status-inactive {{
            background: #f44336;
        }}
        
        .reflection-section {{
            background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            color: #2d3436;
        }}
        
        .reflection-text {{
            white-space: pre-line;
            line-height: 1.6;
            font-size: 1.1em;
        }}
        
        .footer {{
            text-align: center;
            color: white;
            margin-top: 30px;
            opacity: 0.8;
        }}
        
        .refresh-btn {{
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 50px;
            padding: 15px 25px;
            font-size: 1.1em;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }}
        
        .refresh-btn:hover {{
            background: #5a67d8;
            transform: scale(1.05);
        }}
        
        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
            100% {{ transform: scale(1); }}
        }}
        
        .pulse {{
            animation: pulse 2s infinite;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌟 Orbit ZeroLight Dashboard</h1>
            <p>Spiritual Web Monitoring System - Ladang Berkah Digital</p>
            <p>Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="icon">🌍</div>
                <div class="number">{planets}</div>
                <div class="label">Planet Earth</div>
            </div>
            <div class="stat-card">
                <div class="icon">🌊</div>
                <div class="number">{wilayah}</div>
                <div class="label">Wilayah Sea</div>
            </div>
            <div class="stat-card">
                <div class="icon">⚡</div>
                <div class="number">{active_assets}</div>
                <div class="label">Aset Aktif</div>
            </div>
            <div class="stat-card">
                <div class="icon">🔒</div>
                <div class="number">{len([t for t in data['transactions'] if t.get('ssl_valid')])}</div>
                <div class="label">SSL Protected</div>
            </div>
        </div>
        
        <div class="assets-section">
            <h2 class="section-title">🎯 Status Aset Spiritual</h2>
            <div class="assets-grid">'''
        
        # Generate asset cards
        for asset in data['assets']:
            asset_id = asset['id']
            latest_status = asset_status.get(asset_id, {})
            is_alive = latest_status.get('is_alive', False)
            ssl_valid = latest_status.get('ssl_valid', False)
            
            status_class = 'active' if is_alive else 'inactive'
            status_indicator = 'status-active' if is_alive else 'status-inactive'
            ssl_icon = '🔒' if ssl_valid else '🔓'
            
            html_content += f'''
                <div class="asset-card {status_class}">
                    <div class="asset-name">
                        <span class="status-indicator {status_indicator}"></span>
                        {asset['original_name']}
                    </div>
                    <div class="spiritual-name">{asset['spiritual_name']}</div>
                    <div class="asset-details">
                        <strong>Category:</strong> {asset['category']}<br>
                        <strong>IP:</strong> {asset['ip_address']}<br>
                        <strong>SSL:</strong> {ssl_icon} {'Valid' if ssl_valid else 'Invalid'}<br>
                        <strong>Status:</strong> {'🟢 Online' if is_alive else '🔴 Offline'}
                    </div>
                </div>'''
        
        # Add reflection section
        if data['reflections']:
            latest_reflection = data['reflections'][0]
            html_content += f'''
            </div>
        </div>
        
        <div class="reflection-section">
            <h2 class="section-title">🙏 Refleksi Harian Terbaru</h2>
            <div class="reflection-text">{latest_reflection['reflection_text']}</div>
        </div>'''
        
        html_content += '''
        <div class="footer">
            <p>🌟 Barakallahu fiikum - Semoga berkah dan bermanfaat</p>
            <p>Orbit ZeroLight System - Ladang Berkah Digital</p>
        </div>
    </div>
    
    <button class="refresh-btn pulse" onclick="location.reload()">
        🔄 Refresh
    </button>
    
    <script>
        // Auto refresh setiap 30 detik
        setTimeout(() => {
            location.reload();
        }, 30000);
        
        // Animasi untuk kartu aset
        document.querySelectorAll('.asset-card').forEach(card => {
            card.addEventListener('mouseenter', () => {
                card.style.transform = 'scale(1.02)';
            });
            
            card.addEventListener('mouseleave', () => {
                card.style.transform = 'scale(1)';
            });
        });
    </script>
</body>
</html>'''
        
        return html_content
    
    def generate_individual_asset_pages(self):
        """Generate halaman individual untuk setiap aset"""
        data = self.get_spiritual_assets_data()
        
        for asset in data['assets']:
            # Filter transaksi untuk aset ini
            asset_transactions = [t for t in data['transactions'] if t['asset_id'] == asset['id']]
            
            # Generate halaman individual
            html_content = self.generate_asset_page_html(asset, asset_transactions)
            
            # Simpan ke file
            filename = f"{asset['spiritual_name'].replace('.', '_')}.html"
            filepath = self.dashboard_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"📄 Generated: {filename}")
    
    def generate_asset_page_html(self, asset: Dict, transactions: List[Dict]) -> str:
        """Generate HTML untuk halaman aset individual"""
        # Hitung statistik
        total_checks = len(transactions)
        successful_checks = len([t for t in transactions if t['is_alive']])
        success_rate = (successful_checks / total_checks * 100) if total_checks > 0 else 0
        
        html_content = f'''<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌟 {asset['spiritual_name']} - Status Monitor</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 3px solid #667eea;
        }}
        
        .spiritual-name {{
            font-size: 2em;
            color: #667eea;
            margin-bottom: 10px;
        }}
        
        .original-name {{
            font-size: 1.2em;
            color: #666;
            font-style: italic;
        }}
        
        .stats-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-box {{
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }}
        
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .stat-label {{
            color: #666;
            margin-top: 5px;
        }}
        
        .transactions-section {{
            margin-top: 30px;
        }}
        
        .section-title {{
            font-size: 1.5em;
            margin-bottom: 20px;
            color: #333;
        }}
        
        .transaction-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            margin-bottom: 10px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #ddd;
        }}
        
        .transaction-item.success {{
            border-left-color: #4CAF50;
            background: #f8fff8;
        }}
        
        .transaction-item.error {{
            border-left-color: #f44336;
            background: #fff8f8;
        }}
        
        .transaction-time {{
            color: #666;
            font-size: 0.9em;
        }}
        
        .transaction-status {{
            font-weight: bold;
        }}
        
        .success {{
            color: #4CAF50;
        }}
        
        .error {{
            color: #f44336;
        }}
        
        .back-btn {{
            display: inline-block;
            margin-top: 30px;
            padding: 12px 25px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 25px;
            transition: all 0.3s ease;
        }}
        
        .back-btn:hover {{
            background: #5a67d8;
            transform: translateY(-2px);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 class="spiritual-name">🌟 {asset['spiritual_name']}</h1>
            <p class="original-name">{asset['original_name']}</p>
            <p><strong>Category:</strong> {asset['category']} | <strong>IP:</strong> {asset['ip_address']}</p>
        </div>
        
        <div class="stats-row">
            <div class="stat-box">
                <div class="stat-number">{total_checks}</div>
                <div class="stat-label">Total Checks</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{successful_checks}</div>
                <div class="stat-label">Successful</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{success_rate:.1f}%</div>
                <div class="stat-label">Success Rate</div>
            </div>
        </div>
        
        <div class="transactions-section">
            <h2 class="section-title">📊 Recent Activity</h2>'''
        
        # Add transaction history
        for transaction in transactions[:20]:  # Show last 20 transactions
            status_class = 'success' if transaction['is_alive'] else 'error'
            status_text = '✅ Online' if transaction['is_alive'] else '❌ Offline'
            ssl_text = '🔒 SSL' if transaction['ssl_valid'] else '🔓 No SSL'
            
            timestamp = datetime.fromisoformat(transaction['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            
            html_content += f'''
            <div class="transaction-item {status_class}">
                <div>
                    <span class="transaction-status {status_class}">{status_text}</span>
                    <span style="margin-left: 15px;">{ssl_text}</span>
                </div>
                <div class="transaction-time">{timestamp}</div>
            </div>'''
        
        html_content += '''
        </div>
        
        <a href="index.html" class="back-btn">🔙 Back to Dashboard</a>
    </div>
    
    <script>
        // Auto refresh setiap 60 detik
        setTimeout(() => {
            location.reload();
        }, 60000);
    </script>
</body>
</html>'''
        
        return html_content
    
    def save_main_dashboard(self) -> str:
        """Simpan dashboard utama"""
        html_content = self.generate_main_dashboard_html()
        dashboard_file = self.dashboard_dir / "index.html"
        
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"🌟 Dashboard utama tersimpan: {dashboard_file}")
        return str(dashboard_file)
    
    def generate_all_dashboards(self) -> str:
        """Generate semua dashboard dan halaman"""
        print("🎨 Generating spiritual dashboards...")
        
        # Generate dashboard utama
        main_dashboard = self.save_main_dashboard()
        
        # Generate halaman individual
        self.generate_individual_asset_pages()
        
        print(f"✨ Semua dashboard tersimpan di: {self.dashboard_dir}")
        return str(self.dashboard_dir)

class SpiritualDashboardServer:
    """Server untuk melayani dashboard spiritual"""
    
    def __init__(self, dashboard_dir: str, port: int = 9001):
        self.dashboard_dir = Path(dashboard_dir)
        self.port = port
        self.server = None
        
    def start_server(self):
        """Mulai server dashboard"""
        os.chdir(self.dashboard_dir)
        
        handler = http.server.SimpleHTTPRequestHandler
        self.server = socketserver.TCPServer(("", self.port), handler)
        
        print(f"🌐 Dashboard server started at: http://localhost:{self.port}")
        print(f"📁 Serving from: {self.dashboard_dir}")
        
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            print("\n🌸 Server dihentikan dengan lembut...")
            self.server.shutdown()

def main():
    """Fungsi utama untuk dashboard reflection"""
    print("🌟 ORBIT DASHBOARD REFLECTION")
    print("=" * 50)
    print("🎨 Membuat dashboard & refleksi HTML...")
    print()
    
    # Generate dashboard
    generator = SpiritualDashboardGenerator()
    dashboard_dir = generator.generate_all_dashboards()
    
    # Start server
    print("\n🌐 Starting dashboard server...")
    server = SpiritualDashboardServer(dashboard_dir, port=9001)
    
    # Open browser
    try:
        webbrowser.open(f"http://localhost:9001")
    except:
        pass
    
    print("\n✨ Dashboard siap!")
    print("🔗 Akses di: http://localhost:9001")
    print("🌸 Tekan Ctrl+C untuk menghentikan server")
    
    server.start_server()

if __name__ == "__main__":
    main()