#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌟 ORBIT DASHBOARD SYUKUR
Ladang Berkah Digital - ZeroLight Orbit System
Dashboard Spiritual untuk Monitoring Planet & Wilayah
"""

import http.server
import socketserver
import json
import csv
import os
import sqlite3
from datetime import datetime
from urllib.parse import urlparse, parse_qs
import threading
import webbrowser

class DashboardSyukurHandler(http.server.BaseHTTPRequestHandler):
    """Handler untuk Dashboard Syukur Orbit ZeroLight"""
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            self.serve_dashboard()
        elif self.path == '/api/statistik-syukur':
            self.serve_statistik_syukur()
        elif self.path == '/api/planet-earth':
            self.serve_planet_earth()
        elif self.path == '/api/wilayah-sea':
            self.serve_wilayah_sea()
        elif self.path == '/api/refleksi-berkah':
            self.serve_refleksi_berkah()
        else:
            self.send_error(404)
    
    def serve_dashboard(self):
        """Serve main dashboard HTML"""
        html_content = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌟 Orbit Dashboard Syukur - ZeroLight System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            transition: transform 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
        }
        
        .stat-icon {
            font-size: 3em;
            margin-bottom: 15px;
        }
        
        .stat-number {
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 10px;
        }
        
        .stat-label {
            font-size: 1.1em;
            color: #666;
            font-weight: 500;
        }
        
        .content-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .content-card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .content-card h3 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.3em;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .planet-item, .wilayah-item {
            background: #f8f9ff;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
            border-left: 4px solid #667eea;
        }
        
        .planet-name, .wilayah-name {
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }
        
        .planet-info, .wilayah-info {
            font-size: 0.9em;
            color: #666;
        }
        
        .status-active {
            color: #28a745;
            font-weight: bold;
        }
        
        .status-ssl {
            color: #17a2b8;
            font-weight: bold;
        }
        
        .refleksi-card {
            background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            color: #2d3436;
        }
        
        .refleksi-card h3 {
            margin-bottom: 15px;
            font-size: 1.5em;
        }
        
        .refleksi-text {
            font-style: italic;
            line-height: 1.6;
            margin-bottom: 15px;
        }
        
        .berkah-footer {
            text-align: center;
            color: white;
            margin-top: 30px;
            opacity: 0.8;
        }
        
        @media (max-width: 768px) {
            .content-grid {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2em;
            }
        }
        
        .loading {
            text-align: center;
            color: #666;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌟 Orbit Dashboard Syukur</h1>
            <p>Ladang Berkah Digital - ZeroLight System</p>
            <p>Monitoring Planet Earth & Wilayah Sea dengan Penuh Syukur</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-icon">🌍</div>
                <div class="stat-number" id="total-planet">-</div>
                <div class="stat-label">Planet Earth</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-icon">🌊</div>
                <div class="stat-number" id="total-wilayah">-</div>
                <div class="stat-label">Wilayah Sea</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-icon">⚡</div>
                <div class="stat-number" id="total-aktif">-</div>
                <div class="stat-label">Total Aktif</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-icon">🔒</div>
                <div class="stat-number" id="ssl-terlindungi">-</div>
                <div class="stat-label">SSL Terlindungi</div>
            </div>
        </div>
        
        <div class="content-grid">
            <div class="content-card">
                <h3>🌍 Planet Earth yang Ditemukan</h3>
                <div id="planet-list" class="loading">Memuat data planet...</div>
            </div>
            
            <div class="content-card">
                <h3>🌊 Wilayah Sea yang Ditemukan</h3>
                <div id="wilayah-list" class="loading">Memuat data wilayah...</div>
            </div>
        </div>
        
        <div class="refleksi-card">
            <h3>🙏 Refleksi Syukur</h3>
            <div class="refleksi-text" id="refleksi-content">
                Alhamdulillahi rabbil alamiin atas segala penemuan yang telah diberikan 
                dalam perjalanan spiritual ini. Semoga setiap planet dan wilayah yang ditemukan 
                membawa berkah dan manfaat bagi umat.
            </div>
            <div><strong>✨ Berkah melimpah untuk semua ✨</strong></div>
        </div>
        
        <div class="berkah-footer">
            <p>🌟 Ladang Berkah Digital - ZeroLight Orbit System 🌟</p>
            <p>Dibuat dengan penuh syukur dan keberkahan</p>
        </div>
    </div>
    
    <script>
        // Load statistik syukur
        async function loadStatistikSyukur() {
            try {
                const response = await fetch('/api/statistik-syukur');
                const data = await response.json();
                
                document.getElementById('total-planet').textContent = data.total_planet || 0;
                document.getElementById('total-wilayah').textContent = data.total_wilayah || 0;
                document.getElementById('total-aktif').textContent = data.total_aktif || 0;
                document.getElementById('ssl-terlindungi').textContent = data.ssl_terlindungi || 0;
            } catch (error) {
                console.error('Error loading statistik:', error);
            }
        }
        
        // Load planet earth data
        async function loadPlanetEarth() {
            try {
                const response = await fetch('/api/planet-earth');
                const planets = await response.json();
                
                const planetList = document.getElementById('planet-list');
                if (planets.length === 0) {
                    planetList.innerHTML = '<p class="loading">Belum ada planet yang ditemukan</p>';
                    return;
                }
                
                planetList.innerHTML = planets.map(planet => `
                    <div class="planet-item">
                        <div class="planet-name">🌍 ${planet.nama}</div>
                        <div class="planet-info">
                            IP: ${planet.ip || 'N/A'} | 
                            Status: <span class="status-active">${planet.aktif ? 'Aktif' : 'Tidak Aktif'}</span> | 
                            SSL: <span class="status-ssl">${planet.ssl ? 'Terlindungi' : 'Tidak Terlindungi'}</span>
                            ${planet.registrar ? `<br>Registrar: ${planet.registrar}` : ''}
                        </div>
                    </div>
                `).join('');
            } catch (error) {
                console.error('Error loading planet data:', error);
                document.getElementById('planet-list').innerHTML = '<p class="loading">Error memuat data planet</p>';
            }
        }
        
        // Load wilayah sea data
        async function loadWilayahSea() {
            try {
                const response = await fetch('/api/wilayah-sea');
                const wilayah = await response.json();
                
                const wilayahList = document.getElementById('wilayah-list');
                if (wilayah.length === 0) {
                    wilayahList.innerHTML = '<p class="loading">Belum ada wilayah yang ditemukan</p>';
                    return;
                }
                
                // Tampilkan hanya 10 wilayah pertama untuk menghindari overflow
                const displayWilayah = wilayah.slice(0, 10);
                
                wilayahList.innerHTML = displayWilayah.map(w => `
                    <div class="wilayah-item">
                        <div class="wilayah-name">🌊 ${w.nama}</div>
                        <div class="wilayah-info">
                            IP: ${w.ip || 'N/A'} | 
                            HTTP: <span class="${w.http ? 'status-active' : ''}">${w.http ? 'Ya' : 'Tidak'}</span> | 
                            HTTPS: <span class="${w.https ? 'status-ssl' : ''}">${w.https ? 'Ya' : 'Tidak'}</span>
                        </div>
                    </div>
                `).join('');
                
                if (wilayah.length > 10) {
                    wilayahList.innerHTML += `<p style="text-align: center; margin-top: 15px; color: #666;">... dan ${wilayah.length - 10} wilayah lainnya</p>`;
                }
            } catch (error) {
                console.error('Error loading wilayah data:', error);
                document.getElementById('wilayah-list').innerHTML = '<p class="loading">Error memuat data wilayah</p>';
            }
        }
        
        // Load refleksi berkah
        async function loadRefleksiBerkah() {
            try {
                const response = await fetch('/api/refleksi-berkah');
                const refleksi = await response.json();
                
                if (refleksi.content) {
                    document.getElementById('refleksi-content').textContent = refleksi.content;
                }
            } catch (error) {
                console.error('Error loading refleksi:', error);
            }
        }
        
        // Initialize dashboard
        function initDashboard() {
            loadStatistikSyukur();
            loadPlanetEarth();
            loadWilayahSea();
            loadRefleksiBerkah();
            
            // Auto refresh setiap 30 detik
            setInterval(() => {
                loadStatistikSyukur();
                loadPlanetEarth();
                loadWilayahSea();
            }, 30000);
        }
        
        // Start dashboard when page loads
        document.addEventListener('DOMContentLoaded', initDashboard);
    </script>
</body>
</html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html_content.encode('utf-8'))
    
    def serve_statistik_syukur(self):
        """Serve statistics API"""
        try:
            stats = self.get_statistics_from_csv()
            self.send_json_response(stats)
        except Exception as e:
            self.send_json_response({'error': str(e)}, 500)
    
    def serve_planet_earth(self):
        """Serve planet earth data"""
        try:
            planets = self.get_planets_from_csv()
            self.send_json_response(planets)
        except Exception as e:
            self.send_json_response({'error': str(e)}, 500)
    
    def serve_wilayah_sea(self):
        """Serve wilayah sea data"""
        try:
            wilayah = self.get_wilayah_from_csv()
            self.send_json_response(wilayah)
        except Exception as e:
            self.send_json_response({'error': str(e)}, 500)
    
    def serve_refleksi_berkah(self):
        """Serve refleksi berkah content"""
        try:
            refleksi_path = "./log/sea_refleksi.txt"
            content = "Alhamdulillahi rabbil alamiin atas segala penemuan yang telah diberikan dalam perjalanan spiritual ini."
            
            if os.path.exists(refleksi_path):
                with open(refleksi_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    # Ambil bagian syukur dan keberkahan
                    syukur_lines = [line.strip() for line in lines if 'syukur' in line.lower() or 'berkah' in line.lower()]
                    if syukur_lines:
                        content = ' '.join(syukur_lines[:3])  # Ambil 3 baris pertama
            
            self.send_json_response({'content': content})
        except Exception as e:
            self.send_json_response({'error': str(e)}, 500)
    
    def get_statistics_from_csv(self):
        """Get statistics from CSV log file"""
        csv_path = "./log/sea_water.csv"
        stats = {
            'total_planet': 0,
            'total_wilayah': 0,
            'total_aktif': 0,
            'ssl_terlindungi': 0
        }
        
        if not os.path.exists(csv_path):
            return stats
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                planet_count = 0
                wilayah_count = 0
                aktif_count = 0
                ssl_count = 0
                
                for row in reader:
                    if row['Jenis'] == 'Planet_Earth':
                        planet_count += 1
                        if row['Status_Hidup'] == 'True':
                            aktif_count += 1
                        if row['SSL_Terlindungi'] == 'True':
                            ssl_count += 1
                    elif row['Jenis'] == 'Wilayah_Sea':
                        wilayah_count += 1
                        if row['Status_Hidup'] == 'True':
                            aktif_count += 1
                
                stats.update({
                    'total_planet': planet_count,
                    'total_wilayah': wilayah_count,
                    'total_aktif': aktif_count,
                    'ssl_terlindungi': ssl_count
                })
        except Exception as e:
            print(f"Error reading CSV: {e}")
        
        return stats
    
    def get_planets_from_csv(self):
        """Get planet data from CSV"""
        csv_path = "./log/sea_water.csv"
        planets = []
        
        if not os.path.exists(csv_path):
            return planets
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row in reader:
                    if row['Jenis'] == 'Planet_Earth':
                        planets.append({
                            'nama': row['Nama'],
                            'ip': row['IP_Address'],
                            'aktif': row['Status_Hidup'] == 'True',
                            'ssl': row['SSL_Terlindungi'] == 'True',
                            'registrar': row['Registrar'] if row['Registrar'] else None
                        })
        except Exception as e:
            print(f"Error reading planets: {e}")
        
        return planets
    
    def get_wilayah_from_csv(self):
        """Get wilayah data from CSV"""
        csv_path = "./log/sea_water.csv"
        wilayah = []
        
        if not os.path.exists(csv_path):
            return wilayah
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row in reader:
                    if row['Jenis'] == 'Wilayah_Sea':
                        wilayah.append({
                            'nama': row['Nama'],
                            'ip': row['IP_Address'],
                            'http': row['Status_Hidup'] == 'True',
                            'https': row['SSL_Terlindungi'] == 'True',
                            'planet_induk': row['Registrar']  # Planet induk disimpan di kolom Registrar
                        })
        except Exception as e:
            print(f"Error reading wilayah: {e}")
        
        return wilayah
    
    def send_json_response(self, data, status_code=200):
        """Send JSON response"""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))
    
    def log_message(self, format, *args):
        """Override to reduce log noise"""
        pass

def start_dashboard_syukur(port=8083):
    """Start the spiritual dashboard"""
    try:
        with socketserver.TCPServer(("", port), DashboardSyukurHandler) as httpd:
            print(f"🌟 Dashboard Syukur Orbit ZeroLight dimulai di port {port}")
            print(f"🌐 Akses dashboard di: http://localhost:{port}")
            print("🙏 Dengan penuh syukur dan keberkahan...")
            print("✨ Tekan Ctrl+C untuk menghentikan dengan lembut")
            
            # Buka browser otomatis
            threading.Timer(1.0, lambda: webbrowser.open(f'http://localhost:{port}')).start()
            
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🌸 Dashboard dihentikan dengan lembut...")
        print("✨ Terima kasih atas keberkahan yang telah diberikan")
    except Exception as e:
        print(f"⚠️ Terjadi kesalahan: {e}")

if __name__ == "__main__":
    start_dashboard_syukur(9000)