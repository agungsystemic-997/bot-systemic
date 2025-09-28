#!/usr/bin/env python3
"""
ORBIT PROGRESS MONITOR - GALACTIC EXPANSION DASHBOARD
Monitor real-time progress menuju target 1500 web assets
"""

import sqlite3
import time
import json
from datetime import datetime, timedelta
from flask import Flask, render_template_string, jsonify
import threading

app = Flask(__name__)

class ProgressMonitor:
    def __init__(self):
        self.db_path = 'spiritual_web_discovery.db'
        self.target_assets = 1500
        self.start_time = datetime.now()
        self.progress_history = []
        
    def get_current_stats(self):
        """Get current statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Count domains
            cursor.execute('SELECT COUNT(*) FROM domains')
            domain_count = cursor.fetchone()[0]
            
            # Count subdomains
            cursor.execute('SELECT COUNT(*) FROM subdomains')
            subdomain_count = cursor.fetchone()[0]
            
            total_assets = domain_count + subdomain_count
            
            # Recent discoveries (last hour)
            one_hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
            cursor.execute('SELECT COUNT(*) FROM domains WHERE discovered_at > ?', (one_hour_ago,))
            recent_domains = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM subdomains WHERE discovered_at > ?', (one_hour_ago,))
            recent_subdomains = cursor.fetchone()[0]
            
            conn.close()
            
            progress_percentage = (total_assets / self.target_assets) * 100
            remaining = self.target_assets - total_assets
            
            # Calculate ETA
            elapsed_time = datetime.now() - self.start_time
            if total_assets > 66:  # Initial count
                rate = (total_assets - 66) / elapsed_time.total_seconds() * 3600  # per hour
                eta_hours = remaining / rate if rate > 0 else float('inf')
            else:
                eta_hours = float('inf')
                
            stats = {
                'total_assets': total_assets,
                'domain_count': domain_count,
                'subdomain_count': subdomain_count,
                'target_assets': self.target_assets,
                'progress_percentage': round(progress_percentage, 2),
                'remaining': remaining,
                'recent_domains': recent_domains,
                'recent_subdomains': recent_subdomains,
                'eta_hours': round(eta_hours, 2) if eta_hours != float('inf') else 'Unknown',
                'elapsed_time': str(elapsed_time).split('.')[0],
                'timestamp': datetime.now().isoformat()
            }
            
            # Add to history
            self.progress_history.append({
                'timestamp': datetime.now().isoformat(),
                'total_assets': total_assets,
                'progress_percentage': progress_percentage
            })
            
            # Keep only last 100 entries
            if len(self.progress_history) > 100:
                self.progress_history = self.progress_history[-100:]
                
            return stats
            
        except Exception as e:
            return {'error': str(e)}

monitor = ProgressMonitor()

@app.route('/')
def dashboard():
    """Main dashboard"""
    template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>🌌 Galactic Expansion Progress Monitor</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                margin: 0;
                padding: 20px;
                min-height: 100vh;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
                padding: 20px;
                background: rgba(255,255,255,0.1);
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            .stat-card {
                background: rgba(255,255,255,0.15);
                padding: 20px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255,255,255,0.2);
                text-align: center;
            }
            .stat-value {
                font-size: 2.5em;
                font-weight: bold;
                margin: 10px 0;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            .stat-label {
                font-size: 1.1em;
                opacity: 0.9;
            }
            .progress-container {
                background: rgba(255,255,255,0.1);
                padding: 30px;
                border-radius: 15px;
                margin-bottom: 30px;
            }
            .progress-bar {
                width: 100%;
                height: 30px;
                background: rgba(255,255,255,0.2);
                border-radius: 15px;
                overflow: hidden;
                margin: 20px 0;
            }
            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, #4CAF50, #8BC34A);
                border-radius: 15px;
                transition: width 0.5s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
            }
            .spiritual-quote {
                background: rgba(255,255,255,0.1);
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                font-style: italic;
                margin-top: 20px;
            }
            .refresh-btn {
                background: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 25px;
                cursor: pointer;
                font-size: 1em;
                margin: 10px;
            }
            .refresh-btn:hover {
                background: #45a049;
            }
            .galaxy-animation {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                z-index: -1;
            }
            .star {
                position: absolute;
                background: white;
                border-radius: 50%;
                animation: twinkle 2s infinite;
            }
            @keyframes twinkle {
                0%, 100% { opacity: 0.3; }
                50% { opacity: 1; }
            }
        </style>
    </head>
    <body>
        <div class="galaxy-animation" id="galaxy"></div>
        
        <div class="container">
            <div class="header">
                <h1>🌌 Galactic Expansion Progress Monitor</h1>
                <p>Monitoring perjalanan menuju 1500 web assets di galaksi digital</p>
                <button class="refresh-btn" onclick="refreshData()">🔄 Refresh Data</button>
                <span id="lastUpdate"></span>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value" id="totalAssets">-</div>
                    <div class="stat-label">Total Web Assets</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="domainCount">-</div>
                    <div class="stat-label">Domains</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="subdomainCount">-</div>
                    <div class="stat-label">Subdomains</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="remaining">-</div>
                    <div class="stat-label">Remaining to Target</div>
                </div>
            </div>
            
            <div class="progress-container">
                <h2>🎯 Progress to 1500 Assets</h2>
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill" style="width: 0%">0%</div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                    <span>Current: <span id="currentProgress">0</span></span>
                    <span>Target: 1500</span>
                </div>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value" id="recentDiscoveries">-</div>
                    <div class="stat-label">Recent Discoveries (1h)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="eta">-</div>
                    <div class="stat-label">ETA (hours)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="elapsedTime">-</div>
                    <div class="stat-label">Elapsed Time</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="progressPercentage">-</div>
                    <div class="stat-label">Progress %</div>
                </div>
            </div>
            
            <div class="spiritual-quote">
                <p>🕌 "Dan langit itu Kami bangun dengan kekuasaan (Kami) dan sesungguhnya Kami benar-benar meluaskannya."</p>
                <p><strong>- QS. Adz-Dzariyat: 47</strong></p>
                <p>Setiap ekspansi digital mencerminkan keluasan ciptaan Allah di alam semesta yang tak terbatas.</p>
            </div>
        </div>
        
        <script>
            // Create galaxy animation
            function createGalaxy() {
                const galaxy = document.getElementById('galaxy');
                for (let i = 0; i < 100; i++) {
                    const star = document.createElement('div');
                    star.className = 'star';
                    star.style.left = Math.random() * 100 + '%';
                    star.style.top = Math.random() * 100 + '%';
                    star.style.width = Math.random() * 3 + 1 + 'px';
                    star.style.height = star.style.width;
                    star.style.animationDelay = Math.random() * 2 + 's';
                    galaxy.appendChild(star);
                }
            }
            
            function refreshData() {
                fetch('/api/stats')
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            console.error('Error:', data.error);
                            return;
                        }
                        
                        document.getElementById('totalAssets').textContent = data.total_assets;
                        document.getElementById('domainCount').textContent = data.domain_count;
                        document.getElementById('subdomainCount').textContent = data.subdomain_count;
                        document.getElementById('remaining').textContent = data.remaining;
                        document.getElementById('currentProgress').textContent = data.total_assets;
                        document.getElementById('progressPercentage').textContent = data.progress_percentage + '%';
                        document.getElementById('recentDiscoveries').textContent = data.recent_domains + data.recent_subdomains;
                        document.getElementById('eta').textContent = data.eta_hours;
                        document.getElementById('elapsedTime').textContent = data.elapsed_time;
                        
                        // Update progress bar
                        const progressFill = document.getElementById('progressFill');
                        progressFill.style.width = data.progress_percentage + '%';
                        progressFill.textContent = data.progress_percentage + '%';
                        
                        document.getElementById('lastUpdate').textContent = 'Last updated: ' + new Date().toLocaleTimeString();
                    })
                    .catch(error => {
                        console.error('Error fetching data:', error);
                    });
            }
            
            // Initialize
            createGalaxy();
            refreshData();
            
            // Auto refresh every 10 seconds
            setInterval(refreshData, 10000);
        </script>
    </body>
    </html>
    '''
    return template

@app.route('/api/stats')
def api_stats():
    """API endpoint for statistics"""
    return jsonify(monitor.get_current_stats())

if __name__ == '__main__':
    print("🌌 Starting Galactic Expansion Progress Monitor...")
    print("📊 Dashboard available at: http://localhost:9002")
    app.run(host='0.0.0.0', port=9002, debug=False)