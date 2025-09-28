#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPIRITUAL WEB DASHBOARD ENHANCED
Ladang Berkah Digital - ZeroLight Orbit System
Enhanced Web Dashboard with Comprehensive Discovery Integration
"""

import http.server
import socketserver
import json
import sqlite3
import threading
import time
from datetime import datetime
from typing import Dict, List, Any
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from spiritual_web_integration import SpiritualWebIntegration
except ImportError:
    print("⚠️ Could not import SpiritualWebIntegration, using fallback mode")
    SpiritualWebIntegration = None

class EnhancedWebDashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, web_integration=None, **kwargs):
        self.web_integration = web_integration
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        if self.path == '/':
            self.send_enhanced_dashboard_html()
        elif self.path.startswith('/api/scan/'):
            domain = self.path.split('/api/scan/')[1]
            self.send_comprehensive_scan(domain)
        elif self.path.startswith('/api/batch-scan'):
            self.send_batch_scan_form()
        elif self.path == '/api/stats':
            self.send_discovery_stats()
        elif self.path == '/api/report':
            self.send_full_report()
        elif self.path == '/api/database':
            self.send_database_contents()
        else:
            super().do_GET()
    
    def do_POST(self):
        if self.path == '/api/batch-scan':
            self.handle_batch_scan()
        else:
            self.send_error(404)
    
    def send_enhanced_dashboard_html(self):
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>🌟 Spiritual Web Discovery Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }
        
        .container { 
            max-width: 1400px; 
            margin: 0 auto; 
            padding: 20px;
        }
        
        .header { 
            text-align: center; 
            margin-bottom: 30px;
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: rgba(255,255,255,0.15);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .stat-number {
            font-size: 2.5em;
            font-weight: bold;
            color: #4CAF50;
            display: block;
        }
        
        .stat-label {
            font-size: 0.9em;
            opacity: 0.8;
            margin-top: 5px;
        }
        
        .controls {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .control-panel {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        
        input[type="text"], textarea {
            width: 100%;
            padding: 10px;
            border: none;
            border-radius: 5px;
            background: rgba(255,255,255,0.9);
            color: #333;
        }
        
        button {
            padding: 12px 24px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: background 0.3s;
        }
        
        button:hover { background: #45a049; }
        button:disabled { background: #666; cursor: not-allowed; }
        
        .results {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
            margin-top: 20px;
        }
        
        .domain-card {
            background: rgba(255,255,255,0.1);
            margin: 10px 0;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #4CAF50;
        }
        
        .subdomain-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }
        
        .subdomain-item {
            background: rgba(255,255,255,0.1);
            padding: 10px;
            border-radius: 5px;
            font-size: 0.9em;
        }
        
        .status-good { color: #4CAF50; }
        .status-bad { color: #f44336; }
        .status-warning { color: #ff9800; }
        
        .loading {
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 4px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top: 4px solid #4CAF50;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .tabs {
            display: flex;
            margin-bottom: 20px;
        }
        
        .tab {
            padding: 10px 20px;
            background: rgba(255,255,255,0.1);
            border: none;
            color: white;
            cursor: pointer;
            border-radius: 5px 5px 0 0;
            margin-right: 5px;
        }
        
        .tab.active {
            background: rgba(255,255,255,0.2);
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        @media (max-width: 768px) {
            .controls {
                grid-template-columns: 1fr;
            }
            .stats-grid {
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌟 Spiritual Web Discovery Dashboard 🌟</h1>
            <p>Ladang Berkah Digital - ZeroLight Orbit System</p>
            <p>Comprehensive Web, Domain & Subdomain Discovery Platform</p>
        </div>
        
        <div class="stats-grid" id="statsGrid">
            <div class="stat-card">
                <span class="stat-number" id="totalDomains">0</span>
                <span class="stat-label">Total Domains</span>
            </div>
            <div class="stat-card">
                <span class="stat-number" id="totalSubdomains">0</span>
                <span class="stat-label">Total Subdomains</span>
            </div>
            <div class="stat-card">
                <span class="stat-number" id="activeDomains">0</span>
                <span class="stat-label">Active Domains</span>
            </div>
            <div class="stat-card">
                <span class="stat-number" id="activeSubdomains">0</span>
                <span class="stat-label">Active Subdomains</span>
            </div>
            <div class="stat-card">
                <span class="stat-number" id="sslEnabled">0</span>
                <span class="stat-label">SSL Enabled</span>
            </div>
        </div>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('single')">Single Domain</button>
            <button class="tab" onclick="showTab('batch')">Batch Scan</button>
            <button class="tab" onclick="showTab('database')">Database View</button>
            <button class="tab" onclick="showTab('reports')">Reports</button>
        </div>
        
        <div id="singleTab" class="tab-content active">
            <div class="controls">
                <div class="control-panel">
                    <h3>🔍 Single Domain Scan</h3>
                    <div class="form-group">
                        <label for="singleDomain">Domain Name:</label>
                        <input type="text" id="singleDomain" placeholder="e.g., google.com" />
                    </div>
                    <button onclick="scanSingleDomain()">Comprehensive Scan</button>
                </div>
                
                <div class="control-panel">
                    <h3>⚡ Quick Actions</h3>
                    <button onclick="loadStats()" style="margin: 5px;">Refresh Stats</button>
                    <button onclick="loadDatabaseView()" style="margin: 5px;">View Database</button>
                    <button onclick="generateReport()" style="margin: 5px;">Generate Report</button>
                    <button onclick="clearResults()" style="margin: 5px; background: #f44336;">Clear Results</button>
                </div>
            </div>
        </div>
        
        <div id="batchTab" class="tab-content">
            <div class="control-panel">
                <h3>🚀 Batch Domain Scan</h3>
                <div class="form-group">
                    <label for="batchDomains">Domain List (one per line):</label>
                    <textarea id="batchDomains" rows="8" placeholder="google.com&#10;github.com&#10;stackoverflow.com&#10;python.org"></textarea>
                </div>
                <button onclick="scanBatchDomains()">Start Batch Scan</button>
            </div>
        </div>
        
        <div id="databaseTab" class="tab-content">
            <div class="control-panel">
                <h3>💾 Database Contents</h3>
                <button onclick="loadDatabaseView()">Load Database View</button>
            </div>
        </div>
        
        <div id="reportsTab" class="tab-content">
            <div class="control-panel">
                <h3>📊 Comprehensive Reports</h3>
                <button onclick="generateReport()">Generate Full Report</button>
            </div>
        </div>
        
        <div id="results" class="results" style="display: none;"></div>
    </div>

    <script>
        let currentStats = {};
        
        // Load initial stats
        loadStats();
        
        function showTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab
            document.getElementById(tabName + 'Tab').classList.add('active');
            event.target.classList.add('active');
        }
        
        function loadStats() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    currentStats = data;
                    updateStatsDisplay(data);
                })
                .catch(error => console.error('Error loading stats:', error));
        }
        
        function updateStatsDisplay(stats) {
            document.getElementById('totalDomains').textContent = stats.total_domains || 0;
            document.getElementById('totalSubdomains').textContent = stats.total_subdomains || 0;
            document.getElementById('activeDomains').textContent = stats.active_domains || 0;
            document.getElementById('activeSubdomains').textContent = stats.active_subdomains || 0;
            document.getElementById('sslEnabled').textContent = stats.ssl_enabled || 0;
        }
        
        function scanSingleDomain() {
            const domain = document.getElementById('singleDomain').value.trim();
            if (!domain) {
                alert('Please enter a domain name');
                return;
            }
            
            showLoading('Scanning domain: ' + domain + '...');
            
            fetch('/api/scan/' + domain)
                .then(response => response.json())
                .then(data => {
                    displayScanResults(data);
                    loadStats(); // Refresh stats
                })
                .catch(error => {
                    showError('Error scanning domain: ' + error);
                });
        }
        
        function scanBatchDomains() {
            const domainsText = document.getElementById('batchDomains').value.trim();
            if (!domainsText) {
                alert('Please enter domain names');
                return;
            }
            
            const domains = domainsText.split('\\n').filter(d => d.trim());
            showLoading('Starting batch scan for ' + domains.length + ' domains...');
            
            // Send POST request for batch scan
            fetch('/api/batch-scan', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({domains: domains})
            })
            .then(response => response.json())
            .then(data => {
                displayBatchResults(data);
                loadStats(); // Refresh stats
            })
            .catch(error => {
                showError('Error in batch scan: ' + error);
            });
        }
        
        function loadDatabaseView() {
            showLoading('Loading database contents...');
            
            fetch('/api/database')
                .then(response => response.json())
                .then(data => {
                    displayDatabaseView(data);
                })
                .catch(error => {
                    showError('Error loading database: ' + error);
                });
        }
        
        function generateReport() {
            showLoading('Generating comprehensive report...');
            
            fetch('/api/report')
                .then(response => response.json())
                .then(data => {
                    displayReport(data);
                })
                .catch(error => {
                    showError('Error generating report: ' + error);
                });
        }
        
        function showLoading(message) {
            const results = document.getElementById('results');
            results.style.display = 'block';
            results.innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <p>${message}</p>
                </div>
            `;
        }
        
        function showError(message) {
            const results = document.getElementById('results');
            results.style.display = 'block';
            results.innerHTML = `
                <div class="status-bad">
                    <h3>❌ Error</h3>
                    <p>${message}</p>
                </div>
            `;
        }
        
        function displayScanResults(data) {
            let html = '<h2>🔍 Scan Results</h2>';
            
            if (data.domain_info) {
                html += '<div class="domain-card">';
                html += '<h3>🌐 ' + data.domain_info.domain + '</h3>';
                html += '<p><strong>IP:</strong> ' + data.domain_info.ip_address + '</p>';
                html += '<p><strong>HTTP Status:</strong> ' + (data.domain_info.http_status || 'N/A') + '</p>';
                html += '<p><strong>HTTPS Status:</strong> ' + (data.domain_info.https_status || 'N/A') + '</p>';
                html += '<p><strong>SSL Valid:</strong> ' + (data.domain_info.ssl_valid ? '✅ Yes' : '❌ No') + '</p>';
                if (data.domain_info.registrar) {
                    html += '<p><strong>Registrar:</strong> ' + data.domain_info.registrar + '</p>';
                }
                html += '</div>';
            }
            
            if (data.subdomains && data.subdomains.length > 0) {
                html += '<h3>🔍 Discovered Subdomains (' + data.subdomains.length + ')</h3>';
                html += '<div class="subdomain-grid">';
                data.subdomains.forEach(sub => {
                    const statusClass = (sub.http_accessible || sub.https_accessible) ? 'status-good' : 'status-bad';
                    html += '<div class="subdomain-item ' + statusClass + '">';
                    html += '<strong>' + sub.subdomain + '</strong><br>';
                    html += 'IP: ' + sub.ip_address + '<br>';
                    html += 'HTTP: ' + (sub.http_accessible ? '✅' : '❌') + ' ';
                    html += 'HTTPS: ' + (sub.https_accessible ? '✅' : '❌');
                    html += '</div>';
                });
                html += '</div>';
            }
            
            document.getElementById('results').innerHTML = html;
            document.getElementById('results').style.display = 'block';
        }
        
        function displayBatchResults(data) {
            let html = '<h2>🚀 Batch Scan Results</h2>';
            html += '<p><strong>Scan Duration:</strong> ' + (data.stats?.scan_duration || 0).toFixed(2) + ' seconds</p>';
            
            if (data.domains) {
                html += '<h3>📊 Domain Summary</h3>';
                Object.keys(data.domains).forEach(domain => {
                    const domainData = data.domains[domain];
                    html += '<div class="domain-card">';
                    html += '<h4>🌐 ' + domain + '</h4>';
                    html += '<p>IP: ' + domainData.ip_address + ' | ';
                    html += 'SSL: ' + (domainData.ssl_valid ? '✅' : '❌') + ' | ';
                    html += 'Subdomains: ' + (data.subdomains[domain]?.length || 0) + '</p>';
                    html += '</div>';
                });
            }
            
            document.getElementById('results').innerHTML = html;
            document.getElementById('results').style.display = 'block';
        }
        
        function displayDatabaseView(data) {
            let html = '<h2>💾 Database Contents</h2>';
            
            if (data.domains && data.domains.length > 0) {
                html += '<h3>📊 Domains (' + data.domains.length + ')</h3>';
                data.domains.forEach(domain => {
                    html += '<div class="domain-card">';
                    html += '<h4>' + domain.domain + '</h4>';
                    html += '<p>IP: ' + domain.ip_address + ' | Last Checked: ' + domain.last_checked + '</p>';
                    html += '</div>';
                });
            }
            
            if (data.subdomains && data.subdomains.length > 0) {
                html += '<h3>🔍 Subdomains (' + data.subdomains.length + ')</h3>';
                html += '<div class="subdomain-grid">';
                data.subdomains.forEach(sub => {
                    html += '<div class="subdomain-item">';
                    html += '<strong>' + sub.subdomain + '</strong><br>';
                    html += 'Parent: ' + sub.parent_domain + '<br>';
                    html += 'IP: ' + sub.ip_address;
                    html += '</div>';
                });
                html += '</div>';
            }
            
            document.getElementById('results').innerHTML = html;
            document.getElementById('results').style.display = 'block';
        }
        
        function displayReport(data) {
            let html = '<h2>📊 Comprehensive Report</h2>';
            
            if (data.summary) {
                html += '<div class="stats-grid">';
                html += '<div class="stat-card"><span class="stat-number">' + data.summary.total_domains_checked + '</span><span class="stat-label">Total Domains</span></div>';
                html += '<div class="stat-card"><span class="stat-number">' + data.summary.total_subdomains_found + '</span><span class="stat-label">Total Subdomains</span></div>';
                html += '<div class="stat-card"><span class="stat-number">' + data.summary.active_domains + '</span><span class="stat-label">Active Domains</span></div>';
                html += '<div class="stat-card"><span class="stat-number">' + data.summary.ssl_enabled_domains + '</span><span class="stat-label">SSL Enabled</span></div>';
                html += '</div>';
            }
            
            if (data.top_registrars) {
                html += '<h3>🏢 Top Registrars</h3>';
                Object.keys(data.top_registrars).forEach(registrar => {
                    html += '<p>• ' + registrar + ': ' + data.top_registrars[registrar] + ' domains</p>';
                });
            }
            
            document.getElementById('results').innerHTML = html;
            document.getElementById('results').style.display = 'block';
        }
        
        function clearResults() {
            document.getElementById('results').style.display = 'none';
        }
        
        // Allow Enter key in single domain input
        document.getElementById('singleDomain').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                scanSingleDomain();
            }
        });
        
        // Auto-refresh stats every 30 seconds
        setInterval(loadStats, 30000);
    </script>
</body>
</html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def send_comprehensive_scan(self, domain):
        """Perform comprehensive domain scan"""
        if not self.web_integration:
            self.send_error(500, "Web integration not available")
            return
        
        try:
            # Perform comprehensive domain check
            domain_info = self.web_integration.check_domain_comprehensive(domain)
            subdomains = self.web_integration.discover_subdomains(domain)
            
            result = {
                'domain_info': domain_info.__dict__ if domain_info else None,
                'subdomains': [sub.__dict__ for sub in subdomains],
                'scan_timestamp': datetime.now().isoformat()
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result, indent=2).encode())
            
        except Exception as e:
            self.send_error(500, f"Scan failed: {str(e)}")
    
    def handle_batch_scan(self):
        """Handle batch domain scanning"""
        if not self.web_integration:
            self.send_error(500, "Web integration not available")
            return
        
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            domains = data.get('domains', [])
            if not domains:
                self.send_error(400, "No domains provided")
                return
            
            # Perform batch scan
            results = self.web_integration.batch_domain_scan(domains)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(results, indent=2).encode())
            
        except Exception as e:
            self.send_error(500, f"Batch scan failed: {str(e)}")
    
    def send_discovery_stats(self):
        """Send discovery statistics"""
        if not self.web_integration:
            stats = {
                'total_domains': 0,
                'total_subdomains': 0,
                'active_domains': 0,
                'active_subdomains': 0,
                'ssl_enabled': 0
            }
        else:
            stats = {
                'total_domains': len(self.web_integration.discovered_domains),
                'total_subdomains': len(self.web_integration.discovered_subdomains),
                'active_domains': self.web_integration.stats.active_domains,
                'active_subdomains': self.web_integration.stats.active_subdomains,
                'ssl_enabled': self.web_integration.stats.ssl_enabled_domains
            }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(stats).encode())
    
    def send_full_report(self):
        """Send comprehensive discovery report"""
        if not self.web_integration:
            self.send_error(500, "Web integration not available")
            return
        
        try:
            report = self.web_integration.get_discovery_report()
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(report, indent=2).encode())
            
        except Exception as e:
            self.send_error(500, f"Report generation failed: {str(e)}")
    
    def send_database_contents(self):
        """Send database contents"""
        if not self.web_integration:
            self.send_error(500, "Web integration not available")
            return
        
        try:
            with sqlite3.connect(self.web_integration.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Get domains
                domains = conn.execute("SELECT * FROM domains ORDER BY last_checked DESC").fetchall()
                domains_list = [dict(row) for row in domains]
                
                # Get subdomains
                subdomains = conn.execute("SELECT * FROM subdomains ORDER BY last_checked DESC").fetchall()
                subdomains_list = [dict(row) for row in subdomains]
                
                result = {
                    'domains': domains_list,
                    'subdomains': subdomains_list
                }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result, indent=2).encode())
                
        except Exception as e:
            self.send_error(500, f"Database access failed: {str(e)}")

class EnhancedWebDashboard:
    """Enhanced Web Dashboard with Integration"""
    
    def __init__(self, port: int = 8081):
        self.port = port
        self.web_integration = None
        self.server = None
        
        # Initialize web integration if available
        if SpiritualWebIntegration:
            try:
                self.web_integration = SpiritualWebIntegration()
                print("✅ Web integration initialized successfully")
            except Exception as e:
                print(f"⚠️ Web integration failed: {e}")
    
    def start_dashboard(self):
        """Start the enhanced dashboard"""
        print(f"🌟 Starting Enhanced Spiritual Web Dashboard on port {self.port}")
        print(f"🌐 Open your browser to: http://localhost:{self.port}")
        print("🔍 Ready for comprehensive web discovery!")
        
        # Create handler with web integration
        def handler(*args, **kwargs):
            return EnhancedWebDashboardHandler(*args, web_integration=self.web_integration, **kwargs)
        
        with socketserver.TCPServer(("", self.port), handler) as httpd:
            self.server = httpd
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n✨ Enhanced dashboard stopped gracefully")
    
    def stop_dashboard(self):
        """Stop the dashboard"""
        if self.server:
            self.server.shutdown()

def main():
    """Main function"""
    dashboard = EnhancedWebDashboard(port=8081)
    dashboard.start_dashboard()

if __name__ == "__main__":
    main()