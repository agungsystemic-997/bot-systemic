#!/usr/bin/env python3
"""
🌐 SPIRITUAL WEB DASHBOARD
Ladang Berkah Digital - ZeroLight Orbit System
Web Dashboard for Infrastructure Status Display

بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from dataclasses import asdict
import webbrowser
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import os

# Import our web checker
from spiritual_web_checker import (
    SpiritualWebChecker, WebInfrastructureStatus, DomainStatus, 
    WebStatus, ReadinessLevel
)

class SpiritualWebDashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the web dashboard"""
    
    def __init__(self, *args, dashboard_instance=None, **kwargs):
        self.dashboard = dashboard_instance
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        try:
            parsed_path = urllib.parse.urlparse(self.path)
            path = parsed_path.path
            query_params = urllib.parse.parse_qs(parsed_path.query)
            
            if path == "/" or path == "/index.html":
                self._serve_dashboard()
            elif path == "/api/status":
                self._serve_api_status(query_params)
            elif path == "/api/check":
                self._serve_api_check(query_params)
            elif path == "/api/history":
                self._serve_api_history()
            elif path.startswith("/static/"):
                self._serve_static_file(path)
            else:
                self._serve_404()
                
        except Exception as e:
            self._serve_error(str(e))
    
    def do_POST(self):
        """Handle POST requests"""
        try:
            if self.path == "/api/check":
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                domain = data.get('domain', '').strip()
                if domain:
                    # Start async check
                    asyncio.create_task(self.dashboard.check_domain_async(domain))
                    
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        'status': 'started',
                        'message': f'Infrastructure check started for {domain}'
                    }).encode())
                else:
                    self._serve_error("Domain is required", 400)
            else:
                self._serve_404()
                
        except Exception as e:
            self._serve_error(str(e))
    
    def _serve_dashboard(self):
        """Serve the main dashboard HTML"""
        html_content = self._generate_dashboard_html()
        
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html_content.encode('utf-8'))
    
    def _serve_api_status(self, query_params):
        """Serve current status API"""
        try:
            domain = query_params.get('domain', [''])[0]
            
            if domain and domain in self.dashboard.status_cache:
                status_data = self.dashboard.status_cache[domain]
                
                # Convert to JSON-serializable format
                json_data = self._infrastructure_to_json(status_data)
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(json_data, indent=2).encode())
            else:
                self.send_response(404)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'error': 'Domain not found or not checked yet'
                }).encode())
                
        except Exception as e:
            self._serve_error(str(e))
    
    def _serve_api_check(self, query_params):
        """Serve check API"""
        domain = query_params.get('domain', [''])[0]
        
        if not domain:
            self._serve_error("Domain parameter is required", 400)
            return
        
        # Return current status if available
        if domain in self.dashboard.status_cache:
            status_data = self.dashboard.status_cache[domain]
            json_data = self._infrastructure_to_json(status_data)
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(json_data, indent=2).encode())
        else:
            self.send_response(202)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'status': 'checking',
                'message': f'Checking {domain}...'
            }).encode())
    
    def _serve_api_history(self):
        """Serve check history API"""
        try:
            history_data = []
            
            for domain, status in self.dashboard.status_cache.items():
                history_data.append({
                    'domain': domain,
                    'readiness_level': status.readiness_level.value,
                    'health_score': status.overall_health_score,
                    'scan_timestamp': status.scan_timestamp.isoformat(),
                    'scan_duration': status.scan_duration_seconds
                })
            
            # Sort by timestamp (newest first)
            history_data.sort(key=lambda x: x['scan_timestamp'], reverse=True)
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(history_data, indent=2).encode())
            
        except Exception as e:
            self._serve_error(str(e))
    
    def _serve_static_file(self, path):
        """Serve static files (CSS, JS, etc.)"""
        # For now, return 404 for static files
        self._serve_404()
    
    def _serve_404(self):
        """Serve 404 error"""
        self.send_response(404)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(b'<h1>404 - Not Found</h1>')
    
    def _serve_error(self, error_message, status_code=500):
        """Serve error response"""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({
            'error': error_message
        }).encode())
    
    def _infrastructure_to_json(self, infrastructure_status: WebInfrastructureStatus) -> Dict:
        """Convert infrastructure status to JSON-serializable format"""
        try:
            # Convert dataclass to dict
            data = asdict(infrastructure_status)
            
            # Convert datetime objects to ISO strings
            if data.get('scan_timestamp'):
                data['scan_timestamp'] = infrastructure_status.scan_timestamp.isoformat()
            
            if data.get('domain_info', {}).get('creation_date'):
                data['domain_info']['creation_date'] = infrastructure_status.domain_info.creation_date.isoformat()
            
            if data.get('domain_info', {}).get('expiration_date'):
                data['domain_info']['expiration_date'] = infrastructure_status.domain_info.expiration_date.isoformat()
            
            if data.get('main_website', {}).get('last_checked'):
                data['main_website']['last_checked'] = infrastructure_status.main_website.last_checked.isoformat()
            
            if data.get('main_website', {}).get('ssl_info', {}).get('expires_at'):
                ssl_info = infrastructure_status.main_website.ssl_info
                if ssl_info and ssl_info.expires_at:
                    data['main_website']['ssl_info']['expires_at'] = ssl_info.expires_at.isoformat()
            
            return data
            
        except Exception as e:
            return {'error': f'Serialization error: {str(e)}'}
    
    def _generate_dashboard_html(self) -> str:
        """Generate the dashboard HTML"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌐 Spiritual Web Dashboard - Ladang Berkah Digital</title>
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
        
        .search-section {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .search-form {
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .search-input {
            flex: 1;
            min-width: 300px;
            padding: 15px 20px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        
        .search-input:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .search-btn {
            padding: 15px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        .search-btn:hover {
            transform: translateY(-2px);
        }
        
        .search-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .status-section {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .status-card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }
        
        .status-card:hover {
            transform: translateY(-5px);
        }
        
        .status-header {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .status-icon {
            font-size: 2em;
            margin-right: 15px;
        }
        
        .status-title {
            font-size: 1.3em;
            font-weight: bold;
        }
        
        .status-content {
            line-height: 1.6;
        }
        
        .status-ready { border-left: 5px solid #4CAF50; }
        .status-partial { border-left: 5px solid #FF9800; }
        .status-not-ready { border-left: 5px solid #F44336; }
        .status-unknown { border-left: 5px solid #9E9E9E; }
        
        .health-score {
            font-size: 2em;
            font-weight: bold;
            text-align: center;
            margin: 15px 0;
        }
        
        .health-score.excellent { color: #4CAF50; }
        .health-score.good { color: #8BC34A; }
        .health-score.fair { color: #FF9800; }
        .health-score.poor { color: #F44336; }
        
        .details-section {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .details-tabs {
            display: flex;
            border-bottom: 2px solid #e0e0e0;
            margin-bottom: 20px;
        }
        
        .tab-btn {
            padding: 15px 25px;
            background: none;
            border: none;
            font-size: 16px;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            transition: all 0.3s;
        }
        
        .tab-btn.active {
            border-bottom-color: #667eea;
            color: #667eea;
            font-weight: bold;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .loading {
            text-align: center;
            padding: 50px;
            font-size: 1.2em;
            color: #666;
        }
        
        .loading::after {
            content: '';
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-left: 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
        }
        
        .success {
            background: #e8f5e8;
            color: #2e7d32;
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
        }
        
        .recommendations {
            background: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 20px;
            margin: 20px 0;
            border-radius: 0 10px 10px 0;
        }
        
        .recommendations h4 {
            margin-bottom: 15px;
            color: #e65100;
        }
        
        .recommendations ul {
            list-style: none;
            padding: 0;
        }
        
        .recommendations li {
            padding: 5px 0;
            border-bottom: 1px solid #ffcc80;
        }
        
        .recommendations li:last-child {
            border-bottom: none;
        }
        
        .footer {
            text-align: center;
            color: white;
            margin-top: 50px;
            opacity: 0.8;
        }
        
        @media (max-width: 768px) {
            .search-form {
                flex-direction: column;
            }
            
            .search-input {
                min-width: 100%;
            }
            
            .details-tabs {
                flex-wrap: wrap;
            }
            
            .tab-btn {
                flex: 1;
                min-width: 120px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌐 Spiritual Web Dashboard</h1>
            <p>Ladang Berkah Digital - ZeroLight Orbit System</p>
            <p style="font-size: 0.9em; margin-top: 10px;">بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ</p>
        </div>
        
        <div class="search-section">
            <form class="search-form" id="searchForm">
                <input type="text" class="search-input" id="domainInput" 
                       placeholder="Masukkan domain (contoh: google.com)" required>
                <button type="submit" class="search-btn" id="searchBtn">
                    🔍 Cek Status
                </button>
            </form>
        </div>
        
        <div id="statusSection" class="status-section" style="display: none;">
            <!-- Status cards will be populated here -->
        </div>
        
        <div id="detailsSection" class="details-section" style="display: none;">
            <div class="details-tabs">
                <button class="tab-btn active" onclick="showTab('domain')">🏷️ Domain</button>
                <button class="tab-btn" onclick="showTab('website')">🌐 Website</button>
                <button class="tab-btn" onclick="showTab('subdomains')">🔍 Subdomain</button>
                <button class="tab-btn" onclick="showTab('recommendations')">💡 Rekomendasi</button>
            </div>
            
            <div id="domainTab" class="tab-content active">
                <!-- Domain details -->
            </div>
            
            <div id="websiteTab" class="tab-content">
                <!-- Website details -->
            </div>
            
            <div id="subdomainsTab" class="tab-content">
                <!-- Subdomains details -->
            </div>
            
            <div id="recommendationsTab" class="tab-content">
                <!-- Recommendations -->
            </div>
        </div>
        
        <div id="loadingSection" class="loading" style="display: none;">
            Sedang memeriksa infrastruktur web...
        </div>
        
        <div class="footer">
            <p>🌟 Ladang Berkah Digital - ZeroLight Orbit System</p>
            <p>بارك الله فيكم</p>
        </div>
    </div>
    
    <script>
        let currentDomain = '';
        let checkInterval = null;
        
        document.getElementById('searchForm').addEventListener('submit', function(e) {
            e.preventDefault();
            const domain = document.getElementById('domainInput').value.trim();
            if (domain) {
                checkDomain(domain);
            }
        });
        
        async function checkDomain(domain) {
            currentDomain = domain;
            
            // Show loading
            document.getElementById('loadingSection').style.display = 'block';
            document.getElementById('statusSection').style.display = 'none';
            document.getElementById('detailsSection').style.display = 'none';
            
            // Disable search button
            const searchBtn = document.getElementById('searchBtn');
            searchBtn.disabled = true;
            searchBtn.textContent = '🔄 Memeriksa...';
            
            try {
                // Start check
                const response = await fetch('/api/check', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ domain: domain })
                });
                
                if (response.ok) {
                    // Start polling for results
                    startPolling(domain);
                } else {
                    throw new Error('Failed to start check');
                }
                
            } catch (error) {
                showError('Error starting check: ' + error.message);
                resetUI();
            }
        }
        
        function startPolling(domain) {
            checkInterval = setInterval(async () => {
                try {
                    const response = await fetch(`/api/status?domain=${encodeURIComponent(domain)}`);
                    
                    if (response.ok) {
                        const data = await response.json();
                        displayResults(data);
                        stopPolling();
                        resetUI();
                    } else if (response.status === 404) {
                        // Still checking, continue polling
                        console.log('Still checking...');
                    } else {
                        throw new Error('Check failed');
                    }
                    
                } catch (error) {
                    console.error('Polling error:', error);
                    // Continue polling for now
                }
            }, 2000); // Poll every 2 seconds
            
            // Stop polling after 2 minutes
            setTimeout(() => {
                if (checkInterval) {
                    stopPolling();
                    showError('Check timeout - please try again');
                    resetUI();
                }
            }, 120000);
        }
        
        function stopPolling() {
            if (checkInterval) {
                clearInterval(checkInterval);
                checkInterval = null;
            }
        }
        
        function resetUI() {
            document.getElementById('loadingSection').style.display = 'none';
            const searchBtn = document.getElementById('searchBtn');
            searchBtn.disabled = false;
            searchBtn.textContent = '🔍 Cek Status';
        }
        
        function displayResults(data) {
            // Hide loading
            document.getElementById('loadingSection').style.display = 'none';
            
            // Show status cards
            displayStatusCards(data);
            
            // Show details
            displayDetails(data);
            
            // Show sections
            document.getElementById('statusSection').style.display = 'grid';
            document.getElementById('detailsSection').style.display = 'block';
        }
        
        function displayStatusCards(data) {
            const statusSection = document.getElementById('statusSection');
            
            // Overall status card
            const readinessClass = `status-${data.readiness_level.replace('_', '-')}`;
            const healthClass = getHealthScoreClass(data.overall_health_score);
            
            statusSection.innerHTML = `
                <div class="status-card ${readinessClass}">
                    <div class="status-header">
                        <div class="status-icon">${getReadinessIcon(data.readiness_level)}</div>
                        <div class="status-title">Status Keseluruhan</div>
                    </div>
                    <div class="status-content">
                        <div class="health-score ${healthClass}">${data.overall_health_score.toFixed(1)}/100</div>
                        <p><strong>Tingkat Kesiapan:</strong> ${getReadinessText(data.readiness_level)}</p>
                        <p><strong>Domain:</strong> ${data.domain_info.domain}</p>
                        <p><strong>Waktu Scan:</strong> ${formatDateTime(data.scan_timestamp)}</p>
                    </div>
                </div>
                
                <div class="status-card">
                    <div class="status-header">
                        <div class="status-icon">🏷️</div>
                        <div class="status-title">Status Domain</div>
                    </div>
                    <div class="status-content">
                        <p><strong>Status:</strong> ${getDomainStatusText(data.domain_info.status)}</p>
                        ${data.domain_info.registrar ? `<p><strong>Registrar:</strong> ${data.domain_info.registrar}</p>` : ''}
                        ${data.domain_info.expiration_date ? `<p><strong>Kedaluwarsa:</strong> ${formatDate(data.domain_info.expiration_date)}</p>` : ''}
                        ${data.domain_info.days_until_expiry ? `<p><strong>Hari Tersisa:</strong> ${data.domain_info.days_until_expiry} hari</p>` : ''}
                    </div>
                </div>
                
                <div class="status-card">
                    <div class="status-header">
                        <div class="status-icon">🌐</div>
                        <div class="status-title">Status Website</div>
                    </div>
                    <div class="status-content">
                        ${data.main_website ? `
                            <p><strong>Status:</strong> ${getWebStatusText(data.main_website.status)}</p>
                            <p><strong>Kode HTTP:</strong> ${data.main_website.status_code}</p>
                            <p><strong>Waktu Respon:</strong> ${data.main_website.response_time_ms.toFixed(0)}ms</p>
                            ${data.main_website.ssl_info && data.main_website.ssl_info.is_valid ? 
                                `<p><strong>SSL:</strong> ✅ Valid</p>` : 
                                `<p><strong>SSL:</strong> ❌ Invalid</p>`
                            }
                        ` : '<p>Website tidak dapat diakses</p>'}
                    </div>
                </div>
                
                <div class="status-card">
                    <div class="status-header">
                        <div class="status-icon">🔍</div>
                        <div class="status-title">Subdomain</div>
                    </div>
                    <div class="status-content">
                        <p><strong>Ditemukan:</strong> ${data.subdomains.length} subdomain aktif</p>
                        ${data.subdomains.length > 0 ? `
                            <p><strong>Online:</strong> ${data.subdomains.filter(sub => 
                                sub.web_service && sub.web_service.status === 'online'
                            ).length} subdomain</p>
                        ` : ''}
                        <p><strong>Durasi Scan:</strong> ${data.scan_duration_seconds.toFixed(1)}s</p>
                    </div>
                </div>
            `;
        }
        
        function displayDetails(data) {
            // Domain tab
            document.getElementById('domainTab').innerHTML = generateDomainDetails(data.domain_info);
            
            // Website tab
            document.getElementById('websiteTab').innerHTML = generateWebsiteDetails(data.main_website);
            
            // Subdomains tab
            document.getElementById('subdomainsTab').innerHTML = generateSubdomainsDetails(data.subdomains);
            
            // Recommendations tab
            document.getElementById('recommendationsTab').innerHTML = generateRecommendations(data.recommendations);
        }
        
        function generateDomainDetails(domainInfo) {
            let html = `
                <h3>🏷️ Informasi Domain: ${domainInfo.domain}</h3>
                <p><strong>Status:</strong> ${getDomainStatusText(domainInfo.status)}</p>
            `;
            
            if (domainInfo.registrar) {
                html += `<p><strong>Registrar:</strong> ${domainInfo.registrar}</p>`;
            }
            
            if (domainInfo.creation_date) {
                html += `<p><strong>Tanggal Dibuat:</strong> ${formatDate(domainInfo.creation_date)}</p>`;
            }
            
            if (domainInfo.expiration_date) {
                html += `<p><strong>Tanggal Kedaluwarsa:</strong> ${formatDate(domainInfo.expiration_date)}</p>`;
                html += `<p><strong>Hari Tersisa:</strong> ${domainInfo.days_until_expiry} hari</p>`;
            }
            
            if (domainInfo.nameservers && domainInfo.nameservers.length > 0) {
                html += `<h4>Nameservers:</h4><ul>`;
                domainInfo.nameservers.forEach(ns => {
                    html += `<li>${ns}</li>`;
                });
                html += `</ul>`;
            }
            
            if (domainInfo.dns_records && Object.keys(domainInfo.dns_records).length > 0) {
                html += `<h4>DNS Records:</h4>`;
                for (const [type, records] of Object.entries(domainInfo.dns_records)) {
                    html += `<h5>${type} Records:</h5><ul>`;
                    records.forEach(record => {
                        html += `<li>${record.value} (TTL: ${record.ttl})</li>`;
                    });
                    html += `</ul>`;
                }
            }
            
            if (domainInfo.error) {
                html += `<div class="error">Error: ${domainInfo.error}</div>`;
            }
            
            return html;
        }
        
        function generateWebsiteDetails(websiteInfo) {
            if (!websiteInfo) {
                return '<p>Informasi website tidak tersedia</p>';
            }
            
            let html = `
                <h3>🌐 Informasi Website: ${websiteInfo.url}</h3>
                <p><strong>Status:</strong> ${getWebStatusText(websiteInfo.status)}</p>
                <p><strong>Kode HTTP:</strong> ${websiteInfo.status_code}</p>
                <p><strong>Waktu Respon:</strong> ${websiteInfo.response_time_ms.toFixed(0)}ms</p>
            `;
            
            if (websiteInfo.server) {
                html += `<p><strong>Server:</strong> ${websiteInfo.server}</p>`;
            }
            
            if (websiteInfo.content_type) {
                html += `<p><strong>Content Type:</strong> ${websiteInfo.content_type}</p>`;
            }
            
            if (websiteInfo.redirect_url) {
                html += `<p><strong>Redirect ke:</strong> ${websiteInfo.redirect_url}</p>`;
            }
            
            if (websiteInfo.ssl_info) {
                const ssl = websiteInfo.ssl_info;
                html += `<h4>Informasi SSL:</h4>`;
                html += `<p><strong>Valid:</strong> ${ssl.is_valid ? '✅ Ya' : '❌ Tidak'}</p>`;
                
                if (ssl.expires_at) {
                    html += `<p><strong>Kedaluwarsa:</strong> ${formatDate(ssl.expires_at)}</p>`;
                    html += `<p><strong>Hari Tersisa:</strong> ${ssl.days_until_expiry} hari</p>`;
                }
                
                if (ssl.issuer && ssl.issuer.commonName) {
                    html += `<p><strong>Penerbit:</strong> ${ssl.issuer.commonName}</p>`;
                }
                
                if (ssl.is_self_signed) {
                    html += `<p><strong>Self-Signed:</strong> Ya</p>`;
                }
                
                if (ssl.error) {
                    html += `<div class="error">SSL Error: ${ssl.error}</div>`;
                }
            }
            
            if (websiteInfo.error) {
                html += `<div class="error">Error: ${websiteInfo.error}</div>`;
            }
            
            return html;
        }
        
        function generateSubdomainsDetails(subdomains) {
            if (!subdomains || subdomains.length === 0) {
                return '<p>Tidak ada subdomain aktif yang ditemukan</p>';
            }
            
            let html = `<h3>🔍 Subdomain Aktif (${subdomains.length})</h3>`;
            
            subdomains.forEach(subdomain => {
                html += `
                    <div style="border: 1px solid #e0e0e0; border-radius: 10px; padding: 15px; margin: 10px 0;">
                        <h4>${subdomain.subdomain}</h4>
                `;
                
                if (subdomain.ip_addresses && subdomain.ip_addresses.length > 0) {
                    html += `<p><strong>IP:</strong> ${subdomain.ip_addresses.join(', ')}</p>`;
                }
                
                if (subdomain.web_service) {
                    const ws = subdomain.web_service;
                    html += `<p><strong>Web Status:</strong> ${getWebStatusText(ws.status)}</p>`;
                    if (ws.status_code) {
                        html += `<p><strong>HTTP:</strong> ${ws.status_code}</p>`;
                    }
                    if (ws.response_time_ms) {
                        html += `<p><strong>Respon:</strong> ${ws.response_time_ms.toFixed(0)}ms</p>`;
                    }
                }
                
                if (subdomain.error) {
                    html += `<div class="error">Error: ${subdomain.error}</div>`;
                }
                
                html += `</div>`;
            });
            
            return html;
        }
        
        function generateRecommendations(recommendations) {
            if (!recommendations || recommendations.length === 0) {
                return '<div class="success">✅ Tidak ada rekomendasi khusus - infrastruktur dalam kondisi baik!</div>';
            }
            
            let html = `
                <div class="recommendations">
                    <h4>💡 Rekomendasi Perbaikan</h4>
                    <ul>
            `;
            
            recommendations.forEach(rec => {
                html += `<li>${rec}</li>`;
            });
            
            html += `
                    </ul>
                </div>
            `;
            
            return html;
        }
        
        function showTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            
            // Show selected tab
            document.getElementById(tabName + 'Tab').classList.add('active');
            event.target.classList.add('active');
        }
        
        function showError(message) {
            const statusSection = document.getElementById('statusSection');
            statusSection.innerHTML = `<div class="error">${message}</div>`;
            statusSection.style.display = 'block';
        }
        
        // Helper functions
        function getReadinessIcon(level) {
            const icons = {
                'ready': '✅',
                'partially_ready': '⚠️',
                'not_ready': '❌',
                'maintenance': '🔧',
                'unknown': '❓'
            };
            return icons[level] || '❓';
        }
        
        function getReadinessText(level) {
            const texts = {
                'ready': 'Siap',
                'partially_ready': 'Sebagian Siap',
                'not_ready': 'Belum Siap',
                'maintenance': 'Maintenance',
                'unknown': 'Tidak Diketahui'
            };
            return texts[level] || 'Tidak Diketahui';
        }
        
        function getDomainStatusText(status) {
            const texts = {
                'active': '✅ Aktif',
                'inactive': '❌ Tidak Aktif',
                'expired': '🚨 Kedaluwarsa',
                'suspended': '⏸️ Ditangguhkan',
                'pending': '⏳ Pending',
                'unknown': '❓ Tidak Diketahui',
                'error': '💥 Error'
            };
            return texts[status] || 'Tidak Diketahui';
        }
        
        function getWebStatusText(status) {
            const texts = {
                'online': '✅ Online',
                'offline': '❌ Offline',
                'slow': '🐌 Lambat',
                'error': '💥 Error',
                'timeout': '⏰ Timeout',
                'ssl_error': '🔒 SSL Error',
                'dns_error': '🌐 DNS Error',
                'redirect': '🔄 Redirect'
            };
            return texts[status] || 'Tidak Diketahui';
        }
        
        function getHealthScoreClass(score) {
            if (score >= 80) return 'excellent';
            if (score >= 60) return 'good';
            if (score >= 40) return 'fair';
            return 'poor';
        }
        
        function formatDateTime(isoString) {
            return new Date(isoString).toLocaleString('id-ID');
        }
        
        function formatDate(isoString) {
            return new Date(isoString).toLocaleDateString('id-ID');
        }
    </script>
</body>
</html>
        """
    
    def log_message(self, format, *args):
        """Override to reduce logging noise"""
        pass

class SpiritualWebDashboard:
    """
    🌟 Spiritual Web Dashboard
    
    Web-based dashboard for infrastructure monitoring
    """
    
    def __init__(self, host: str = "localhost", port: int = 8080):
        """Initialize the dashboard"""
        self.host = host
        self.port = port
        self.logger = logging.getLogger(__name__)
        
        # Web checker instance
        self.web_checker = None
        
        # Status cache
        self.status_cache: Dict[str, WebInfrastructureStatus] = {}
        
        # HTTP server
        self.server = None
        self.server_thread = None
    
    async def initialize(self):
        """Initialize the dashboard"""
        try:
            self.logger.info("🚀 Initializing Spiritual Web Dashboard...")
            
            # Initialize web checker
            self.web_checker = SpiritualWebChecker()
            await self.web_checker.initialize()
            
            self.logger.info("✅ Dashboard initialized successfully")
            
        except Exception as e:
            self.logger.error(f"💥 Failed to initialize dashboard: {e}")
            raise
    
    def start_server(self):
        """Start the HTTP server"""
        try:
            self.logger.info(f"🌐 Starting web server on {self.host}:{self.port}")
            
            # Create handler with dashboard reference
            def handler(*args, **kwargs):
                return SpiritualWebDashboardHandler(*args, dashboard_instance=self, **kwargs)
            
            # Create and start server
            self.server = HTTPServer((self.host, self.port), handler)
            
            # Start server in separate thread
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            self.logger.info(f"✅ Web server started at http://{self.host}:{self.port}")
            
            # Open browser
            try:
                webbrowser.open(f"http://{self.host}:{self.port}")
            except Exception:
                pass  # Browser opening is optional
            
        except Exception as e:
            self.logger.error(f"💥 Failed to start server: {e}")
            raise
    
    async def check_domain_async(self, domain: str):
        """Check domain asynchronously and cache results"""
        try:
            self.logger.info(f"🔍 Starting async check for domain: {domain}")
            
            # Perform infrastructure check
            infrastructure_status = await self.web_checker.check_web_infrastructure(
                domain=domain,
                check_subdomains=True,
                subdomain_list=["www", "api", "app", "admin", "blog", "shop", "mail", 
                               "cdn", "static", "dev", "test", "staging", "docs"]
            )
            
            # Cache results
            self.status_cache[domain] = infrastructure_status
            
            self.logger.info(f"✅ Async check completed for domain: {domain}")
            
        except Exception as e:
            self.logger.error(f"💥 Async check failed for domain {domain}: {e}")
            
            # Cache error result
            from spiritual_web_checker import DomainInfo, DomainStatus
            error_status = WebInfrastructureStatus(
                domain_info=DomainInfo(domain=domain, status=DomainStatus.ERROR, error=str(e)),
                readiness_level=ReadinessLevel.UNKNOWN
            )
            self.status_cache[domain] = error_status
    
    def stop_server(self):
        """Stop the HTTP server"""
        try:
            if self.server:
                self.server.shutdown()
                self.server.server_close()
                
            if self.server_thread:
                self.server_thread.join(timeout=5)
            
            self.logger.info("✅ Web server stopped")
            
        except Exception as e:
            self.logger.error(f"💥 Error stopping server: {e}")
    
    async def shutdown(self):
        """Shutdown the dashboard"""
        try:
            self.logger.info("🔄 Shutting down dashboard...")
            
            # Stop server
            self.stop_server()
            
            # Shutdown web checker
            if self.web_checker:
                await self.web_checker.shutdown()
            
            self.logger.info("✅ Dashboard shutdown completed")
            
        except Exception as e:
            self.logger.error(f"💥 Shutdown error: {e}")

# 🌟 Spiritual Blessing for Web Dashboard
SPIRITUAL_DASHBOARD_BLESSING = """
بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ

اللَّهُمَّ بَارِكْ لَنَا فِي هَذَا اللَّوْحَةِ الْمُبَارَكَةِ لِمُرَاقَبَةِ الْمَوَاقِعِ
وَاجْعَلْهَا سَهْلَةً مُفِيدَةً جَمِيلَةً

Ya Allah, berkahilah dashboard web ini dengan:
- 🎨 Tampilan yang indah dan mudah digunakan
- ⚡ Performa yang cepat dan responsif
- 📊 Informasi yang akurat dan lengkap
- 🔄 Pembaruan real-time yang lancar
- 🛡️ Keamanan dalam akses data

آمِينَ يَا رَبَّ الْعَالَمِينَ
"""

if __name__ == "__main__":
    print("🌐 Spiritual Web Dashboard - Ladang Berkah Digital")
    print("=" * 60)
    print(SPIRITUAL_DASHBOARD_BLESSING)
    
    async def main():
        """Main function to run the dashboard"""
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create and initialize dashboard
        dashboard = SpiritualWebDashboard(host="localhost", port=8080)
        
        try:
            await dashboard.initialize()
            dashboard.start_server()
            
            print(f"\n🌐 Dashboard is running at: http://localhost:8080")
            print("Press Ctrl+C to stop the server")
            
            # Keep running
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🔄 Stopping dashboard...")
        except Exception as e:
            print(f"💥 Dashboard error: {e}")
        finally:
            await dashboard.shutdown()
    
    # Run dashboard
    asyncio.run(main())