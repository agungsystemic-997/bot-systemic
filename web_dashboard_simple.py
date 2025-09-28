#!/usr/bin/env python3
"""
Simple Web Dashboard for Domain Checking
"""

import http.server
import socketserver
import json
import urllib.parse
import requests
import socket
import ssl
import dns.resolver
import whois
from datetime import datetime
import threading
import time

class WebDashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_dashboard_html()
        elif self.path.startswith('/check/'):
            domain = self.path.split('/check/')[1]
            self.send_domain_check(domain)
        elif self.path == '/api/status':
            self.send_status_api()
        else:
            super().do_GET()
    
    def send_dashboard_html(self):
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>🌟 Spiritual Web Checker Dashboard</title>
    <meta charset="utf-8">
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        .header { 
            text-align: center; 
            margin-bottom: 30px; 
        }
        .check-form { 
            text-align: center; 
            margin-bottom: 30px; 
        }
        input[type="text"] { 
            padding: 10px; 
            width: 300px; 
            border: none;
            border-radius: 5px;
            margin-right: 10px;
        }
        button { 
            padding: 10px 20px; 
            background: #4CAF50; 
            color: white; 
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover { background: #45a049; }
        .results { 
            margin-top: 20px; 
        }
        .status-good { color: #4CAF50; }
        .status-bad { color: #f44336; }
        .domain-info { 
            background: rgba(255,255,255,0.1); 
            padding: 15px; 
            margin: 10px 0; 
            border-radius: 10px;
        }
        .subdomain-list {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .subdomain-item {
            background: rgba(255,255,255,0.1);
            padding: 10px;
            border-radius: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌟 Spiritual Web Infrastructure Checker 🌟</h1>
            <p>Ladang Berkah Digital - ZeroLight Orbit System</p>
            <p>Check domain, subdomain status, and readiness</p>
        </div>
        
        <div class="check-form">
            <input type="text" id="domainInput" placeholder="Enter domain (e.g., google.com)" />
            <button onclick="checkDomain()">Check Domain</button>
        </div>
        
        <div id="results" class="results"></div>
    </div>

    <script>
        function checkDomain() {
            const domain = document.getElementById('domainInput').value;
            if (!domain) {
                alert('Please enter a domain name');
                return;
            }
            
            document.getElementById('results').innerHTML = '<p>🔍 Checking domain: ' + domain + '...</p>';
            
            fetch('/check/' + domain)
                .then(response => response.json())
                .then(data => displayResults(data))
                .catch(error => {
                    document.getElementById('results').innerHTML = '<p class="status-bad">❌ Error: ' + error + '</p>';
                });
        }
        
        function displayResults(data) {
            let html = '<div class="domain-info">';
            html += '<h2>🌐 Domain: ' + data.domain + '</h2>';
            html += '<p><strong>Checked at:</strong> ' + data.timestamp + '</p>';
            
            // Main domain status
            html += '<h3>📊 Main Domain Status:</h3>';
            html += '<ul>';
            for (let check of data.main_checks) {
                const statusClass = check.status === 'success' ? 'status-good' : 'status-bad';
                html += '<li class="' + statusClass + '">' + check.message + '</li>';
            }
            html += '</ul>';
            
            // Subdomains
            if (data.subdomains && data.subdomains.length > 0) {
                html += '<h3>🔍 Subdomain Status:</h3>';
                html += '<div class="subdomain-list">';
                for (let sub of data.subdomains) {
                    html += '<div class="subdomain-item">';
                    html += '<h4>' + sub.name + '</h4>';
                    for (let check of sub.checks) {
                        const statusClass = check.status === 'success' ? 'status-good' : 'status-bad';
                        html += '<p class="' + statusClass + '">' + check.message + '</p>';
                    }
                    html += '</div>';
                }
                html += '</div>';
            }
            
            html += '</div>';
            document.getElementById('results').innerHTML = html;
        }
        
        // Allow Enter key to trigger check
        document.getElementById('domainInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                checkDomain();
            }
        });
    </script>
</body>
</html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def send_domain_check(self, domain):
        """Perform domain check and return JSON results"""
        results = {
            'domain': domain,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'main_checks': [],
            'subdomains': []
        }
        
        # Main domain checks
        try:
            # DNS Resolution
            ip = socket.gethostbyname(domain)
            results['main_checks'].append({
                'status': 'success',
                'message': f'✅ DNS Resolution: {ip}'
            })
        except Exception as e:
            results['main_checks'].append({
                'status': 'error',
                'message': f'❌ DNS Resolution failed: {str(e)}'
            })
        
        # HTTP Check
        try:
            response = requests.get(f'http://{domain}', timeout=10)
            results['main_checks'].append({
                'status': 'success',
                'message': f'✅ HTTP Status: {response.status_code}'
            })
        except Exception as e:
            results['main_checks'].append({
                'status': 'error',
                'message': f'❌ HTTP failed: {str(e)}'
            })
        
        # HTTPS Check
        try:
            response = requests.get(f'https://{domain}', timeout=10)
            results['main_checks'].append({
                'status': 'success',
                'message': f'✅ HTTPS Status: {response.status_code}'
            })
            
            # SSL Certificate
            context = ssl.create_default_context()
            with socket.create_connection((domain, 443), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert = ssock.getpeercert()
                    results['main_checks'].append({
                        'status': 'success',
                        'message': f'✅ SSL Certificate: Valid until {cert["notAfter"]}'
                    })
        except Exception as e:
            results['main_checks'].append({
                'status': 'error',
                'message': f'❌ HTTPS/SSL failed: {str(e)}'
            })
        
        # WHOIS
        try:
            w = whois.whois(domain)
            if w.expiration_date:
                exp_date = w.expiration_date[0] if isinstance(w.expiration_date, list) else w.expiration_date
                results['main_checks'].append({
                    'status': 'success',
                    'message': f'✅ Domain expires: {exp_date}'
                })
            if w.registrar:
                results['main_checks'].append({
                    'status': 'success',
                    'message': f'✅ Registrar: {w.registrar}'
                })
        except Exception as e:
            results['main_checks'].append({
                'status': 'error',
                'message': f'❌ WHOIS failed: {str(e)}'
            })
        
        # Check common subdomains
        common_subdomains = ['www', 'mail', 'ftp', 'blog', 'shop', 'api', 'admin']
        for sub in common_subdomains:
            subdomain = f'{sub}.{domain}'
            sub_result = {
                'name': subdomain,
                'checks': []
            }
            
            try:
                ip = socket.gethostbyname(subdomain)
                sub_result['checks'].append({
                    'status': 'success',
                    'message': f'✅ Resolves to: {ip}'
                })
                
                # HTTP check
                try:
                    response = requests.get(f'http://{subdomain}', timeout=5)
                    sub_result['checks'].append({
                        'status': 'success',
                        'message': f'✅ HTTP: {response.status_code}'
                    })
                except:
                    sub_result['checks'].append({
                        'status': 'error',
                        'message': '❌ HTTP: Not accessible'
                    })
                
                # HTTPS check
                try:
                    response = requests.get(f'https://{subdomain}', timeout=5)
                    sub_result['checks'].append({
                        'status': 'success',
                        'message': f'✅ HTTPS: {response.status_code}'
                    })
                except:
                    sub_result['checks'].append({
                        'status': 'error',
                        'message': '❌ HTTPS: Not accessible'
                    })
                
                results['subdomains'].append(sub_result)
                
            except Exception as e:
                sub_result['checks'].append({
                    'status': 'error',
                    'message': '❌ Domain not found'
                })
                # Only add if there were some checks
                if sub_result['checks']:
                    results['subdomains'].append(sub_result)
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(results, indent=2).encode())
    
    def send_status_api(self):
        """Send system status"""
        status = {
            'status': 'online',
            'timestamp': datetime.now().isoformat(),
            'system': 'Spiritual Web Checker Dashboard',
            'version': '1.0.0'
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(status).encode())

def start_dashboard(port=8080):
    """Start the web dashboard"""
    print(f"🌟 Starting Spiritual Web Checker Dashboard on port {port}")
    print(f"🌐 Open your browser to: http://localhost:{port}")
    print("🔍 Ready to check domains and subdomains!")
    
    with socketserver.TCPServer(("", port), WebDashboardHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n✨ Dashboard stopped gracefully")

if __name__ == "__main__":
    start_dashboard()