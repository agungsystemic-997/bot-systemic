#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Dashboard Starter
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and start the enhanced dashboard
try:
    from spiritual_web_dashboard_enhanced import EnhancedWebDashboard
    
    print("Starting Enhanced Spiritual Web Dashboard on port 8082...")
    dashboard = EnhancedWebDashboard(port=8082)
    dashboard.start_dashboard()
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Starting fallback dashboard...")
    
    # Fallback to simple dashboard
    import http.server
    import socketserver
    
    class SimpleHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                html = """
                <html>
                <head><title>Enhanced Dashboard</title></head>
                <body>
                    <h1>Enhanced Spiritual Web Dashboard</h1>
                    <p>Dashboard is running on port 8082</p>
                    <p>Integration mode: Fallback</p>
                </body>
                </html>
                """
                self.wfile.write(html.encode())
            else:
                super().do_GET()
    
    with socketserver.TCPServer(("", 8082), SimpleHandler) as httpd:
        print("Fallback dashboard running on port 8082")
        httpd.serve_forever()

except Exception as e:
    print(f"Error starting dashboard: {e}")