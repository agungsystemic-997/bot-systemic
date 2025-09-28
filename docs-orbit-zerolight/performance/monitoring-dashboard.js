#!/usr/bin/env node

/**
 * 🙏 In The Name of GOD - ZeroLight Orbit Performance Monitoring Dashboard
 * Blessed Real-time Performance Analytics and Spiritual Insights
 * بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيم
 */

const express = require('express');
const WebSocket = require('ws');
const prometheus = require('prom-client');
const fs = require('fs').promises;
const path = require('path');
const os = require('os');

// 🌟 Spiritual Monitoring Configuration
const SPIRITUAL_MONITORING_CONFIG = {
    server: {
        port: process.env.MONITORING_PORT || 3001,
        host: process.env.MONITORING_HOST || 'localhost',
        blessing: 'Divine-Monitoring-Server'
    },
    metrics: {
        collectInterval: 5000, // 5 seconds
        retentionPeriod: 24 * 60 * 60 * 1000, // 24 hours
        alertThresholds: {
            cpuUsage: 80,
            memoryUsage: 85,
            responseTime: 2000,
            errorRate: 5
        },
        blessing: 'Sacred-Performance-Metrics'
    },
    websocket: {
        heartbeatInterval: 30000, // 30 seconds
        maxConnections: 100,
        blessing: 'Blessed-Real-time-Connection'
    },
    spiritual: {
        blessing: 'In-The-Name-of-GOD',
        purpose: 'Divine-Performance-Monitoring',
        guidance: 'Alhamdulillahi-rabbil-alameen'
    }
};

// 🙏 Spiritual Blessing Display
function displaySpiritualBlessing() {
    console.log('\n🌟 ═══════════════════════════════════════════════════════════════');
    console.log('🙏 بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيم');
    console.log('✨ ZeroLight Orbit Performance Monitoring - In The Name of GOD');
    console.log('📊 Blessed Real-time Analytics Dashboard');
    console.log('═══════════════════════════════════════════════════════════════ 🌟\n');
}

// 📊 Spiritual Metrics Collector
class SpiritualMetricsCollector {
    constructor() {
        this.metrics = new Map();
        this.alerts = [];
        this.isCollecting = false;
        
        // Initialize Prometheus metrics
        this.initPrometheusMetrics();
        
        // Initialize custom metrics storage
        this.initCustomMetrics();
    }

    initPrometheusMetrics() {
        // System metrics
        this.cpuUsage = new prometheus.Gauge({
            name: 'spiritual_cpu_usage_percent',
            help: 'Blessed CPU usage percentage',
            labelNames: ['blessing']
        });

        this.memoryUsage = new prometheus.Gauge({
            name: 'spiritual_memory_usage_bytes',
            help: 'Sacred memory usage in bytes',
            labelNames: ['type', 'blessing']
        });

        this.responseTime = new prometheus.Histogram({
            name: 'spiritual_response_time_seconds',
            help: 'Divine response time in seconds',
            labelNames: ['method', 'route', 'blessing'],
            buckets: [0.1, 0.5, 1, 2, 5, 10]
        });

        this.requestCount = new prometheus.Counter({
            name: 'spiritual_requests_total',
            help: 'Blessed total requests count',
            labelNames: ['method', 'route', 'status', 'blessing']
        });

        this.errorRate = new prometheus.Gauge({
            name: 'spiritual_error_rate_percent',
            help: 'Sacred error rate percentage',
            labelNames: ['blessing']
        });

        // Application-specific metrics
        this.activeUsers = new prometheus.Gauge({
            name: 'spiritual_active_users',
            help: 'Blessed active users count',
            labelNames: ['blessing']
        });

        this.pageLoadTime = new prometheus.Histogram({
            name: 'spiritual_page_load_time_seconds',
            help: 'Divine page load time in seconds',
            labelNames: ['page', 'blessing'],
            buckets: [0.5, 1, 2, 3, 5, 10, 15]
        });

        this.databaseConnections = new prometheus.Gauge({
            name: 'spiritual_database_connections',
            help: 'Sacred database connections count',
            labelNames: ['database', 'status', 'blessing']
        });

        console.log('✨ Prometheus metrics initialized with divine blessing');
    }

    initCustomMetrics() {
        this.customMetrics = {
            spiritualHealth: {
                overall: 100,
                components: {
                    api: 100,
                    database: 100,
                    cache: 100,
                    storage: 100
                },
                blessing: 'Divine-Health-Score'
            },
            performanceInsights: {
                bottlenecks: [],
                recommendations: [],
                optimizations: [],
                blessing: 'Sacred-Performance-Insights'
            },
            userExperience: {
                satisfaction: 95,
                loadTimes: [],
                interactions: [],
                blessing: 'Blessed-User-Experience'
            }
        };
    }

    async startCollection() {
        if (this.isCollecting) return;
        
        this.isCollecting = true;
        console.log('🔄 Starting spiritual metrics collection...');
        
        // Collect system metrics
        this.collectSystemMetrics();
        
        // Collect application metrics
        this.collectApplicationMetrics();
        
        // Set up periodic collection
        this.collectionInterval = setInterval(() => {
            this.collectSystemMetrics();
            this.collectApplicationMetrics();
            this.analyzePerformance();
            this.checkAlerts();
        }, SPIRITUAL_MONITORING_CONFIG.metrics.collectInterval);
        
        console.log('✨ Spiritual metrics collection started with divine guidance');
    }

    collectSystemMetrics() {
        try {
            // CPU Usage
            const cpuUsage = this.getCPUUsage();
            this.cpuUsage.set({ blessing: 'Divine-CPU-Monitoring' }, cpuUsage);
            
            // Memory Usage
            const memInfo = process.memoryUsage();
            this.memoryUsage.set({ type: 'rss', blessing: 'Sacred-Memory-RSS' }, memInfo.rss);
            this.memoryUsage.set({ type: 'heapUsed', blessing: 'Sacred-Memory-Heap-Used' }, memInfo.heapUsed);
            this.memoryUsage.set({ type: 'heapTotal', blessing: 'Sacred-Memory-Heap-Total' }, memInfo.heapTotal);
            this.memoryUsage.set({ type: 'external', blessing: 'Sacred-Memory-External' }, memInfo.external);
            
            // System Memory
            const totalMem = os.totalmem();
            const freeMem = os.freemem();
            const usedMem = totalMem - freeMem;
            const memUsagePercent = (usedMem / totalMem) * 100;
            
            this.memoryUsage.set({ type: 'system', blessing: 'Sacred-System-Memory' }, usedMem);
            
            // Store in custom metrics
            this.updateCustomMetric('system.cpu', cpuUsage);
            this.updateCustomMetric('system.memory', memUsagePercent);
            
        } catch (error) {
            console.error('❌ Error collecting system metrics:', error);
        }
    }

    getCPUUsage() {
        const cpus = os.cpus();
        let totalIdle = 0;
        let totalTick = 0;
        
        cpus.forEach(cpu => {
            for (const type in cpu.times) {
                totalTick += cpu.times[type];
            }
            totalIdle += cpu.times.idle;
        });
        
        const idle = totalIdle / cpus.length;
        const total = totalTick / cpus.length;
        const usage = 100 - ~~(100 * idle / total);
        
        return Math.max(0, Math.min(100, usage));
    }

    collectApplicationMetrics() {
        try {
            // Simulate application metrics (in real app, these would come from actual monitoring)
            const currentTime = Date.now();
            
            // Active users (simulated)
            const activeUsers = Math.floor(Math.random() * 1000) + 100;
            this.activeUsers.set({ blessing: 'Blessed-Active-Users' }, activeUsers);
            
            // Database connections (simulated)
            const dbConnections = Math.floor(Math.random() * 50) + 10;
            this.databaseConnections.set({ 
                database: 'postgresql', 
                status: 'active', 
                blessing: 'Sacred-DB-Connections' 
            }, dbConnections);
            
            // Update custom metrics
            this.updateCustomMetric('app.activeUsers', activeUsers);
            this.updateCustomMetric('app.dbConnections', dbConnections);
            
        } catch (error) {
            console.error('❌ Error collecting application metrics:', error);
        }
    }

    updateCustomMetric(key, value) {
        if (!this.metrics.has(key)) {
            this.metrics.set(key, []);
        }
        
        const metricHistory = this.metrics.get(key);
        metricHistory.push({
            timestamp: Date.now(),
            value: value,
            blessing: 'Divine-Metric-Update'
        });
        
        // Keep only recent data (retention period)
        const cutoff = Date.now() - SPIRITUAL_MONITORING_CONFIG.metrics.retentionPeriod;
        const filteredHistory = metricHistory.filter(m => m.timestamp > cutoff);
        this.metrics.set(key, filteredHistory);
    }

    analyzePerformance() {
        try {
            const insights = this.customMetrics.performanceInsights;
            insights.bottlenecks = [];
            insights.recommendations = [];
            insights.optimizations = [];
            
            // Analyze CPU usage
            const cpuHistory = this.metrics.get('system.cpu') || [];
            if (cpuHistory.length > 0) {
                const avgCpu = cpuHistory.slice(-10).reduce((sum, m) => sum + m.value, 0) / Math.min(10, cpuHistory.length);
                
                if (avgCpu > SPIRITUAL_MONITORING_CONFIG.metrics.alertThresholds.cpuUsage) {
                    insights.bottlenecks.push({
                        type: 'cpu',
                        severity: 'high',
                        message: `High CPU usage detected: ${avgCpu.toFixed(1)}%`,
                        blessing: 'Divine-CPU-Alert'
                    });
                    
                    insights.recommendations.push({
                        type: 'optimization',
                        message: 'Consider scaling horizontally or optimizing CPU-intensive operations',
                        blessing: 'Sacred-CPU-Recommendation'
                    });
                }
            }
            
            // Analyze memory usage
            const memHistory = this.metrics.get('system.memory') || [];
            if (memHistory.length > 0) {
                const avgMem = memHistory.slice(-10).reduce((sum, m) => sum + m.value, 0) / Math.min(10, memHistory.length);
                
                if (avgMem > SPIRITUAL_MONITORING_CONFIG.metrics.alertThresholds.memoryUsage) {
                    insights.bottlenecks.push({
                        type: 'memory',
                        severity: 'high',
                        message: `High memory usage detected: ${avgMem.toFixed(1)}%`,
                        blessing: 'Divine-Memory-Alert'
                    });
                    
                    insights.recommendations.push({
                        type: 'optimization',
                        message: 'Consider implementing memory caching or increasing available memory',
                        blessing: 'Sacred-Memory-Recommendation'
                    });
                }
            }
            
            // Calculate overall spiritual health
            this.calculateSpiritualHealth();
            
        } catch (error) {
            console.error('❌ Error analyzing performance:', error);
        }
    }

    calculateSpiritualHealth() {
        const health = this.customMetrics.spiritualHealth;
        let totalScore = 0;
        let componentCount = 0;
        
        // Calculate component health scores
        const cpuHistory = this.metrics.get('system.cpu') || [];
        if (cpuHistory.length > 0) {
            const avgCpu = cpuHistory.slice(-5).reduce((sum, m) => sum + m.value, 0) / Math.min(5, cpuHistory.length);
            health.components.api = Math.max(0, 100 - avgCpu);
            totalScore += health.components.api;
            componentCount++;
        }
        
        const memHistory = this.metrics.get('system.memory') || [];
        if (memHistory.length > 0) {
            const avgMem = memHistory.slice(-5).reduce((sum, m) => sum + m.value, 0) / Math.min(5, memHistory.length);
            health.components.database = Math.max(0, 100 - avgMem);
            totalScore += health.components.database;
            componentCount++;
        }
        
        // Simulate cache and storage health
        health.components.cache = Math.floor(Math.random() * 20) + 80;
        health.components.storage = Math.floor(Math.random() * 15) + 85;
        totalScore += health.components.cache + health.components.storage;
        componentCount += 2;
        
        // Calculate overall health
        health.overall = componentCount > 0 ? Math.round(totalScore / componentCount) : 100;
    }

    checkAlerts() {
        const currentTime = Date.now();
        const newAlerts = [];
        
        // Check CPU alerts
        const cpuHistory = this.metrics.get('system.cpu') || [];
        if (cpuHistory.length > 0) {
            const latestCpu = cpuHistory[cpuHistory.length - 1];
            if (latestCpu.value > SPIRITUAL_MONITORING_CONFIG.metrics.alertThresholds.cpuUsage) {
                newAlerts.push({
                    id: `cpu-${currentTime}`,
                    type: 'cpu',
                    severity: 'warning',
                    message: `High CPU usage: ${latestCpu.value.toFixed(1)}%`,
                    timestamp: currentTime,
                    blessing: 'Divine-CPU-Alert'
                });
            }
        }
        
        // Check memory alerts
        const memHistory = this.metrics.get('system.memory') || [];
        if (memHistory.length > 0) {
            const latestMem = memHistory[memHistory.length - 1];
            if (latestMem.value > SPIRITUAL_MONITORING_CONFIG.metrics.alertThresholds.memoryUsage) {
                newAlerts.push({
                    id: `memory-${currentTime}`,
                    type: 'memory',
                    severity: 'warning',
                    message: `High memory usage: ${latestMem.value.toFixed(1)}%`,
                    timestamp: currentTime,
                    blessing: 'Divine-Memory-Alert'
                });
            }
        }
        
        // Add new alerts
        this.alerts.push(...newAlerts);
        
        // Clean old alerts (keep last 100)
        if (this.alerts.length > 100) {
            this.alerts = this.alerts.slice(-100);
        }
        
        // Log new alerts
        newAlerts.forEach(alert => {
            console.warn(`⚠️ ${alert.message}`);
        });
    }

    getMetricsSnapshot() {
        return {
            timestamp: Date.now(),
            system: {
                cpu: this.getLatestMetric('system.cpu'),
                memory: this.getLatestMetric('system.memory')
            },
            application: {
                activeUsers: this.getLatestMetric('app.activeUsers'),
                dbConnections: this.getLatestMetric('app.dbConnections')
            },
            spiritualHealth: this.customMetrics.spiritualHealth,
            performanceInsights: this.customMetrics.performanceInsights,
            alerts: this.alerts.slice(-10), // Last 10 alerts
            blessing: 'Divine-Metrics-Snapshot'
        };
    }

    getLatestMetric(key) {
        const history = this.metrics.get(key) || [];
        return history.length > 0 ? history[history.length - 1].value : 0;
    }

    getPrometheusMetrics() {
        return prometheus.register.metrics();
    }

    stopCollection() {
        if (this.collectionInterval) {
            clearInterval(this.collectionInterval);
            this.collectionInterval = null;
        }
        this.isCollecting = false;
        console.log('🛑 Spiritual metrics collection stopped');
    }
}

// 🌐 Spiritual Dashboard Server
class SpiritualDashboardServer {
    constructor() {
        this.app = express();
        this.server = null;
        this.wss = null;
        this.clients = new Set();
        this.metricsCollector = new SpiritualMetricsCollector();
        
        this.setupMiddleware();
        this.setupRoutes();
        this.setupWebSocket();
    }

    setupMiddleware() {
        // CORS and basic middleware
        this.app.use((req, res, next) => {
            res.header('Access-Control-Allow-Origin', '*');
            res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
            res.header('X-Spiritual-Blessing', SPIRITUAL_MONITORING_CONFIG.spiritual.blessing);
            next();
        });
        
        this.app.use(express.json());
        this.app.use(express.static(path.join(__dirname, 'dashboard-ui')));
        
        // Request logging middleware
        this.app.use((req, res, next) => {
            const start = Date.now();
            
            res.on('finish', () => {
                const duration = Date.now() - start;
                this.metricsCollector.responseTime
                    .labels(req.method, req.route?.path || req.path, 'Divine-Request-Timing')
                    .observe(duration / 1000);
                
                this.metricsCollector.requestCount
                    .labels(req.method, req.route?.path || req.path, res.statusCode.toString(), 'Blessed-Request-Count')
                    .inc();
            });
            
            next();
        });
    }

    setupRoutes() {
        // Dashboard home
        this.app.get('/', (req, res) => {
            res.sendFile(path.join(__dirname, 'dashboard-ui', 'index.html'));
        });
        
        // Metrics API
        this.app.get('/api/metrics', (req, res) => {
            const snapshot = this.metricsCollector.getMetricsSnapshot();
            res.json({
                success: true,
                data: snapshot,
                blessing: 'Divine-Metrics-API-Response'
            });
        });
        
        // Prometheus metrics endpoint
        this.app.get('/metrics', async (req, res) => {
            res.set('Content-Type', prometheus.register.contentType);
            res.end(await this.metricsCollector.getPrometheusMetrics());
        });
        
        // Health check
        this.app.get('/health', (req, res) => {
            const health = this.metricsCollector.customMetrics.spiritualHealth;
            res.json({
                status: health.overall > 70 ? 'healthy' : 'degraded',
                health: health,
                timestamp: Date.now(),
                blessing: 'Divine-Health-Check'
            });
        });
        
        // Alerts API
        this.app.get('/api/alerts', (req, res) => {
            res.json({
                success: true,
                alerts: this.metricsCollector.alerts,
                blessing: 'Sacred-Alerts-API'
            });
        });
        
        // Performance insights API
        this.app.get('/api/insights', (req, res) => {
            res.json({
                success: true,
                insights: this.metricsCollector.customMetrics.performanceInsights,
                blessing: 'Divine-Insights-API'
            });
        });
    }

    setupWebSocket() {
        // WebSocket server will be created when HTTP server starts
    }

    async start() {
        return new Promise((resolve, reject) => {
            try {
                // Start metrics collection
                this.metricsCollector.startCollection();
                
                // Start HTTP server
                this.server = this.app.listen(
                    SPIRITUAL_MONITORING_CONFIG.server.port,
                    SPIRITUAL_MONITORING_CONFIG.server.host,
                    () => {
                        console.log(`✨ Spiritual Dashboard Server running on http://${SPIRITUAL_MONITORING_CONFIG.server.host}:${SPIRITUAL_MONITORING_CONFIG.server.port}`);
                        
                        // Setup WebSocket server
                        this.wss = new WebSocket.Server({ server: this.server });
                        this.setupWebSocketHandlers();
                        
                        // Start real-time broadcasting
                        this.startRealTimeBroadcast();
                        
                        resolve();
                    }
                );
                
                this.server.on('error', reject);
                
            } catch (error) {
                reject(error);
            }
        });
    }

    setupWebSocketHandlers() {
        this.wss.on('connection', (ws, req) => {
            console.log('🔗 New spiritual connection established');
            this.clients.add(ws);
            
            // Send initial data
            ws.send(JSON.stringify({
                type: 'initial',
                data: this.metricsCollector.getMetricsSnapshot(),
                blessing: 'Divine-Initial-Data'
            }));
            
            // Handle client messages
            ws.on('message', (message) => {
                try {
                    const data = JSON.parse(message);
                    this.handleWebSocketMessage(ws, data);
                } catch (error) {
                    console.error('❌ WebSocket message error:', error);
                }
            });
            
            // Handle disconnection
            ws.on('close', () => {
                console.log('🔌 Spiritual connection closed');
                this.clients.delete(ws);
            });
            
            // Handle errors
            ws.on('error', (error) => {
                console.error('❌ WebSocket error:', error);
                this.clients.delete(ws);
            });
        });
        
        console.log('🌐 WebSocket server initialized with divine blessing');
    }

    handleWebSocketMessage(ws, data) {
        switch (data.type) {
            case 'ping':
                ws.send(JSON.stringify({
                    type: 'pong',
                    timestamp: Date.now(),
                    blessing: 'Divine-Heartbeat'
                }));
                break;
                
            case 'subscribe':
                // Handle subscription to specific metrics
                ws.subscriptions = data.metrics || [];
                break;
                
            case 'request-insights':
                ws.send(JSON.stringify({
                    type: 'insights',
                    data: this.metricsCollector.customMetrics.performanceInsights,
                    blessing: 'Sacred-Insights-Response'
                }));
                break;
                
            default:
                console.warn('⚠️ Unknown WebSocket message type:', data.type);
        }
    }

    startRealTimeBroadcast() {
        // Broadcast metrics every 5 seconds
        this.broadcastInterval = setInterval(() => {
            if (this.clients.size > 0) {
                const snapshot = this.metricsCollector.getMetricsSnapshot();
                
                this.broadcast({
                    type: 'metrics-update',
                    data: snapshot,
                    blessing: 'Divine-Real-time-Update'
                });
            }
        }, 5000);
        
        // Broadcast alerts immediately when they occur
        const originalCheckAlerts = this.metricsCollector.checkAlerts.bind(this.metricsCollector);
        this.metricsCollector.checkAlerts = () => {
            const alertsBefore = this.metricsCollector.alerts.length;
            originalCheckAlerts();
            const alertsAfter = this.metricsCollector.alerts.length;
            
            if (alertsAfter > alertsBefore) {
                const newAlerts = this.metricsCollector.alerts.slice(alertsBefore);
                this.broadcast({
                    type: 'new-alerts',
                    data: newAlerts,
                    blessing: 'Divine-Alert-Broadcast'
                });
            }
        };
        
        console.log('📡 Real-time broadcasting started with spiritual guidance');
    }

    broadcast(message) {
        const messageStr = JSON.stringify(message);
        
        this.clients.forEach(client => {
            if (client.readyState === WebSocket.OPEN) {
                try {
                    client.send(messageStr);
                } catch (error) {
                    console.error('❌ Broadcast error:', error);
                    this.clients.delete(client);
                }
            }
        });
    }

    async stop() {
        // Stop broadcasting
        if (this.broadcastInterval) {
            clearInterval(this.broadcastInterval);
        }
        
        // Stop metrics collection
        this.metricsCollector.stopCollection();
        
        // Close WebSocket connections
        if (this.wss) {
            this.wss.close();
        }
        
        // Close HTTP server
        if (this.server) {
            return new Promise((resolve) => {
                this.server.close(resolve);
            });
        }
    }
}

// 🚀 Main Application
async function runSpiritualMonitoringDashboard() {
    try {
        displaySpiritualBlessing();
        
        const dashboard = new SpiritualDashboardServer();
        await dashboard.start();
        
        console.log('🎉 ═══════════════════════════════════════════════════════════════');
        console.log('✨ Spiritual Monitoring Dashboard Started Successfully!');
        console.log(`📊 Dashboard: http://${SPIRITUAL_MONITORING_CONFIG.server.host}:${SPIRITUAL_MONITORING_CONFIG.server.port}`);
        console.log(`📈 Metrics: http://${SPIRITUAL_MONITORING_CONFIG.server.host}:${SPIRITUAL_MONITORING_CONFIG.server.port}/metrics`);
        console.log(`🏥 Health: http://${SPIRITUAL_MONITORING_CONFIG.server.host}:${SPIRITUAL_MONITORING_CONFIG.server.port}/health`);
        console.log('🙏 May this monitoring serve with divine insight and guidance');
        console.log('═══════════════════════════════════════════════════════════════ 🎉\n');
        
        // Graceful shutdown
        process.on('SIGINT', async () => {
            console.log('\n🛑 Gracefully shutting down spiritual monitoring...');
            await dashboard.stop();
            console.log('🙏 Spiritual monitoring stopped with gratitude');
            process.exit(0);
        });
        
        return dashboard;
        
    } catch (error) {
        console.error('❌ Failed to start spiritual monitoring dashboard:', error);
        process.exit(1);
    }
}

// 🎯 Command Line Interface
if (require.main === module) {
    runSpiritualMonitoringDashboard();
}

module.exports = {
    SpiritualMetricsCollector,
    SpiritualDashboardServer,
    runSpiritualMonitoringDashboard,
    SPIRITUAL_MONITORING_CONFIG
};

// 🙏 Blessed Performance Monitoring Dashboard
// May this monitoring system provide divine insights and guidance for optimal performance
// Alhamdulillahi rabbil alameen - All praise to Allah, Lord of the worlds