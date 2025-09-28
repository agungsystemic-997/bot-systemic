// 🙏 In The Name of GOD - ZeroLight Orbit Browser Extension
// Blessed Cross-Browser Extension with Divine Features
// بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيم

// ═══════════════════════════════════════════════════════════════
// 🌟 SPIRITUAL BROWSER EXTENSION CONFIGURATION
// ═══════════════════════════════════════════════════════════════

const SpiritualBrowserConfig = {
    // Extension Identity
    name: "ZeroLight Orbit Browser Extension",
    version: "1.0.0",
    blessing: "In-The-Name-of-GOD",
    purpose: "Divine-Browser-Experience",
    
    // Spiritual Colors - Divine Color Palette
    spiritualColors: {
        divineGold: '#FFD700',
        sacredBlue: '#1E3A8A',
        blessedGreen: '#059669',
        holyWhite: '#FFFFF0',
        spiritualPurple: '#7C3AED',
        celestialSilver: '#C0C0C0',
        angelicPink: '#EC4899',
        peacefulTeal: '#0D9488',
        darkBackground: '#0F172A',
        darkSurface: '#1E293B',
    },
    
    // API Configuration
    apiConfig: {
        baseUrl: 'https://api.zerolight-orbit.com',
        websocketUrl: 'wss://ws.zerolight-orbit.com',
        timeout: 30000,
        maxRetries: 3,
    },
    
    // Security Configuration
    securityConfig: {
        encryptionKeySize: 32,
        sessionTimeout: 3600000, // 1 hour in milliseconds
        maxLoginAttempts: 5,
        passwordMinLength: 8,
    },
    
    // Storage Configuration
    storageConfig: {
        prefix: 'zerolight_orbit_',
        maxStorageSize: 10485760, // 10MB
        compressionEnabled: true,
    },
    
    // Spiritual Features
    spiritualFeatures: [
        'Divine Page Analysis',
        'Sacred Content Filtering',
        'Blessed Security Scanner',
        'Spiritual Productivity Tools',
        'Holy Privacy Protection',
        'Celestial Ad Blocker',
        'Angelic Password Manager',
        'Peaceful Focus Mode'
    ]
};

// 🙏 Spiritual Blessing Display
function displaySpiritualBrowserBlessing() {
    console.log('\n🌟 ═══════════════════════════════════════════════════════════════');
    console.log('🙏 بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيم');
    console.log('✨ ZeroLight Orbit Browser Extension - In The Name of GOD');
    console.log('🌐 Blessed Cross-Browser Experience');
    console.log('🚀 Divine JavaScript Extension with Sacred Features');
    console.log('═══════════════════════════════════════════════════════════════ 🌟\n');
}

// ═══════════════════════════════════════════════════════════════
// 🔐 SPIRITUAL SECURITY MANAGER
// ═══════════════════════════════════════════════════════════════

class SpiritualSecurityManager {
    constructor() {
        this.currentUser = null;
        this.currentSession = null;
        this.encryptionKey = null;
        this.failedAttempts = 0;
        this.isLocked = false;
        this.blessing = "Divine-Security-Manager";
        
        this.initializeSecurity();
    }
    
    async initializeSecurity() {
        try {
            // Generate or load encryption key
            this.encryptionKey = await this.getOrCreateEncryptionKey();
            
            // Initialize secure storage
            await this.initializeSecureStorage();
            
            console.log('🔐 Spiritual security manager initialized with divine blessing');
            return true;
        } catch (error) {
            console.error('❌ Security initialization failed:', error);
            return false;
        }
    }
    
    async authenticateUser(username, password) {
        if (this.isLocked) {
            this.showSecurityAlert('Account Locked', 'Too many failed attempts. Please try again later.');
            return false;
        }
        
        try {
            // Hash password with salt
            const passwordHash = await this.hashPassword(password, username);
            
            // Check credentials (in real extension, check against API/storage)
            if (await this.verifyCredentials(username, passwordHash)) {
                // Create user session
                this.currentUser = {
                    id: this.generateSecureId(),
                    username: username,
                    email: `${username}@zerolight-orbit.com`,
                    displayName: this.capitalizeFirst(username),
                    lastLogin: new Date(),
                    spiritualScore: 100.0,
                    blessing: "Divine-User-Blessed"
                };
                
                // Create session
                this.currentSession = {
                    sessionId: this.generateSecureId(),
                    userId: this.currentUser.id,
                    createdAt: new Date(),
                    expiresAt: new Date(Date.now() + SpiritualBrowserConfig.securityConfig.sessionTimeout),
                    isActive: true,
                    blessing: "Divine-Session-Blessed"
                };
                
                this.failedAttempts = 0;
                await this.saveSession();
                this.notifyAuthenticationChange(true);
                
                console.log(`🙏 User ${username} authenticated with divine blessing`);
                return true;
            } else {
                this.failedAttempts++;
                if (this.failedAttempts >= SpiritualBrowserConfig.securityConfig.maxLoginAttempts) {
                    this.isLocked = true;
                    this.showSecurityAlert('Account Locked', 'Maximum login attempts exceeded.');
                }
                return false;
            }
        } catch (error) {
            console.error('❌ Authentication error:', error);
            return false;
        }
    }
    
    async logoutUser() {
        if (this.currentSession) {
            this.currentSession.isActive = false;
            await this.saveSession();
        }
        
        this.currentUser = null;
        this.currentSession = null;
        this.notifyAuthenticationChange(false);
        
        console.log('🙏 User logged out with divine blessing');
    }
    
    isAuthenticated() {
        if (!this.currentUser || !this.currentSession) {
            return false;
        }
        
        if (!this.currentSession.isActive) {
            return false;
        }
        
        if (new Date() > new Date(this.currentSession.expiresAt)) {
            this.logoutUser();
            return false;
        }
        
        return true;
    }
    
    async encryptData(data) {
        if (!this.encryptionKey) {
            throw new Error('Encryption key not initialized');
        }
        
        const encoder = new TextEncoder();
        const dataBuffer = encoder.encode(JSON.stringify(data));
        
        // Use Web Crypto API for encryption
        const key = await crypto.subtle.importKey(
            'raw',
            this.encryptionKey,
            { name: 'AES-GCM' },
            false,
            ['encrypt']
        );
        
        const iv = crypto.getRandomValues(new Uint8Array(12));
        const encryptedBuffer = await crypto.subtle.encrypt(
            { name: 'AES-GCM', iv: iv },
            key,
            dataBuffer
        );
        
        // Combine IV and encrypted data
        const combined = new Uint8Array(iv.length + encryptedBuffer.byteLength);
        combined.set(iv);
        combined.set(new Uint8Array(encryptedBuffer), iv.length);
        
        return btoa(String.fromCharCode(...combined));
    }
    
    async decryptData(encryptedData) {
        if (!this.encryptionKey) {
            throw new Error('Encryption key not initialized');
        }
        
        const combined = new Uint8Array(atob(encryptedData).split('').map(c => c.charCodeAt(0)));
        const iv = combined.slice(0, 12);
        const encrypted = combined.slice(12);
        
        const key = await crypto.subtle.importKey(
            'raw',
            this.encryptionKey,
            { name: 'AES-GCM' },
            false,
            ['decrypt']
        );
        
        const decryptedBuffer = await crypto.subtle.decrypt(
            { name: 'AES-GCM', iv: iv },
            key,
            encrypted
        );
        
        const decoder = new TextDecoder();
        return JSON.parse(decoder.decode(decryptedBuffer));
    }
    
    async getOrCreateEncryptionKey() {
        const stored = await this.getFromStorage('encryption_key');
        if (stored) {
            return new Uint8Array(stored);
        }
        
        const key = crypto.getRandomValues(new Uint8Array(SpiritualBrowserConfig.securityConfig.encryptionKeySize));
        await this.saveToStorage('encryption_key', Array.from(key));
        return key;
    }
    
    async initializeSecureStorage() {
        // Initialize storage structure
        const storageStructure = {
            users: {},
            sessions: {},
            settings: {},
            spiritualData: {},
            blessing: "Divine-Storage-Initialized"
        };
        
        const existing = await this.getFromStorage('storage_structure');
        if (!existing) {
            await this.saveToStorage('storage_structure', storageStructure);
        }
    }
    
    async hashPassword(password, salt) {
        const encoder = new TextEncoder();
        const data = encoder.encode(password + salt);
        const hashBuffer = await crypto.subtle.digest('SHA-256', data);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    }
    
    async verifyCredentials(username, passwordHash) {
        // For demo purposes, accept any username with password "spiritual123"
        const demoHash = await this.hashPassword("spiritual123", username);
        return passwordHash === demoHash;
    }
    
    generateSecureId() {
        return Array.from(crypto.getRandomValues(new Uint8Array(16)))
            .map(b => b.toString(16).padStart(2, '0'))
            .join('');
    }
    
    capitalizeFirst(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }
    
    async saveSession() {
        if (this.currentSession) {
            await this.saveToStorage(`session_${this.currentSession.sessionId}`, this.currentSession);
        }
    }
    
    async saveToStorage(key, data) {
        const fullKey = SpiritualBrowserConfig.storageConfig.prefix + key;
        
        if (typeof chrome !== 'undefined' && chrome.storage) {
            // Chrome extension storage
            return new Promise((resolve) => {
                chrome.storage.local.set({ [fullKey]: data }, resolve);
            });
        } else {
            // Fallback to localStorage
            localStorage.setItem(fullKey, JSON.stringify(data));
        }
    }
    
    async getFromStorage(key) {
        const fullKey = SpiritualBrowserConfig.storageConfig.prefix + key;
        
        if (typeof chrome !== 'undefined' && chrome.storage) {
            // Chrome extension storage
            return new Promise((resolve) => {
                chrome.storage.local.get([fullKey], (result) => {
                    resolve(result[fullKey]);
                });
            });
        } else {
            // Fallback to localStorage
            const stored = localStorage.getItem(fullKey);
            return stored ? JSON.parse(stored) : null;
        }
    }
    
    showSecurityAlert(title, message) {
        if (typeof chrome !== 'undefined' && chrome.notifications) {
            chrome.notifications.create({
                type: 'basic',
                iconUrl: 'icons/security-alert.png',
                title: title,
                message: message
            });
        } else {
            alert(`${title}: ${message}`);
        }
    }
    
    notifyAuthenticationChange(authenticated) {
        // Send message to content scripts and popup
        const message = {
            type: 'AUTHENTICATION_CHANGED',
            authenticated: authenticated,
            user: this.currentUser,
            blessing: "Divine-Auth-Notification"
        };
        
        if (typeof chrome !== 'undefined' && chrome.runtime) {
            chrome.runtime.sendMessage(message);
        }
        
        // Dispatch custom event for web pages
        window.dispatchEvent(new CustomEvent('spiritualAuthChanged', { detail: message }));
    }
}

// ═══════════════════════════════════════════════════════════════
// 📊 SPIRITUAL CONTENT ANALYZER
// ═══════════════════════════════════════════════════════════════

class SpiritualContentAnalyzer {
    constructor() {
        this.analysisResults = {};
        this.spiritualKeywords = [
            'spiritual', 'divine', 'blessed', 'sacred', 'holy', 'peaceful',
            'meditation', 'prayer', 'wisdom', 'enlightenment', 'harmony',
            'compassion', 'gratitude', 'mindfulness', 'serenity', 'tranquil'
        ];
        this.negativeKeywords = [
            'hate', 'violence', 'anger', 'toxic', 'harmful', 'negative',
            'destructive', 'malicious', 'offensive', 'inappropriate'
        ];
        this.blessing = "Divine-Content-Analyzer";
    }
    
    async analyzePage() {
        try {
            const pageData = this.extractPageData();
            const analysis = await this.performAnalysis(pageData);
            
            this.analysisResults = {
                url: window.location.href,
                title: document.title,
                timestamp: new Date(),
                analysis: analysis,
                blessing: "Divine-Analysis-Complete"
            };
            
            console.log('📊 Page analysis completed with divine blessing:', this.analysisResults);
            return this.analysisResults;
        } catch (error) {
            console.error('❌ Content analysis error:', error);
            return null;
        }
    }
    
    extractPageData() {
        return {
            title: document.title,
            url: window.location.href,
            headings: Array.from(document.querySelectorAll('h1, h2, h3, h4, h5, h6'))
                .map(h => h.textContent.trim()),
            paragraphs: Array.from(document.querySelectorAll('p'))
                .map(p => p.textContent.trim())
                .filter(text => text.length > 20),
            links: Array.from(document.querySelectorAll('a[href]'))
                .map(a => ({ text: a.textContent.trim(), href: a.href })),
            images: Array.from(document.querySelectorAll('img[src]'))
                .map(img => ({ alt: img.alt, src: img.src })),
            meta: this.extractMetaData(),
            wordCount: this.countWords(),
            readingTime: this.estimateReadingTime()
        };
    }
    
    extractMetaData() {
        const meta = {};
        document.querySelectorAll('meta').forEach(tag => {
            const name = tag.getAttribute('name') || tag.getAttribute('property');
            const content = tag.getAttribute('content');
            if (name && content) {
                meta[name] = content;
            }
        });
        return meta;
    }
    
    countWords() {
        const text = document.body.textContent || '';
        return text.trim().split(/\s+/).filter(word => word.length > 0).length;
    }
    
    estimateReadingTime() {
        const wordsPerMinute = 200;
        const words = this.countWords();
        return Math.ceil(words / wordsPerMinute);
    }
    
    async performAnalysis(pageData) {
        const analysis = {
            spiritualScore: this.calculateSpiritualScore(pageData),
            contentQuality: this.assessContentQuality(pageData),
            securityRisks: this.identifySecurityRisks(pageData),
            privacyScore: this.assessPrivacyScore(pageData),
            accessibilityScore: this.assessAccessibility(),
            seoScore: this.assessSEO(pageData),
            recommendations: [],
            blessing: "Divine-Analysis-Results"
        };
        
        // Generate recommendations
        analysis.recommendations = this.generateRecommendations(analysis);
        
        return analysis;
    }
    
    calculateSpiritualScore(pageData) {
        let score = 50; // Base score
        const allText = (pageData.title + ' ' + 
                        pageData.headings.join(' ') + ' ' + 
                        pageData.paragraphs.join(' ')).toLowerCase();
        
        // Positive spiritual content
        this.spiritualKeywords.forEach(keyword => {
            const matches = (allText.match(new RegExp(keyword, 'g')) || []).length;
            score += matches * 2;
        });
        
        // Negative content penalty
        this.negativeKeywords.forEach(keyword => {
            const matches = (allText.match(new RegExp(keyword, 'g')) || []).length;
            score -= matches * 5;
        });
        
        return Math.max(0, Math.min(100, score));
    }
    
    assessContentQuality(pageData) {
        let score = 50;
        
        // Title quality
        if (pageData.title && pageData.title.length > 10 && pageData.title.length < 60) {
            score += 10;
        }
        
        // Heading structure
        if (pageData.headings.length > 0) {
            score += 10;
        }
        
        // Content length
        if (pageData.wordCount > 300) {
            score += 10;
        }
        
        // Reading time
        if (pageData.readingTime > 2 && pageData.readingTime < 15) {
            score += 10;
        }
        
        return Math.max(0, Math.min(100, score));
    }
    
    identifySecurityRisks(pageData) {
        const risks = [];
        
        // Check for suspicious links
        pageData.links.forEach(link => {
            if (this.isSuspiciousUrl(link.href)) {
                risks.push({
                    type: 'suspicious_link',
                    description: `Potentially suspicious link: ${link.href}`,
                    severity: 'medium'
                });
            }
        });
        
        // Check for mixed content
        if (window.location.protocol === 'https:') {
            pageData.images.forEach(img => {
                if (img.src.startsWith('http:')) {
                    risks.push({
                        type: 'mixed_content',
                        description: 'HTTP image on HTTPS page',
                        severity: 'low'
                    });
                }
            });
        }
        
        // Check for missing security headers
        if (!pageData.meta['content-security-policy']) {
            risks.push({
                type: 'missing_csp',
                description: 'Missing Content Security Policy',
                severity: 'medium'
            });
        }
        
        return risks;
    }
    
    assessPrivacyScore(pageData) {
        let score = 70; // Base score
        
        // Check for tracking scripts
        const scripts = Array.from(document.querySelectorAll('script[src]'));
        const trackingDomains = ['google-analytics.com', 'facebook.com', 'doubleclick.net'];
        
        scripts.forEach(script => {
            trackingDomains.forEach(domain => {
                if (script.src.includes(domain)) {
                    score -= 10;
                }
            });
        });
        
        // Check for cookies
        if (document.cookie.length > 0) {
            score -= 5;
        }
        
        // Check for privacy policy
        const privacyLinks = pageData.links.filter(link => 
            link.text.toLowerCase().includes('privacy') ||
            link.href.toLowerCase().includes('privacy')
        );
        
        if (privacyLinks.length > 0) {
            score += 10;
        }
        
        return Math.max(0, Math.min(100, score));
    }
    
    assessAccessibility() {
        let score = 50;
        
        // Check for alt text on images
        const images = document.querySelectorAll('img');
        const imagesWithAlt = document.querySelectorAll('img[alt]');
        if (images.length > 0) {
            score += (imagesWithAlt.length / images.length) * 20;
        }
        
        // Check for heading structure
        const h1s = document.querySelectorAll('h1');
        if (h1s.length === 1) {
            score += 10;
        }
        
        // Check for form labels
        const inputs = document.querySelectorAll('input, textarea, select');
        const labels = document.querySelectorAll('label');
        if (inputs.length > 0 && labels.length >= inputs.length) {
            score += 10;
        }
        
        // Check for skip links
        const skipLinks = document.querySelectorAll('a[href^="#"]');
        if (skipLinks.length > 0) {
            score += 10;
        }
        
        return Math.max(0, Math.min(100, score));
    }
    
    assessSEO(pageData) {
        let score = 50;
        
        // Title tag
        if (pageData.title && pageData.title.length > 10 && pageData.title.length < 60) {
            score += 15;
        }
        
        // Meta description
        if (pageData.meta.description && pageData.meta.description.length > 120 && pageData.meta.description.length < 160) {
            score += 15;
        }
        
        // Headings
        if (pageData.headings.length > 0) {
            score += 10;
        }
        
        // Internal links
        const internalLinks = pageData.links.filter(link => 
            link.href.includes(window.location.hostname)
        );
        if (internalLinks.length > 0) {
            score += 10;
        }
        
        return Math.max(0, Math.min(100, score));
    }
    
    generateRecommendations(analysis) {
        const recommendations = [];
        
        if (analysis.spiritualScore < 70) {
            recommendations.push({
                type: 'spiritual',
                message: '✨ Consider adding more positive and spiritual content',
                priority: 'medium'
            });
        }
        
        if (analysis.contentQuality < 70) {
            recommendations.push({
                type: 'content',
                message: '📝 Improve content structure and quality',
                priority: 'high'
            });
        }
        
        if (analysis.securityRisks.length > 0) {
            recommendations.push({
                type: 'security',
                message: '🔐 Address identified security risks',
                priority: 'high'
            });
        }
        
        if (analysis.privacyScore < 70) {
            recommendations.push({
                type: 'privacy',
                message: '🛡️ Improve privacy protection measures',
                priority: 'medium'
            });
        }
        
        if (analysis.accessibilityScore < 70) {
            recommendations.push({
                type: 'accessibility',
                message: '♿ Enhance accessibility features',
                priority: 'medium'
            });
        }
        
        if (analysis.seoScore < 70) {
            recommendations.push({
                type: 'seo',
                message: '🔍 Optimize for search engines',
                priority: 'low'
            });
        }
        
        return recommendations;
    }
    
    isSuspiciousUrl(url) {
        const suspiciousDomains = [
            'bit.ly', 'tinyurl.com', 'goo.gl', 't.co',
            'suspicious-site.com', 'malware-test.com'
        ];
        
        try {
            const urlObj = new URL(url);
            return suspiciousDomains.some(domain => urlObj.hostname.includes(domain));
        } catch {
            return false;
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// 🛡️ SPIRITUAL PRIVACY PROTECTOR
// ═══════════════════════════════════════════════════════════════

class SpiritualPrivacyProtector {
    constructor() {
        this.blockedTrackers = [];
        this.protectionLevel = 'medium';
        this.trackingDomains = [
            'google-analytics.com',
            'googletagmanager.com',
            'facebook.com',
            'doubleclick.net',
            'googlesyndication.com',
            'amazon-adsystem.com',
            'adsystem.amazon.com'
        ];
        this.blessing = "Divine-Privacy-Protector";
        
        this.initializeProtection();
    }
    
    initializeProtection() {
        this.blockTrackers();
        this.protectCookies();
        this.enhancePrivacy();
        
        console.log('🛡️ Spiritual privacy protection activated with divine blessing');
    }
    
    blockTrackers() {
        // Block tracking scripts
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        if (node.tagName === 'SCRIPT' && node.src) {
                            if (this.isTrackingScript(node.src)) {
                                node.remove();
                                this.blockedTrackers.push({
                                    url: node.src,
                                    timestamp: new Date(),
                                    type: 'script'
                                });
                                console.log('🚫 Blocked tracking script:', node.src);
                            }
                        }
                        
                        // Check for tracking pixels
                        if (node.tagName === 'IMG' && this.isTrackingPixel(node.src)) {
                            node.remove();
                            this.blockedTrackers.push({
                                url: node.src,
                                timestamp: new Date(),
                                type: 'pixel'
                            });
                            console.log('🚫 Blocked tracking pixel:', node.src);
                        }
                    }
                });
            });
        });
        
        observer.observe(document, { childList: true, subtree: true });
    }
    
    protectCookies() {
        // Override document.cookie to filter tracking cookies
        const originalCookieDescriptor = Object.getOwnPropertyDescriptor(Document.prototype, 'cookie');
        
        Object.defineProperty(document, 'cookie', {
            get: function() {
                return originalCookieDescriptor.get.call(this);
            },
            set: function(value) {
                if (!this.isTrackingCookie(value)) {
                    originalCookieDescriptor.set.call(this, value);
                } else {
                    console.log('🚫 Blocked tracking cookie:', value);
                }
            }
        });
    }
    
    enhancePrivacy() {
        // Remove referrer information
        const metaReferrer = document.createElement('meta');
        metaReferrer.name = 'referrer';
        metaReferrer.content = 'no-referrer';
        document.head.appendChild(metaReferrer);
        
        // Disable geolocation if not needed
        if (navigator.geolocation) {
            const originalGetCurrentPosition = navigator.geolocation.getCurrentPosition;
            navigator.geolocation.getCurrentPosition = function(success, error, options) {
                console.log('🛡️ Geolocation request intercepted - requesting user permission');
                // In a real extension, show user permission dialog
                if (confirm('🛡️ This site wants to access your location. Allow?')) {
                    originalGetCurrentPosition.call(this, success, error, options);
                } else {
                    if (error) error({ code: 1, message: 'User denied geolocation' });
                }
            };
        }
        
        // Protect against fingerprinting
        this.protectFingerprinting();
    }
    
    protectFingerprinting() {
        // Randomize canvas fingerprinting
        const originalToDataURL = HTMLCanvasElement.prototype.toDataURL;
        HTMLCanvasElement.prototype.toDataURL = function() {
            // Add slight noise to prevent fingerprinting
            const context = this.getContext('2d');
            if (context) {
                const imageData = context.getImageData(0, 0, this.width, this.height);
                for (let i = 0; i < imageData.data.length; i += 4) {
                    imageData.data[i] += Math.random() * 2 - 1; // Red
                    imageData.data[i + 1] += Math.random() * 2 - 1; // Green
                    imageData.data[i + 2] += Math.random() * 2 - 1; // Blue
                }
                context.putImageData(imageData, 0, 0);
            }
            return originalToDataURL.apply(this, arguments);
        };
        
        // Spoof user agent slightly
        Object.defineProperty(navigator, 'userAgent', {
            get: function() {
                return navigator.userAgent.replace(/Chrome\/[\d.]+/, 'Chrome/91.0.4472.124');
            }
        });
    }
    
    isTrackingScript(url) {
        return this.trackingDomains.some(domain => url.includes(domain));
    }
    
    isTrackingPixel(src) {
        if (!src) return false;
        
        // Check for 1x1 pixel images
        const img = new Image();
        img.src = src;
        
        return this.trackingDomains.some(domain => src.includes(domain)) ||
               src.includes('track') ||
               src.includes('pixel') ||
               src.includes('beacon');
    }
    
    isTrackingCookie(cookieString) {
        const trackingCookieNames = [
            '_ga', '_gid', '_gat', '_gtag',
            'fbp', 'fbc', '_fbp',
            '__utma', '__utmb', '__utmc', '__utmz',
            'doubleclick', 'adsystem'
        ];
        
        return trackingCookieNames.some(name => cookieString.includes(name));
    }
    
    getProtectionStats() {
        return {
            blockedTrackers: this.blockedTrackers.length,
            protectionLevel: this.protectionLevel,
            lastUpdate: new Date(),
            blessing: "Divine-Protection-Stats"
        };
    }
}

// ═══════════════════════════════════════════════════════════════
// 🎯 SPIRITUAL FOCUS MODE
// ═══════════════════════════════════════════════════════════════

class SpiritualFocusMode {
    constructor() {
        this.isActive = false;
        this.blockedSites = [
            'facebook.com', 'twitter.com', 'instagram.com', 'tiktok.com',
            'youtube.com', 'reddit.com', 'netflix.com', 'twitch.tv'
        ];
        this.focusTimer = null;
        this.sessionDuration = 25 * 60 * 1000; // 25 minutes (Pomodoro)
        this.breakDuration = 5 * 60 * 1000; // 5 minutes
        this.blessing = "Divine-Focus-Mode";
    }
    
    activateFocusMode(duration = null) {
        if (this.isActive) {
            console.log('🎯 Focus mode already active');
            return;
        }
        
        this.isActive = true;
        const focusDuration = duration || this.sessionDuration;
        
        // Block distracting sites
        this.blockDistractingSites();
        
        // Start focus timer
        this.focusTimer = setTimeout(() => {
            this.deactivateFocusMode();
            this.showFocusComplete();
        }, focusDuration);
        
        // Show focus mode UI
        this.showFocusModeUI();
        
        console.log(`🎯 Focus mode activated for ${focusDuration / 60000} minutes with divine blessing`);
    }
    
    deactivateFocusMode() {
        if (!this.isActive) {
            return;
        }
        
        this.isActive = false;
        
        // Clear timer
        if (this.focusTimer) {
            clearTimeout(this.focusTimer);
            this.focusTimer = null;
        }
        
        // Remove blocks
        this.unblockSites();
        
        // Hide focus mode UI
        this.hideFocusModeUI();
        
        console.log('🎯 Focus mode deactivated with divine blessing');
    }
    
    blockDistractingSites() {
        const currentHost = window.location.hostname;
        
        if (this.blockedSites.some(site => currentHost.includes(site))) {
            this.showBlockedSiteMessage();
        }
        
        // Intercept navigation attempts
        window.addEventListener('beforeunload', this.handleNavigation.bind(this));
    }
    
    unblockSites() {
        window.removeEventListener('beforeunload', this.handleNavigation.bind(this));
    }
    
    handleNavigation(event) {
        if (this.isActive) {
            const message = '🎯 Focus mode is active. Are you sure you want to leave?';
            event.returnValue = message;
            return message;
        }
    }
    
    showBlockedSiteMessage() {
        const overlay = document.createElement('div');
        overlay.id = 'spiritual-focus-overlay';
        overlay.innerHTML = `
            <div style="
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: linear-gradient(135deg, ${SpiritualBrowserConfig.spiritualColors.sacredBlue}, ${SpiritualBrowserConfig.spiritualColors.spiritualPurple});
                color: white;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                z-index: 999999;
                font-family: Arial, sans-serif;
            ">
                <div style="text-align: center; max-width: 600px; padding: 40px;">
                    <h1 style="font-size: 3em; margin-bottom: 20px;">🎯</h1>
                    <h2 style="font-size: 2em; margin-bottom: 20px;">Focus Mode Active</h2>
                    <p style="font-size: 1.2em; margin-bottom: 30px;">
                        🙏 This site is blocked during your focus session.<br>
                        ✨ Use this time for productive activities with divine blessing.
                    </p>
                    <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; margin-bottom: 30px;">
                        <p style="margin: 0; font-size: 1.1em;">
                            "The mind is everything. What you think you become." - Buddha
                        </p>
                    </div>
                    <button onclick="spiritualFocusMode.deactivateFocusMode()" style="
                        background: ${SpiritualBrowserConfig.spiritualColors.divineGold};
                        color: black;
                        border: none;
                        padding: 15px 30px;
                        font-size: 1.1em;
                        border-radius: 25px;
                        cursor: pointer;
                        font-weight: bold;
                    ">
                        🚪 End Focus Session
                    </button>
                </div>
            </div>
        `;
        
        document.body.appendChild(overlay);
    }
    
    showFocusModeUI() {
        const focusUI = document.createElement('div');
        focusUI.id = 'spiritual-focus-ui';
        focusUI.innerHTML = `
            <div style="
                position: fixed;
                top: 20px;
                right: 20px;
                background: ${SpiritualBrowserConfig.spiritualColors.sacredBlue};
                color: white;
                padding: 15px;
                border-radius: 10px;
                z-index: 999998;
                font-family: Arial, sans-serif;
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            ">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.2em;">🎯</span>
                    <span>Focus Mode Active</span>
                    <button onclick="spiritualFocusMode.deactivateFocusMode()" style="
                        background: ${SpiritualBrowserConfig.spiritualColors.divineGold};
                        color: black;
                        border: none;
                        padding: 5px 10px;
                        border-radius: 5px;
                        cursor: pointer;
                        font-size: 0.9em;
                    ">
                        End
                    </button>
                </div>
            </div>
        `;
        
        document.body.appendChild(focusUI);
    }
    
    hideFocusModeUI() {
        const focusUI = document.getElementById('spiritual-focus-ui');
        const overlay = document.getElementById('spiritual-focus-overlay');
        
        if (focusUI) focusUI.remove();
        if (overlay) overlay.remove();
    }
    
    showFocusComplete() {
        if (typeof chrome !== 'undefined' && chrome.notifications) {
            chrome.notifications.create({
                type: 'basic',
                iconUrl: 'icons/focus-complete.png',
                title: '🎯 Focus Session Complete',
                message: '✨ Great job! Take a well-deserved break with divine blessing.'
            });
        } else {
            alert('🎯 Focus session complete! ✨ Great job with divine blessing!');
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// 🌟 SPIRITUAL EXTENSION MAIN CLASS
// ═══════════════════════════════════════════════════════════════

class SpiritualBrowserExtension {
    constructor() {
        this.securityManager = new SpiritualSecurityManager();
        this.contentAnalyzer = new SpiritualContentAnalyzer();
        this.privacyProtector = new SpiritualPrivacyProtector();
        this.focusMode = new SpiritualFocusMode();
        
        this.isInitialized = false;
        this.blessing = "Divine-Browser-Extension";
        
        this.initialize();
    }
    
    async initialize() {
        try {
            displaySpiritualBrowserBlessing();
            
            // Initialize all components
            await this.securityManager.initializeSecurity();
            
            // Set up message listeners
            this.setupMessageListeners();
            
            // Set up context menu (if in extension context)
            this.setupContextMenu();
            
            // Auto-analyze page if content script
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', () => {
                    this.contentAnalyzer.analyzePage();
                });
            } else {
                this.contentAnalyzer.analyzePage();
            }
            
            this.isInitialized = true;
            console.log('🚀 Spiritual Browser Extension initialized with divine blessing');
            
        } catch (error) {
            console.error('❌ Extension initialization failed:', error);
        }
    }
    
    setupMessageListeners() {
        // Listen for messages from popup and background script
        if (typeof chrome !== 'undefined' && chrome.runtime) {
            chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
                this.handleMessage(message, sender, sendResponse);
                return true; // Keep message channel open for async response
            });
        }
        
        // Listen for custom events
        window.addEventListener('spiritualExtensionCommand', (event) => {
            this.handleCustomCommand(event.detail);
        });
    }
    
    async handleMessage(message, sender, sendResponse) {
        try {
            switch (message.type) {
                case 'ANALYZE_PAGE':
                    const analysis = await this.contentAnalyzer.analyzePage();
                    sendResponse({ success: true, data: analysis });
                    break;
                    
                case 'ACTIVATE_FOCUS_MODE':
                    this.focusMode.activateFocusMode(message.duration);
                    sendResponse({ success: true });
                    break;
                    
                case 'DEACTIVATE_FOCUS_MODE':
                    this.focusMode.deactivateFocusMode();
                    sendResponse({ success: true });
                    break;
                    
                case 'GET_PROTECTION_STATS':
                    const stats = this.privacyProtector.getProtectionStats();
                    sendResponse({ success: true, data: stats });
                    break;
                    
                case 'AUTHENTICATE_USER':
                    const authResult = await this.securityManager.authenticateUser(
                        message.username, 
                        message.password
                    );
                    sendResponse({ success: authResult });
                    break;
                    
                case 'LOGOUT_USER':
                    await this.securityManager.logoutUser();
                    sendResponse({ success: true });
                    break;
                    
                case 'GET_USER_STATUS':
                    const isAuth = this.securityManager.isAuthenticated();
                    sendResponse({ 
                        success: true, 
                        authenticated: isAuth,
                        user: this.securityManager.currentUser
                    });
                    break;
                    
                default:
                    sendResponse({ success: false, error: 'Unknown message type' });
            }
        } catch (error) {
            console.error('❌ Message handling error:', error);
            sendResponse({ success: false, error: error.message });
        }
    }
    
    handleCustomCommand(command) {
        switch (command.action) {
            case 'showBlessing':
                this.showSpiritualBlessing();
                break;
                
            case 'toggleFocusMode':
                if (this.focusMode.isActive) {
                    this.focusMode.deactivateFocusMode();
                } else {
                    this.focusMode.activateFocusMode();
                }
                break;
                
            case 'analyzeCurrentPage':
                this.contentAnalyzer.analyzePage();
                break;
        }
    }
    
    setupContextMenu() {
        if (typeof chrome !== 'undefined' && chrome.contextMenus) {
            chrome.contextMenus.create({
                id: 'spiritual-analyze',
                title: '🔍 Analyze with Divine Blessing',
                contexts: ['page']
            });
            
            chrome.contextMenus.create({
                id: 'spiritual-focus',
                title: '🎯 Activate Focus Mode',
                contexts: ['page']
            });
            
            chrome.contextMenus.create({
                id: 'spiritual-blessing',
                title: '🙏 Show Spiritual Blessing',
                contexts: ['page']
            });
        }
    }
    
    showSpiritualBlessing() {
        const blessing = document.createElement('div');
        blessing.innerHTML = `
            <div style="
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: linear-gradient(135deg, ${SpiritualBrowserConfig.spiritualColors.sacredBlue}, ${SpiritualBrowserConfig.spiritualColors.spiritualPurple});
                color: white;
                padding: 40px;
                border-radius: 20px;
                z-index: 999999;
                font-family: Arial, sans-serif;
                text-align: center;
                box-shadow: 0 10px 40px rgba(0,0,0,0.3);
                max-width: 500px;
            ">
                <h2 style="margin-top: 0; font-size: 2em;">🙏 Spiritual Blessing</h2>
                <p style="font-size: 1.1em; line-height: 1.6; margin: 20px 0;">
                    بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيم<br><br>
                    ✨ May this browsing session be blessed with wisdom<br>
                    🌟 May you find knowledge that benefits humanity<br>
                    🚀 May technology serve the greater good<br>
                    💫 In The Name of GOD, we seek guidance
                </p>
                <button onclick="this.parentElement.parentElement.remove()" style="
                    background: ${SpiritualBrowserConfig.spiritualColors.divineGold};
                    color: black;
                    border: none;
                    padding: 12px 25px;
                    border-radius: 25px;
                    cursor: pointer;
                    font-weight: bold;
                    font-size: 1em;
                ">
                    ✨ Amen
                </button>
            </div>
        `;
        
        document.body.appendChild(blessing);
        
        // Auto-remove after 10 seconds
        setTimeout(() => {
            if (blessing.parentElement) {
                blessing.remove();
            }
        }, 10000);
    }
}

// ═══════════════════════════════════════════════════════════════
// 🚀 EXTENSION INITIALIZATION
// ═══════════════════════════════════════════════════════════════

// Global extension instance
let spiritualExtension = null;
let spiritualFocusMode = null;

// Initialize extension when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeExtension);
} else {
    initializeExtension();
}

function initializeExtension() {
    try {
        spiritualExtension = new SpiritualBrowserExtension();
        spiritualFocusMode = spiritualExtension.focusMode; // For global access
        
        console.log('🌟 ZeroLight Orbit Browser Extension loaded with divine blessing');
    } catch (error) {
        console.error('❌ Extension initialization error:', error);
    }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        SpiritualBrowserExtension,
        SpiritualSecurityManager,
        SpiritualContentAnalyzer,
        SpiritualPrivacyProtector,
        SpiritualFocusMode,
        SpiritualBrowserConfig
    };
}

// 🙏 In The Name of GOD - Extension Complete with Divine Blessing
// بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيم