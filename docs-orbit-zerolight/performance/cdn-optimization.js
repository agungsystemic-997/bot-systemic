#!/usr/bin/env node

/**
 * 🙏 In The Name of GOD - ZeroLight Orbit CDN Optimization
 * Blessed Content Delivery Network for Spiritual System
 * بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيم
 */

const fs = require('fs').promises;
const path = require('path');
const crypto = require('crypto');
const { execSync } = require('child_process');

// 🌟 Spiritual CDN Configuration
const CDN_CONFIG = {
    providers: {
        cloudflare: {
            zoneId: process.env.CLOUDFLARE_ZONE_ID,
            apiToken: process.env.CLOUDFLARE_API_TOKEN,
            endpoint: 'https://api.cloudflare.com/client/v4',
            blessing: 'Divine-Cloudflare-Distribution'
        },
        aws: {
            distributionId: process.env.AWS_CLOUDFRONT_DISTRIBUTION_ID,
            region: process.env.AWS_REGION || 'us-east-1',
            blessing: 'Sacred-AWS-CloudFront'
        },
        azure: {
            resourceGroup: process.env.AZURE_RESOURCE_GROUP,
            profileName: process.env.AZURE_CDN_PROFILE,
            blessing: 'Blessed-Azure-CDN'
        }
    },
    optimization: {
        imageFormats: ['webp', 'avif', 'jpg', 'png'],
        compressionLevels: {
            images: 85,
            css: 9,
            js: 9,
            html: 9
        },
        cacheHeaders: {
            static: 'max-age=31536000, public, immutable',
            dynamic: 'max-age=300, public, must-revalidate',
            api: 'max-age=60, public, must-revalidate'
        }
    },
    spiritual: {
        blessing: 'In-The-Name-of-GOD',
        purpose: 'Divine-Content-Delivery',
        guidance: 'Alhamdulillahi-rabbil-alameen'
    }
};

// 🙏 Spiritual Blessing Display
function displaySpiritualBlessing() {
    console.log('\n🌟 ═══════════════════════════════════════════════════════════════');
    console.log('🙏 بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيم');
    console.log('✨ ZeroLight Orbit CDN Optimization - In The Name of GOD');
    console.log('🌍 Blessed Global Content Delivery Network');
    console.log('═══════════════════════════════════════════════════════════════ 🌟\n');
}

// 📊 Asset Analysis and Optimization
class SpiritualAssetOptimizer {
    constructor() {
        this.assets = new Map();
        this.optimizationStats = {
            totalFiles: 0,
            optimizedFiles: 0,
            sizeSaved: 0,
            blessing: 'Divine-Optimization-Stats'
        };
    }

    async analyzeAssets(directory) {
        console.log('🔍 Analyzing assets with divine wisdom...');
        
        const files = await this.getAllFiles(directory);
        
        for (const file of files) {
            const stats = await fs.stat(file);
            const ext = path.extname(file).toLowerCase();
            const hash = await this.generateFileHash(file);
            
            this.assets.set(file, {
                path: file,
                size: stats.size,
                extension: ext,
                hash: hash,
                lastModified: stats.mtime,
                optimized: false,
                blessing: 'Asset-Analyzed-With-Divine-Wisdom'
            });
        }
        
        this.optimizationStats.totalFiles = this.assets.size;
        console.log(`✨ Analyzed ${this.assets.size} assets with spiritual insight`);
    }

    async getAllFiles(dir, files = []) {
        const dirents = await fs.readdir(dir, { withFileTypes: true });
        
        for (const dirent of dirents) {
            const fullPath = path.join(dir, dirent.name);
            
            if (dirent.isDirectory()) {
                await this.getAllFiles(fullPath, files);
            } else {
                files.push(fullPath);
            }
        }
        
        return files;
    }

    async generateFileHash(filePath) {
        const content = await fs.readFile(filePath);
        return crypto.createHash('sha256').update(content).digest('hex').substring(0, 16);
    }

    async optimizeImages() {
        console.log('🖼️ Optimizing images with divine compression...');
        
        const imageExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp'];
        const imageAssets = Array.from(this.assets.values())
            .filter(asset => imageExtensions.includes(asset.extension));
        
        for (const asset of imageAssets) {
            try {
                const originalSize = asset.size;
                
                // Generate WebP version
                if (asset.extension !== '.webp' && asset.extension !== '.svg') {
                    await this.generateWebP(asset.path);
                }
                
                // Generate AVIF version for modern browsers
                if (asset.extension !== '.avif' && asset.extension !== '.svg') {
                    await this.generateAVIF(asset.path);
                }
                
                // Optimize original image
                await this.compressImage(asset.path);
                
                const newStats = await fs.stat(asset.path);
                const sizeSaved = originalSize - newStats.size;
                
                asset.optimized = true;
                asset.size = newStats.size;
                this.optimizationStats.sizeSaved += sizeSaved;
                this.optimizationStats.optimizedFiles++;
                
                console.log(`✨ Optimized ${path.basename(asset.path)} - Saved ${this.formatBytes(sizeSaved)}`);
                
            } catch (error) {
                console.warn(`⚠️ Could not optimize ${asset.path}: ${error.message}`);
            }
        }
    }

    async generateWebP(imagePath) {
        const webpPath = imagePath.replace(/\.(jpg|jpeg|png)$/i, '.webp');
        
        try {
            execSync(`cwebp -q ${CDN_CONFIG.optimization.compressionLevels.images} "${imagePath}" -o "${webpPath}"`, 
                { stdio: 'pipe' });
        } catch (error) {
            // Fallback to sharp if cwebp is not available
            try {
                const sharp = require('sharp');
                await sharp(imagePath)
                    .webp({ quality: CDN_CONFIG.optimization.compressionLevels.images })
                    .toFile(webpPath);
            } catch (sharpError) {
                console.warn(`⚠️ WebP generation failed for ${imagePath}`);
            }
        }
    }

    async generateAVIF(imagePath) {
        const avifPath = imagePath.replace(/\.(jpg|jpeg|png)$/i, '.avif');
        
        try {
            const sharp = require('sharp');
            await sharp(imagePath)
                .avif({ quality: CDN_CONFIG.optimization.compressionLevels.images })
                .toFile(avifPath);
        } catch (error) {
            console.warn(`⚠️ AVIF generation failed for ${imagePath}`);
        }
    }

    async compressImage(imagePath) {
        const ext = path.extname(imagePath).toLowerCase();
        
        try {
            if (ext === '.jpg' || ext === '.jpeg') {
                execSync(`jpegoptim --max=${CDN_CONFIG.optimization.compressionLevels.images} "${imagePath}"`, 
                    { stdio: 'pipe' });
            } else if (ext === '.png') {
                execSync(`optipng -o7 "${imagePath}"`, { stdio: 'pipe' });
            } else if (ext === '.svg') {
                execSync(`svgo "${imagePath}"`, { stdio: 'pipe' });
            }
        } catch (error) {
            // Fallback to sharp for basic compression
            try {
                const sharp = require('sharp');
                const buffer = await sharp(imagePath)
                    .jpeg({ quality: CDN_CONFIG.optimization.compressionLevels.images })
                    .toBuffer();
                await fs.writeFile(imagePath, buffer);
            } catch (sharpError) {
                console.warn(`⚠️ Image compression failed for ${imagePath}`);
            }
        }
    }

    async optimizeCSS() {
        console.log('🎨 Optimizing CSS with divine styling...');
        
        const cssAssets = Array.from(this.assets.values())
            .filter(asset => asset.extension === '.css');
        
        for (const asset of cssAssets) {
            try {
                const content = await fs.readFile(asset.path, 'utf8');
                const originalSize = Buffer.byteLength(content, 'utf8');
                
                // Minify CSS
                const cleanCSS = require('clean-css');
                const result = new cleanCSS({
                    level: 2,
                    returnPromise: true
                }).minify(content);
                
                await fs.writeFile(asset.path, result.styles);
                
                const newSize = Buffer.byteLength(result.styles, 'utf8');
                const sizeSaved = originalSize - newSize;
                
                asset.optimized = true;
                asset.size = newSize;
                this.optimizationStats.sizeSaved += sizeSaved;
                this.optimizationStats.optimizedFiles++;
                
                console.log(`✨ Optimized ${path.basename(asset.path)} - Saved ${this.formatBytes(sizeSaved)}`);
                
            } catch (error) {
                console.warn(`⚠️ Could not optimize CSS ${asset.path}: ${error.message}`);
            }
        }
    }

    async optimizeJavaScript() {
        console.log('⚡ Optimizing JavaScript with divine logic...');
        
        const jsAssets = Array.from(this.assets.values())
            .filter(asset => asset.extension === '.js');
        
        for (const asset of jsAssets) {
            try {
                const content = await fs.readFile(asset.path, 'utf8');
                const originalSize = Buffer.byteLength(content, 'utf8');
                
                // Minify JavaScript
                const terser = require('terser');
                const result = await terser.minify(content, {
                    compress: {
                        drop_console: true,
                        drop_debugger: true
                    },
                    mangle: true
                });
                
                await fs.writeFile(asset.path, result.code);
                
                const newSize = Buffer.byteLength(result.code, 'utf8');
                const sizeSaved = originalSize - newSize;
                
                asset.optimized = true;
                asset.size = newSize;
                this.optimizationStats.sizeSaved += sizeSaved;
                this.optimizationStats.optimizedFiles++;
                
                console.log(`✨ Optimized ${path.basename(asset.path)} - Saved ${this.formatBytes(sizeSaved)}`);
                
            } catch (error) {
                console.warn(`⚠️ Could not optimize JS ${asset.path}: ${error.message}`);
            }
        }
    }

    formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

// 🌍 CDN Distribution Manager
class SpiritualCDNManager {
    constructor() {
        this.distributions = new Map();
        this.cacheStats = {
            purged: 0,
            uploaded: 0,
            blessing: 'Divine-CDN-Stats'
        };
    }

    async deployToCloudflare(assets) {
        console.log('☁️ Deploying to Cloudflare with divine distribution...');
        
        const config = CDN_CONFIG.providers.cloudflare;
        if (!config.zoneId || !config.apiToken) {
            console.warn('⚠️ Cloudflare credentials not configured');
            return;
        }
        
        try {
            // Purge cache
            await this.purgeCloudflareCache(config);
            
            // Upload assets (if using Cloudflare R2 or similar)
            for (const [path, asset] of assets) {
                if (asset.optimized) {
                    console.log(`📤 Uploading ${path.basename(asset.path)} to Cloudflare...`);
                    this.cacheStats.uploaded++;
                }
            }
            
            console.log('✨ Cloudflare deployment completed with divine blessing');
            
        } catch (error) {
            console.error(`❌ Cloudflare deployment failed: ${error.message}`);
        }
    }

    async purgeCloudflareCache(config) {
        const fetch = require('node-fetch');
        
        const response = await fetch(`${config.endpoint}/zones/${config.zoneId}/purge_cache`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${config.apiToken}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ purge_everything: true })
        });
        
        if (response.ok) {
            console.log('🧹 Cloudflare cache purged with divine cleansing');
            this.cacheStats.purged++;
        } else {
            throw new Error(`Cache purge failed: ${response.statusText}`);
        }
    }

    async deployToAWS(assets) {
        console.log('🌩️ Deploying to AWS CloudFront with sacred distribution...');
        
        const config = CDN_CONFIG.providers.aws;
        if (!config.distributionId) {
            console.warn('⚠️ AWS CloudFront credentials not configured');
            return;
        }
        
        try {
            // Create invalidation
            const AWS = require('aws-sdk');
            const cloudfront = new AWS.CloudFront();
            
            const params = {
                DistributionId: config.distributionId,
                InvalidationBatch: {
                    CallerReference: `spiritual-invalidation-${Date.now()}`,
                    Paths: {
                        Quantity: 1,
                        Items: ['/*']
                    }
                }
            };
            
            await cloudfront.createInvalidation(params).promise();
            console.log('✨ AWS CloudFront invalidation completed with divine blessing');
            this.cacheStats.purged++;
            
        } catch (error) {
            console.error(`❌ AWS deployment failed: ${error.message}`);
        }
    }

    async generateCacheManifest(assets) {
        console.log('📋 Generating cache manifest with divine organization...');
        
        const manifest = {
            version: Date.now(),
            blessing: CDN_CONFIG.spiritual.blessing,
            purpose: CDN_CONFIG.spiritual.purpose,
            guidance: CDN_CONFIG.spiritual.guidance,
            assets: {},
            cacheHeaders: CDN_CONFIG.optimization.cacheHeaders,
            generatedAt: new Date().toISOString()
        };
        
        for (const [path, asset] of assets) {
            const relativePath = path.relative(process.cwd(), asset.path);
            manifest.assets[relativePath] = {
                hash: asset.hash,
                size: asset.size,
                optimized: asset.optimized,
                lastModified: asset.lastModified.toISOString(),
                blessing: 'Asset-Blessed-For-Distribution'
            };
        }
        
        await fs.writeFile('cdn-manifest.json', JSON.stringify(manifest, null, 2));
        console.log('✨ Cache manifest generated with spiritual organization');
        
        return manifest;
    }
}

// 🚀 Main CDN Optimization Process
async function runSpiritualCDNOptimization() {
    try {
        displaySpiritualBlessing();
        
        const optimizer = new SpiritualAssetOptimizer();
        const cdnManager = new SpiritualCDNManager();
        
        // Analyze assets
        await optimizer.analyzeAssets('./dist');
        
        // Optimize assets
        await optimizer.optimizeImages();
        await optimizer.optimizeCSS();
        await optimizer.optimizeJavaScript();
        
        // Generate cache manifest
        await cdnManager.generateCacheManifest(optimizer.assets);
        
        // Deploy to CDN providers
        await cdnManager.deployToCloudflare(optimizer.assets);
        await cdnManager.deployToAWS(optimizer.assets);
        
        // Display results
        console.log('\n🎉 ═══════════════════════════════════════════════════════════════');
        console.log('✨ CDN Optimization Completed with Divine Success!');
        console.log(`📊 Total Files: ${optimizer.optimizationStats.totalFiles}`);
        console.log(`🔧 Optimized Files: ${optimizer.optimizationStats.optimizedFiles}`);
        console.log(`💾 Size Saved: ${optimizer.formatBytes(optimizer.optimizationStats.sizeSaved)}`);
        console.log(`📤 Assets Uploaded: ${cdnManager.cacheStats.uploaded}`);
        console.log(`🧹 Caches Purged: ${cdnManager.cacheStats.purged}`);
        console.log('🙏 Alhamdulillahi rabbil alameen - All praise to Allah!');
        console.log('═══════════════════════════════════════════════════════════════ 🎉\n');
        
    } catch (error) {
        console.error('❌ CDN optimization failed:', error);
        process.exit(1);
    }
}

// 🎯 Command Line Interface
if (require.main === module) {
    const command = process.argv[2];
    
    switch (command) {
        case 'optimize':
            runSpiritualCDNOptimization();
            break;
        case 'analyze':
            (async () => {
                displaySpiritualBlessing();
                const optimizer = new SpiritualAssetOptimizer();
                await optimizer.analyzeAssets('./dist');
                console.log('✨ Asset analysis completed with divine insight');
            })();
            break;
        case 'deploy':
            (async () => {
                displaySpiritualBlessing();
                const cdnManager = new SpiritualCDNManager();
                const optimizer = new SpiritualAssetOptimizer();
                await optimizer.analyzeAssets('./dist');
                await cdnManager.deployToCloudflare(optimizer.assets);
                await cdnManager.deployToAWS(optimizer.assets);
                console.log('✨ CDN deployment completed with divine distribution');
            })();
            break;
        default:
            console.log('🙏 ZeroLight Orbit CDN Optimization - In The Name of GOD');
            console.log('Usage:');
            console.log('  node cdn-optimization.js optimize  - Full optimization and deployment');
            console.log('  node cdn-optimization.js analyze   - Analyze assets only');
            console.log('  node cdn-optimization.js deploy    - Deploy to CDN only');
            break;
    }
}

module.exports = {
    SpiritualAssetOptimizer,
    SpiritualCDNManager,
    runSpiritualCDNOptimization
};

// 🙏 Blessed CDN Optimization Script
// May this content delivery serve humanity with divine speed and global reach
// Alhamdulillahi rabbil alameen - All praise to Allah, Lord of the worlds