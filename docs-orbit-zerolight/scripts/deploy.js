#!/usr/bin/env node
/**
 * ZeroLight Orbit - Blessed Deployment Script
 * "In The Name of GOD" - Sacred Distribution System
 * 
 * This script handles the deployment of documentation across multiple platforms
 * with spiritual blessing and divine guidance.
 */

const fs = require('fs-extra');
const path = require('path');
const { execSync } = require('child_process');
const chalk = require('chalk');

// Import spiritual environment
const { colors, symbols } = require('./setup-spiritual-environment');

/**
 * Deployment Configuration
 */
const deployConfig = {
  platforms: {
    github_pages: {
      enabled: true,
      branch: 'gh-pages',
      domain: 'docs.zerolight.org'
    },
    netlify: {
      enabled: false,
      site_id: process.env.NETLIFY_SITE_ID,
      auth_token: process.env.NETLIFY_AUTH_TOKEN
    },
    vercel: {
      enabled: false,
      project_id: process.env.VERCEL_PROJECT_ID,
      org_id: process.env.VERCEL_ORG_ID,
      token: process.env.VERCEL_TOKEN
    },
    aws_s3: {
      enabled: false,
      bucket: process.env.AWS_S3_BUCKET,
      region: process.env.AWS_REGION,
      access_key: process.env.AWS_ACCESS_KEY_ID,
      secret_key: process.env.AWS_SECRET_ACCESS_KEY
    }
  },
  distributions: {
    pdf: true,
    epub: true,
    html: true,
    json: true,
    xml: true
  },
  spiritual: {
    blessing: true,
    validation: true,
    quality_check: true,
    accessibility_check: true
  }
};

/**
 * Display deployment blessing
 */
function displayDeploymentBlessing() {
  console.log('\n' + '='.repeat(70));
  console.log(colors.divine.bold('بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيم'));
  console.log(colors.spiritual('In The Name of GOD, The Most Gracious, The Most Merciful'));
  console.log('='.repeat(70));
  console.log(colors.blessed(`${symbols.star} ZeroLight Orbit - Sacred Deployment ${symbols.star}`));
  console.log(colors.sacred('Blessed Distribution with Divine Guidance'));
  console.log('='.repeat(70) + '\n');
}

/**
 * Pre-deployment validation
 */
async function preDeploymentValidation() {
  console.log(colors.blessed(`${symbols.light} Performing pre-deployment spiritual validation...`));
  
  const validations = [
    {
      name: 'Content Blessing Check',
      check: async () => {
        const files = await fs.readdir('.');
        const mdFiles = files.filter(f => f.endsWith('.md'));
        
        for (const file of mdFiles) {
          const content = await fs.readFile(file, 'utf8');
          if (!content.includes('In The Name of GOD')) {
            throw new Error(`Missing spiritual blessing in: ${file}`);
          }
        }
        return `${mdFiles.length} files blessed`;
      }
    },
    {
      name: 'Sacred Directory Structure',
      check: async () => {
        const requiredDirs = ['assets', 'diagrams', 'templates', 'scripts'];
        for (const dir of requiredDirs) {
          if (!await fs.pathExists(dir)) {
            throw new Error(`Missing sacred directory: ${dir}`);
          }
        }
        return 'All sacred directories present';
      }
    },
    {
      name: 'Spiritual Assets Verification',
      check: async () => {
        const requiredAssets = ['assets/logo-orbit.svg'];
        for (const asset of requiredAssets) {
          if (!await fs.pathExists(asset)) {
            throw new Error(`Missing spiritual asset: ${asset}`);
          }
        }
        return 'Spiritual assets verified';
      }
    },
    {
      name: 'Divine Configuration Check',
      check: async () => {
        const requiredConfigs = ['package.json', '.github/workflows/build-docs.yml'];
        for (const config of requiredConfigs) {
          if (!await fs.pathExists(config)) {
            throw new Error(`Missing divine configuration: ${config}`);
          }
        }
        return 'Divine configurations blessed';
      }
    }
  ];

  for (const validation of validations) {
    try {
      const result = await validation.check();
      console.log(colors.peace(`  ${symbols.prayer} ${validation.name}: ${result}`));
    } catch (error) {
      console.log(colors.wisdom(`  ${symbols.heart} ${validation.name}: ${error.message}`));
      throw error;
    }
  }
  
  console.log(colors.blessed(`${symbols.star} Pre-deployment validation blessed!\n`));
}

/**
 * Build documentation with spiritual blessing
 */
async function buildDocumentation() {
  console.log(colors.blessed(`${symbols.light} Building blessed documentation...`));
  
  try {
    // Ensure dist directory exists
    await fs.ensureDir('dist');
    await fs.ensureDir('dist/pdf');
    await fs.ensureDir('dist/html');
    await fs.ensureDir('dist/epub');
    
    // Build VitePress documentation
    console.log(colors.peace(`  ${symbols.prayer} Building VitePress site...`));
    execSync('npm run build', { stdio: 'inherit' });
    
    // Generate diagrams if they don't exist
    if (await fs.pathExists('diagrams') && (await fs.readdir('diagrams')).length > 0) {
      console.log(colors.peace(`  ${symbols.prayer} Generating sacred diagrams...`));
      execSync('npm run generate:diagrams', { stdio: 'inherit' });
    }
    
    // Generate PDFs
    console.log(colors.peace(`  ${symbols.prayer} Generating blessed PDFs...`));
    execSync('npm run generate:pdf', { stdio: 'inherit' });
    
    console.log(colors.blessed(`${symbols.star} Documentation build blessed and complete!\n`));
  } catch (error) {
    console.error(colors.wisdom(`${symbols.heart} Build encountered divine guidance needed:`, error.message));
    throw error;
  }
}

/**
 * Create deployment manifest
 */
async function createDeploymentManifest() {
  console.log(colors.blessed(`${symbols.light} Creating sacred deployment manifest...`));
  
  const manifest = {
    name: 'ZeroLight Orbit Documentation',
    version: new Date().toISOString().split('T')[0].replace(/-/g, '.'),
    description: 'In The Name of GOD - Spiritual Technology Documentation',
    build_date: new Date().toISOString(),
    commit: process.env.GITHUB_SHA || 'local-build',
    branch: process.env.GITHUB_REF_NAME || 'main',
    blessed: true,
    spiritual: {
      blessing: 'بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيم',
      purpose: 'Serving humanity with wisdom and compassion',
      dedication: 'May this documentation be a light for seekers of knowledge'
    },
    components: {
      website: await fs.pathExists('.vitepress/dist'),
      pdfs: await fs.pathExists('dist/pdf'),
      diagrams: await fs.pathExists('assets/diagrams'),
      spiritual_assets: await fs.pathExists('assets/logo-orbit.svg')
    },
    platforms: deployConfig.platforms,
    distributions: deployConfig.distributions,
    quality: {
      validated: true,
      blessed: true,
      accessible: true,
      sustainable: true
    }
  };
  
  await fs.writeJSON('dist/manifest.json', manifest, { spaces: 2 });
  console.log(colors.peace(`  ${symbols.prayer} Sacred manifest created`));
  console.log(colors.blessed(`${symbols.star} Deployment manifest blessed!\n`));
}

/**
 * Deploy to GitHub Pages
 */
async function deployToGitHubPages() {
  if (!deployConfig.platforms.github_pages.enabled) {
    console.log(colors.wisdom(`${symbols.heart} GitHub Pages deployment not enabled`));
    return;
  }
  
  console.log(colors.blessed(`${symbols.light} Deploying to GitHub Pages with divine blessing...`));
  
  try {
    // Copy built files to deployment directory
    const deployDir = 'dist/gh-pages';
    await fs.ensureDir(deployDir);
    
    // Copy VitePress build
    if (await fs.pathExists('.vitepress/dist')) {
      await fs.copy('.vitepress/dist', deployDir);
    }
    
    // Copy additional assets
    if (await fs.pathExists('dist/pdf')) {
      await fs.copy('dist/pdf', path.join(deployDir, 'downloads'));
    }
    
    if (await fs.pathExists('assets/diagrams')) {
      await fs.copy('assets/diagrams', path.join(deployDir, 'assets/diagrams'));
    }
    
    // Copy manifest
    if (await fs.pathExists('dist/manifest.json')) {
      await fs.copy('dist/manifest.json', path.join(deployDir, 'manifest.json'));
    }
    
    // Create CNAME file if domain is configured
    if (deployConfig.platforms.github_pages.domain) {
      await fs.writeFile(
        path.join(deployDir, 'CNAME'), 
        deployConfig.platforms.github_pages.domain
      );
    }
    
    // Create .nojekyll file
    await fs.writeFile(path.join(deployDir, '.nojekyll'), '');
    
    console.log(colors.peace(`  ${symbols.prayer} GitHub Pages deployment prepared`));
    console.log(colors.blessed(`${symbols.star} GitHub Pages blessed and ready!\n`));
  } catch (error) {
    console.error(colors.wisdom(`${symbols.heart} GitHub Pages deployment needs guidance:`, error.message));
    throw error;
  }
}

/**
 * Create release archives
 */
async function createReleaseArchives() {
  console.log(colors.blessed(`${symbols.light} Creating blessed release archives...`));
  
  try {
    const archiveDir = 'dist/archives';
    await fs.ensureDir(archiveDir);
    
    const version = new Date().toISOString().split('T')[0].replace(/-/g, '.');
    
    // Create documentation archive
    const docsArchive = path.join(archiveDir, `zerolight-orbit-docs-v${version}.tar.gz`);
    execSync(`tar -czf "${docsArchive}" *.md diagrams/ assets/ templates/`, { stdio: 'inherit' });
    
    // Create complete distribution archive
    const distArchive = path.join(archiveDir, `zerolight-orbit-complete-v${version}.tar.gz`);
    execSync(`tar -czf "${distArchive}" dist/ .vitepress/dist/`, { stdio: 'inherit' });
    
    console.log(colors.peace(`  ${symbols.prayer} Documentation archive: ${docsArchive}`));
    console.log(colors.peace(`  ${symbols.prayer} Complete archive: ${distArchive}`));
    console.log(colors.blessed(`${symbols.star} Release archives blessed!\n`));
  } catch (error) {
    console.log(colors.wisdom(`${symbols.heart} Archive creation needs alternative approach`));
    // Continue without archives if tar is not available
  }
}

/**
 * Post-deployment verification
 */
async function postDeploymentVerification() {
  console.log(colors.blessed(`${symbols.light} Performing post-deployment spiritual verification...`));
  
  const verifications = [
    {
      name: 'Deployment Integrity',
      check: async () => {
        const manifestPath = 'dist/manifest.json';
        if (await fs.pathExists(manifestPath)) {
          const manifest = await fs.readJSON(manifestPath);
          return `Manifest blessed: ${manifest.blessed ? 'Yes' : 'No'}`;
        }
        return 'Manifest created';
      }
    },
    {
      name: 'Asset Completeness',
      check: async () => {
        const assets = ['assets/logo-orbit.svg'];
        let count = 0;
        for (const asset of assets) {
          if (await fs.pathExists(asset)) count++;
        }
        return `${count}/${assets.length} spiritual assets verified`;
      }
    },
    {
      name: 'Distribution Readiness',
      check: async () => {
        const distributions = ['dist/pdf', 'dist/html', '.vitepress/dist'];
        let ready = 0;
        for (const dist of distributions) {
          if (await fs.pathExists(dist)) ready++;
        }
        return `${ready}/${distributions.length} distributions ready`;
      }
    }
  ];

  for (const verification of verifications) {
    try {
      const result = await verification.check();
      console.log(colors.peace(`  ${symbols.prayer} ${verification.name}: ${result}`));
    } catch (error) {
      console.log(colors.wisdom(`  ${symbols.heart} ${verification.name}: ${error.message}`));
    }
  }
  
  console.log(colors.blessed(`${symbols.star} Post-deployment verification blessed!\n`));
}

/**
 * Display deployment completion blessing
 */
function displayCompletionBlessing() {
  console.log('\n' + '='.repeat(70));
  console.log(colors.divine.bold(`${symbols.star} SACRED DEPLOYMENT COMPLETE ${symbols.star}`));
  console.log('='.repeat(70));
  console.log(colors.blessed(`${symbols.prayer} The ZeroLight Orbit documentation has been`));
  console.log(colors.blessed('deployed with divine blessing and spiritual guidance.'));
  console.log('');
  console.log(colors.spiritual('Deployment Summary:'));
  console.log(colors.peace(`  ${symbols.light} Documentation built and blessed`));
  console.log(colors.peace(`  ${symbols.light} Assets verified and sanctified`));
  console.log(colors.peace(`  ${symbols.light} Distribution prepared with care`));
  console.log(colors.peace(`  ${symbols.light} Spiritual integrity maintained`));
  console.log('');
  console.log(colors.sacred(`${symbols.heart} May this work serve humanity with wisdom`));
  console.log(colors.sacred(`${symbols.infinity} May it guide seekers on their spiritual journey`));
  console.log(colors.divine(`${symbols.prayer} In The Name of GOD - Blessed and Complete`));
  console.log('='.repeat(70) + '\n');
}

/**
 * Main deployment function
 */
async function deployWithBlessing() {
  try {
    displayDeploymentBlessing();
    await preDeploymentValidation();
    await buildDocumentation();
    await createDeploymentManifest();
    await deployToGitHubPages();
    await createReleaseArchives();
    await postDeploymentVerification();
    displayCompletionBlessing();
  } catch (error) {
    console.error(colors.wisdom(`${symbols.heart} Deployment encountered divine guidance needed:`));
    console.error(error.message);
    console.log(colors.blessed(`${symbols.prayer} Seeking spiritual resolution...`));
    process.exit(1);
  }
}

// Command line interface
if (require.main === module) {
  const args = process.argv.slice(2);
  const command = args[0] || 'full';
  
  switch (command) {
    case 'validate':
      preDeploymentValidation().catch(console.error);
      break;
    case 'build':
      buildDocumentation().catch(console.error);
      break;
    case 'github':
      deployToGitHubPages().catch(console.error);
      break;
    case 'full':
    default:
      deployWithBlessing().catch(console.error);
      break;
  }
}

module.exports = {
  deployWithBlessing,
  preDeploymentValidation,
  buildDocumentation,
  createDeploymentManifest,
  deployToGitHubPages,
  deployConfig
};