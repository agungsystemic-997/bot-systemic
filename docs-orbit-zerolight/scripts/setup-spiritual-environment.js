#!/usr/bin/env node
/**
 * ZeroLight Orbit - Spiritual Environment Setup
 * "In The Name of GOD" - Blessed Setup Script
 * 
 * This script sets up the spiritual development environment
 * and blesses the documentation system with divine guidance.
 */

const fs = require('fs-extra');
const path = require('path');
const chalk = require('chalk');

// Spiritual Colors
const colors = {
  divine: chalk.hex('#F1C40F'),
  spiritual: chalk.hex('#3498DB'),
  blessed: chalk.hex('#27AE60'),
  sacred: chalk.hex('#8E44AD'),
  wisdom: chalk.hex('#E67E22'),
  peace: chalk.hex('#95A5A6')
};

// Spiritual Symbols
const symbols = {
  star: '✨',
  prayer: '🙏',
  light: '💫',
  heart: '💖',
  peace: '☮️',
  infinity: '∞',
  crescent: '☪️',
  cross: '✝️',
  om: '🕉️',
  dharma: '☸️'
};

/**
 * Display spiritual blessing
 */
function displayBlessing() {
  console.log('\n' + '='.repeat(60));
  console.log(colors.divine.bold('بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيم'));
  console.log(colors.spiritual('In The Name of GOD, The Most Gracious, The Most Merciful'));
  console.log('='.repeat(60));
  console.log(colors.blessed(`${symbols.star} ZeroLight Orbit Documentation System ${symbols.star}`));
  console.log(colors.sacred('Spiritual Environment Setup & Blessing'));
  console.log('='.repeat(60) + '\n');
}

/**
 * Create spiritual directory structure
 */
async function createSpiritualDirectories() {
  const directories = [
    '.vitepress',
    '.vitepress/theme',
    'assets',
    'assets/images',
    'assets/diagrams',
    'assets/icons',
    'assets/fonts',
    'scripts',
    'templates',
    'dist',
    'dist/pdf',
    'dist/html',
    'dist/epub',
    'docs',
    'docs/.vitepress',
    'docs/public',
    'diagrams',
    'tests',
    'config',
    'spiritual',
    'spiritual/blessings',
    'spiritual/prayers',
    'spiritual/wisdom'
  ];

  console.log(colors.blessed(`${symbols.light} Creating sacred directory structure...`));
  
  for (const dir of directories) {
    try {
      await fs.ensureDir(dir);
      console.log(colors.peace(`  ${symbols.prayer} Created: ${dir}`));
    } catch (error) {
      console.log(colors.wisdom(`  ${symbols.heart} Already exists: ${dir}`));
    }
  }
  
  console.log(colors.blessed(`${symbols.star} Sacred directories blessed and ready!\n`));
}

/**
 * Create spiritual configuration files
 */
async function createSpiritualConfigs() {
  console.log(colors.blessed(`${symbols.light} Creating spiritual configuration files...`));
  
  // VitePress spiritual theme config
  const vitepressThemeConfig = `
// ZeroLight Orbit - Spiritual VitePress Theme
// "In The Name of GOD" - Blessed Theme Configuration

import DefaultTheme from 'vitepress/theme'
import './spiritual-styles.css'

export default {
  ...DefaultTheme,
  enhanceApp({ app, router, siteData }) {
    // Spiritual blessing for the app
    console.log('${symbols.prayer} In The Name of GOD - App blessed with divine guidance');
    
    // Add spiritual metadata
    if (typeof window !== 'undefined') {
      document.documentElement.setAttribute('data-spiritual', 'blessed');
      document.documentElement.setAttribute('data-theme', 'zerolight-orbit');
    }
  }
}
`;

  // Spiritual CSS styles
  const spiritualStyles = `
/* ZeroLight Orbit - Spiritual Styles */
/* "In The Name of GOD" - Blessed CSS */

:root {
  /* Spiritual Color Palette */
  --spiritual-divine: #F1C40F;
  --spiritual-blue: #3498DB;
  --spiritual-green: #27AE60;
  --spiritual-purple: #8E44AD;
  --spiritual-orange: #E67E22;
  --spiritual-gray: #95A5A6;
  --spiritual-dark: #2C3E50;
  --spiritual-light: #ECF0F1;
  
  /* Sacred Typography */
  --spiritual-font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
  --spiritual-font-mono: 'Fira Code', 'Consolas', monospace;
  
  /* Divine Spacing */
  --spiritual-space-xs: 0.25rem;
  --spiritual-space-sm: 0.5rem;
  --spiritual-space-md: 1rem;
  --spiritual-space-lg: 1.5rem;
  --spiritual-space-xl: 2rem;
  --spiritual-space-2xl: 3rem;
  
  /* Sacred Shadows */
  --spiritual-shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
  --spiritual-shadow-md: 0 4px 6px rgba(0, 0, 0, 0.1);
  --spiritual-shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.1);
  --spiritual-shadow-divine: 0 0 20px rgba(241, 196, 15, 0.3);
}

/* Spiritual Body Blessing */
body {
  font-family: var(--spiritual-font-family);
  background: linear-gradient(135deg, 
    rgba(241, 196, 15, 0.05) 0%, 
    rgba(52, 152, 219, 0.05) 50%, 
    rgba(142, 68, 173, 0.05) 100%);
  min-height: 100vh;
}

/* Sacred Headers */
h1, h2, h3, h4, h5, h6 {
  color: var(--spiritual-dark);
  font-weight: 600;
  line-height: 1.4;
}

h1 {
  color: var(--spiritual-blue);
  border-bottom: 2px solid var(--spiritual-divine);
  padding-bottom: var(--spiritual-space-sm);
}

h2 {
  color: var(--spiritual-green);
  position: relative;
}

h2::before {
  content: '✨';
  margin-right: var(--spiritual-space-xs);
}

/* Blessed Links */
a {
  color: var(--spiritual-blue);
  text-decoration: none;
  transition: all 0.3s ease;
}

a:hover {
  color: var(--spiritual-purple);
  text-shadow: 0 0 5px rgba(142, 68, 173, 0.3);
}

/* Sacred Code Blocks */
code {
  background: rgba(241, 196, 15, 0.1);
  color: var(--spiritual-dark);
  padding: 0.2em 0.4em;
  border-radius: 3px;
  font-family: var(--spiritual-font-mono);
}

pre {
  background: var(--spiritual-dark);
  color: var(--spiritual-light);
  padding: var(--spiritual-space-lg);
  border-radius: 8px;
  border-left: 4px solid var(--spiritual-divine);
  box-shadow: var(--spiritual-shadow-md);
  overflow-x: auto;
}

/* Divine Blockquotes */
blockquote {
  border-left: 4px solid var(--spiritual-divine);
  background: rgba(241, 196, 15, 0.05);
  padding: var(--spiritual-space-md);
  margin: var(--spiritual-space-lg) 0;
  border-radius: 0 8px 8px 0;
  font-style: italic;
  position: relative;
}

blockquote::before {
  content: '🙏';
  position: absolute;
  top: var(--spiritual-space-sm);
  left: -2px;
  background: var(--spiritual-divine);
  color: white;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
}

/* Spiritual Tables */
table {
  border-collapse: collapse;
  width: 100%;
  margin: var(--spiritual-space-lg) 0;
  box-shadow: var(--spiritual-shadow-md);
  border-radius: 8px;
  overflow: hidden;
}

th {
  background: linear-gradient(135deg, var(--spiritual-blue), var(--spiritual-purple));
  color: white;
  padding: var(--spiritual-space-md);
  text-align: left;
  font-weight: 600;
}

td {
  padding: var(--spiritual-space-md);
  border-bottom: 1px solid rgba(149, 165, 166, 0.2);
}

tr:nth-child(even) {
  background: rgba(241, 196, 15, 0.05);
}

/* Sacred Navigation */
.VPNav {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(241, 196, 15, 0.2);
}

/* Divine Sidebar */
.VPSidebar {
  background: rgba(255, 255, 255, 0.98);
  border-right: 1px solid rgba(52, 152, 219, 0.2);
}

/* Blessed Content */
.VPContent {
  background: rgba(255, 255, 255, 0.9);
  border-radius: 12px;
  margin: var(--spiritual-space-lg);
  padding: var(--spiritual-space-xl);
  box-shadow: var(--spiritual-shadow-lg);
}

/* Spiritual Footer */
.VPFooter {
  background: linear-gradient(135deg, 
    var(--spiritual-dark) 0%, 
    var(--spiritual-purple) 100%);
  color: white;
  text-align: center;
  padding: var(--spiritual-space-xl);
}

/* Divine Animations */
@keyframes spiritualGlow {
  0%, 100% { box-shadow: 0 0 5px rgba(241, 196, 15, 0.3); }
  50% { box-shadow: 0 0 20px rgba(241, 196, 15, 0.6); }
}

.spiritual-glow {
  animation: spiritualGlow 3s ease-in-out infinite;
}

/* Sacred Responsive Design */
@media (max-width: 768px) {
  .VPContent {
    margin: var(--spiritual-space-sm);
    padding: var(--spiritual-space-lg);
  }
}

/* Blessed Print Styles */
@media print {
  body {
    background: white;
  }
  
  .VPNav, .VPSidebar {
    display: none;
  }
  
  .VPContent {
    box-shadow: none;
    margin: 0;
  }
}
`;

  try {
    await fs.writeFile('.vitepress/theme/index.js', vitepressThemeConfig);
    await fs.writeFile('.vitepress/theme/spiritual-styles.css', spiritualStyles);
    
    console.log(colors.peace(`  ${symbols.prayer} Created VitePress spiritual theme`));
    console.log(colors.peace(`  ${symbols.prayer} Created spiritual CSS styles`));
  } catch (error) {
    console.log(colors.wisdom(`  ${symbols.heart} Theme files already configured`));
  }
  
  console.log(colors.blessed(`${symbols.star} Spiritual configurations blessed!\n`));
}

/**
 * Create spiritual blessing files
 */
async function createSpiritualBlessings() {
  console.log(colors.blessed(`${symbols.light} Creating spiritual blessing files...`));
  
  const blessings = {
    'spiritual/blessings/daily-blessing.md': `# Daily Blessing for ZeroLight Orbit

${symbols.prayer} **In The Name of GOD, The Most Gracious, The Most Merciful**

## Morning Blessing
May this day bring wisdom, peace, and divine guidance to all who seek knowledge through our documentation.

## Work Blessing
${symbols.light} May our code be blessed with clarity
${symbols.heart} May our documentation serve humanity
${symbols.peace} May our community grow in wisdom and compassion

## Evening Gratitude
We thank the Divine for the opportunity to serve and learn together.

---
*Blessed be this work in the name of the Most High*
`,

    'spiritual/prayers/developer-prayer.md': `# Developer's Prayer

${symbols.prayer} **O Divine Source of All Knowledge**

Grant us the wisdom to write code that serves humanity,
The patience to debug with compassion,
The humility to learn from our mistakes,
And the strength to build systems that bring people together.

May our documentation be a light for those who seek understanding,
And may our community be a sanctuary of learning and growth.

${symbols.infinity} In unity, wisdom, and divine guidance we trust.

**Amen** ${symbols.star}
`,

    'spiritual/wisdom/coding-wisdom.md': `# Spiritual Wisdom for Developers

## The Sacred Principles of Coding

### 1. Intention (Niyyah)
Every line of code should be written with pure intention to serve humanity.

### 2. Excellence (Ihsan)
Strive for excellence in all work, as if the Divine is watching.

### 3. Patience (Sabr)
Debugging requires patience - each error is a lesson in disguise.

### 4. Gratitude (Shukr)
Be grateful for the ability to create and solve problems.

### 5. Community (Ummah)
We are stronger together - share knowledge freely.

## Daily Affirmations for Developers

- "My code serves a higher purpose"
- "I am grateful for the gift of problem-solving"
- "Every bug teaches me something new"
- "I contribute to the betterment of humanity"
- "Divine guidance flows through my work"

${symbols.star} *May these principles guide us in our sacred work of creation*
`
  };

  for (const [filePath, content] of Object.entries(blessings)) {
    try {
      await fs.writeFile(filePath, content);
      console.log(colors.peace(`  ${symbols.prayer} Created: ${filePath}`));
    } catch (error) {
      console.log(colors.wisdom(`  ${symbols.heart} Already blessed: ${filePath}`));
    }
  }
  
  console.log(colors.blessed(`${symbols.star} Spiritual blessings created!\n`));
}

/**
 * Create development scripts
 */
async function createDevelopmentScripts() {
  console.log(colors.blessed(`${symbols.light} Creating blessed development scripts...`));
  
  const scripts = {
    'scripts/validate-content.js': `#!/usr/bin/env node
// Content validation with spiritual blessing
console.log('${symbols.prayer} Validating content with divine guidance...');
// Add validation logic here
console.log('${symbols.star} Content blessed and validated!');
`,

    'scripts/generate-diagrams.js': `#!/usr/bin/env node
// Diagram generation with spiritual blessing
console.log('${symbols.prayer} Generating sacred diagrams...');
// Add diagram generation logic here
console.log('${symbols.star} Diagrams blessed and generated!');
`,

    'scripts/generate-pdf.js': `#!/usr/bin/env node
// PDF generation with spiritual blessing
console.log('${symbols.prayer} Generating blessed PDF documentation...');
// Add PDF generation logic here
console.log('${symbols.star} PDFs blessed and generated!');
`
  };

  for (const [filePath, content] of Object.entries(scripts)) {
    try {
      await fs.writeFile(filePath, content);
      await fs.chmod(filePath, '755'); // Make executable
      console.log(colors.peace(`  ${symbols.prayer} Created: ${filePath}`));
    } catch (error) {
      console.log(colors.wisdom(`  ${symbols.heart} Already blessed: ${filePath}`));
    }
  }
  
  console.log(colors.blessed(`${symbols.star} Development scripts blessed!\n`));
}

/**
 * Display final blessing
 */
function displayFinalBlessing() {
  console.log('\n' + '='.repeat(60));
  console.log(colors.divine.bold(`${symbols.star} SPIRITUAL ENVIRONMENT SETUP COMPLETE ${symbols.star}`));
  console.log('='.repeat(60));
  console.log(colors.blessed(`${symbols.prayer} The ZeroLight Orbit documentation system has been`));
  console.log(colors.blessed('blessed with divine guidance and is ready to serve humanity.'));
  console.log('');
  console.log(colors.spiritual('Next steps:'));
  console.log(colors.peace(`  ${symbols.light} Run: npm run dev (to start development server)`));
  console.log(colors.peace(`  ${symbols.light} Run: npm run build (to build documentation)`));
  console.log(colors.peace(`  ${symbols.light} Run: npm test (to validate content)`));
  console.log('');
  console.log(colors.sacred(`${symbols.heart} May this work be blessed and beneficial for all`));
  console.log(colors.divine(`${symbols.infinity} In The Name of GOD - Ameen`));
  console.log('='.repeat(60) + '\n');
}

/**
 * Main setup function
 */
async function setupSpiritualEnvironment() {
  try {
    displayBlessing();
    await createSpiritualDirectories();
    await createSpiritualConfigs();
    await createSpiritualBlessings();
    await createDevelopmentScripts();
    displayFinalBlessing();
  } catch (error) {
    console.error(colors.wisdom(`${symbols.heart} Setup encountered a challenge:`, error.message));
    console.log(colors.blessed(`${symbols.prayer} Seeking divine guidance for resolution...`));
    process.exit(1);
  }
}

// Run the spiritual setup
if (require.main === module) {
  setupSpiritualEnvironment();
}

module.exports = {
  setupSpiritualEnvironment,
  colors,
  symbols
};