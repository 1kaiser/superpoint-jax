const puppeteer = require('puppeteer');
const http = require('http');
const fs = require('fs');
const path = require('path');

// Simple static server
const PORT = 3000;
const ROOT = path.resolve(__dirname, '../../'); // Repo root

const server = http.createServer((req, res) => {
    // Basic file serving logic
    let filePath = path.join(ROOT, req.url === '/' ? 'demo/viewer/index.html' : req.url);

    // Handle query params or fragments if any (though not expected here)
    if (filePath.includes('?')) filePath = filePath.split('?')[0];

    // Check if file exists
    if (!fs.existsSync(filePath)) {
        console.error(`404: ${filePath}`);
        res.writeHead(404);
        res.end('Not found');
        return;
    }

    const ext = path.extname(filePath);
    let contentType = 'text/plain';
    if (ext === '.html') contentType = 'text/html';
    if (ext === '.js') contentType = 'text/javascript';
    if (ext === '.glb') contentType = 'model/gltf-binary';
    if (ext === '.css') contentType = 'text/css';

    fs.readFile(filePath, (err, content) => {
        if (err) {
            res.writeHead(500);
            res.end(`Server Error: ${err.code}`);
        } else {
            res.writeHead(200, { 'Content-Type': contentType });
            res.end(content, 'utf-8');
        }
    });
});

server.listen(PORT, async () => {
    console.log(`Server running at http://localhost:${PORT}/`);

    try {
        const browser = await puppeteer.launch({
            args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-web-security'] // Often needed in CI/Docker
        });
        const page = await browser.newPage();

        // Handle console logs from browser
        page.on('console', msg => console.log('PAGE LOG:', msg.text()));

        console.log('Navigating to viewer...');
        await page.goto(`http://localhost:${PORT}/demo/viewer/index.html`, { waitUntil: 'networkidle0' });

        // Wait for model-viewer to load the model
        // We look for the class 'model-loaded' added by our script
        console.log('Waiting for model to load...');
        try {
            await page.waitForSelector('body.model-loaded', { timeout: 30000 }); // 30s timeout
            console.log('Model loaded successfully in browser!');
        } catch (e) {
            console.error('Timeout waiting for model load. Check if .glb exists and is valid.');
            // Take screenshot anyway
        }

        // Wait a bit for rendering to settle
        await new Promise(r => setTimeout(r, 2000));

        const screenshotPath = path.join(ROOT, 'output', 'model_viewer_screenshot.png');
        await page.screenshot({ path: screenshotPath, fullPage: true });
        console.log(`Screenshot saved to ${screenshotPath}`);

        await browser.close();
    } catch (error) {
        console.error('Puppeteer Error:', error);
    } finally {
        server.close();
    }
});
