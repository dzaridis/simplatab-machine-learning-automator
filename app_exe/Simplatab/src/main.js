const { app, BrowserWindow, dialog, ipcMain, shell } = require('electron');
const fs = require('fs');
const { exec, execSync } = require('child_process');
const path = require('path');

let mainWindow;

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1200,  // Set the width to 1200 pixels
        height: 800, // Set the height to 800 pixels
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false
        }
    });

    mainWindow.loadFile(path.join(__dirname, 'index.html'));
}

app.on('ready', createWindow);

ipcMain.handle('select-directory', async (event, args) => {
    const result = await dialog.showOpenDialog(mainWindow, {
        properties: ['openDirectory']
    });
    return result.filePaths[0];
});

ipcMain.handle('check-docker-installed', (event, args) => {
    try {
        execSync('docker --version');
        return true;
    } catch (err) {
        return false;
    }
});

ipcMain.handle('open-docker-link', (event, args) => {
    shell.openExternal('https://www.docker.com/products/docker-desktop');
});

ipcMain.handle('start-docker-compose', (event, inputDir, outputDir) => {
    const dockerComposePath = path.join(__dirname, '..', 'docker-compose.yml');
    console.log(`Docker Compose file path: ${dockerComposePath}`);
    
    if (!fs.existsSync(dockerComposePath)) {
        console.error(`Docker Compose file not found at path: ${dockerComposePath}`);
        return;
    }

    let dockerComposeFile = fs.readFileSync(dockerComposePath, 'utf8');
    dockerComposeFile = dockerComposeFile.replace('${INPUT_FOLDER}', inputDir);
    dockerComposeFile = dockerComposeFile.replace('${OUTPUT_FOLDER}', outputDir);

    fs.writeFileSync(dockerComposePath, dockerComposeFile, 'utf8');
    console.log('Updated docker-compose.yml with paths:');
    console.log(`Input Dir: ${inputDir}`);
    console.log(`Output Dir: ${outputDir}`);

    exec('docker-compose up -d', { cwd: path.dirname(dockerComposePath) }, (err, stdout, stderr) => {
        if (err) {
            console.error(`Error starting Docker Compose: ${stderr}`);
            return;
        }
        console.log(`Docker Compose started: ${stdout}`);
        checkServiceStatus();
    });
});

ipcMain.handle('switch-to-vision-impaired-mode', (event, args) => {
    mainWindow.loadFile(path.join(__dirname, 'vision-impaired.html'));
});

ipcMain.handle('switch-to-normal-mode', (event, args) => {
    mainWindow.loadFile(path.join(__dirname, 'index.html'));
});

function checkServiceStatus() {
    let serviceUp = false;
    let interval = setInterval(() => {
        try {
            let logs = execSync('docker-compose logs api', { cwd: path.join(__dirname, '..') }).toString();
            console.log('Checking Docker Compose logs:');
            console.log(logs);
            if (logs.includes('Running on http://127.0.0.1:5000')) {
                serviceUp = true;
            }
        } catch (error) {
            console.error(`Error checking logs: ${error}`);
        }

        if (serviceUp) {
            clearInterval(interval);
            console.log('Service is up. Opening browser...');
            shell.openExternal('http://localhost:5000');
        }
    }, 3000); // Check every 3 seconds
}

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});
