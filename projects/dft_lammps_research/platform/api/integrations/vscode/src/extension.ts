/**
 * DFT+LAMMPS VS Code Extension
 * 
 * Main extension file - handles activation, commands, and tree views.
 */

import * as vscode from 'vscode';
import * as path from 'path';
import { DFTLAMMPSClient } from './client';
import { ProjectsProvider, ProjectTreeItem } from './projectsProvider';
import { CalculationsProvider, CalculationTreeItem } from './calculationsProvider';
import { StructureViewer } from './structureViewer';

let client: DFTLAMMPSClient;
let projectsProvider: ProjectsProvider;
let calculationsProvider: CalculationsProvider;
let statusBarItem: vscode.StatusBarItem;

export function activate(context: vscode.ExtensionContext) {
    console.log('DFT+LAMMPS extension activated');

    // Initialize client
    client = new DFTLAMMPSClient(context);

    // Initialize tree providers
    projectsProvider = new ProjectsProvider(client);
    calculationsProvider = new CalculationsProvider(client);

    // Register tree data providers
    vscode.window.registerTreeDataProvider('dft-lammps-explorer', projectsProvider);

    // Status bar
    statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBarItem.command = 'dft-lammps.openSettings';
    context.subscriptions.push(statusBarItem);

    // Update authentication state
    updateAuthenticationState();

    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('dft-lammps.login', handleLogin),
        vscode.commands.registerCommand('dft-lammps.logout', handleLogout),
        vscode.commands.registerCommand('dft-lammps.refreshProjects', () => projectsProvider.refresh()),
        vscode.commands.registerCommand('dft-lammps.createProject', handleCreateProject),
        vscode.commands.registerCommand('dft-lammps.viewProject', handleViewProject),
        vscode.commands.registerCommand('dft-lammps.submitCalculation', handleSubmitCalculation),
        vscode.commands.registerCommand('dft-lammps.viewCalculation', handleViewCalculation),
        vscode.commands.registerCommand('dft-lammps.uploadStructure', handleUploadStructure),
        vscode.commands.registerCommand('dft-lammps.openSettings', handleOpenSettings),
    );

    // File watchers for auto-upload
    setupFileWatcher(context);
}

export function deactivate() {
    console.log('DFT+LAMMPS extension deactivated');
}

function updateAuthenticationState() {
    const isAuthenticated = client.isAuthenticated();
    vscode.commands.executeCommand('setContext', 'dft-lammps:authenticated', isAuthenticated);
    
    if (isAuthenticated) {
        statusBarItem.text = '$(account) DFT+LAMMPS';
        statusBarItem.tooltip = 'Connected to DFT+LAMMPS';
        statusBarItem.show();
    } else {
        statusBarItem.text = '$(sign-in) DFT+LAMMPS';
        statusBarItem.tooltip = 'Click to sign in';
        statusBarItem.show();
    }
}

async function handleLogin() {
    const apiKey = await vscode.window.showInputBox({
        prompt: 'Enter your DFT+LAMMPS API Key',
        password: true,
        ignoreFocusOut: true,
        placeHolder: 'dftl_...'
    });

    if (apiKey) {
        try {
            await client.authenticate(apiKey);
            vscode.window.showInformationMessage('Successfully signed in to DFT+LAMMPS');
            updateAuthenticationState();
            projectsProvider.refresh();
        } catch (error) {
            vscode.window.showErrorMessage(`Authentication failed: ${error}`);
        }
    }
}

async function handleLogout() {
    await client.logout();
    updateAuthenticationState();
    vscode.window.showInformationMessage('Signed out from DFT+LAMMPS');
}

async function handleCreateProject() {
    const name = await vscode.window.showInputBox({
        prompt: 'Project name',
        placeHolder: 'My Research Project'
    });

    if (!name) return;

    const description = await vscode.window.showInputBox({
        prompt: 'Project description (optional)',
        placeHolder: 'Study of Li-S battery materials'
    });

    const projectType = await vscode.window.showQuickPick([
        { label: 'Battery Screening', value: 'battery_screening' },
        { label: 'Catalysis', value: 'catalysis' },
        { label: 'Phonon Calculations', value: 'phonon' },
        { label: 'Custom', value: 'custom' }
    ], { placeHolder: 'Select project type' });

    if (!projectType) return;

    try {
        const project = await client.createProject({
            name,
            description,
            projectType: projectType.value
        });

        vscode.window.showInformationMessage(`Project created: ${project.name}`);
        projectsProvider.refresh();
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to create project: ${error}`);
    }
}

async function handleViewProject(item: ProjectTreeItem) {
    const panel = vscode.window.createWebviewPanel(
        'dft-lammps-project',
        `Project: ${item.project.name}`,
        vscode.ViewColumn.One,
        { enableScripts: true }
    );

    panel.webview.html = getProjectHtml(item.project);
}

async function handleSubmitCalculation(item: ProjectTreeItem) {
    // Get active editor file
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('Open a structure file to submit calculation');
        return;
    }

    const document = editor.document;
    const content = document.getText();
    const filename = path.basename(document.fileName);

    // Select calculation type
    const calcType = await vscode.window.showQuickPick([
        { label: 'DFT (Density Functional Theory)', value: 'dft' },
        { label: 'LAMMPS (Molecular Dynamics)', value: 'lammps' },
        { label: 'ML Potential', value: 'ml' }
    ], { placeHolder: 'Select calculation type' });

    if (!calcType) return;

    // Get parameters
    const paramsInput = await vscode.window.showInputBox({
        prompt: 'Calculation parameters (JSON)',
        placeHolder: '{"ecut": 500, "kpoints": "4 4 4"}'
    });

    let params = {};
    if (paramsInput) {
        try {
            params = JSON.parse(paramsInput);
        } catch {
            vscode.window.showErrorMessage('Invalid JSON parameters');
            return;
        }
    }

    try {
        // Upload structure first
        const structure = await client.uploadStructure(item.project.id, {
            name: filename,
            format: path.extname(filename).slice(1) || 'poscar',
            data: content
        });

        // Submit calculation
        const calculation = await client.submitCalculation(item.project.id, {
            structureId: structure.id,
            calculationType: calcType.value,
            parameters: params
        });

        vscode.window.showInformationMessage(
            `Calculation submitted: ${calculation.id}`,
            'View Status'
        ).then(selection => {
            if (selection === 'View Status') {
                vscode.commands.executeCommand('dft-lammps.viewCalculation', calculation);
            }
        });

        // Start monitoring
        monitorCalculation(calculation.id);

    } catch (error) {
        vscode.window.showErrorMessage(`Failed to submit calculation: ${error}`);
    }
}

async function handleViewCalculation(calculation: any) {
    const panel = vscode.window.createWebviewPanel(
        'dft-lammps-calculation',
        `Calculation: ${calculation.id}`,
        vscode.ViewColumn.One,
        { enableScripts: true }
    );

    // Poll for updates
    const interval = setInterval(async () => {
        try {
            const updated = await client.getCalculation(calculation.id);
            panel.webview.html = getCalculationHtml(updated);

            if (updated.status === 'completed' || updated.status === 'failed') {
                clearInterval(interval);
                
                if (vscode.workspace.getConfiguration('dft-lammps').get('notifications')) {
                    vscode.window.showInformationMessage(
                        `Calculation ${calculation.id} ${updated.status}`
                    );
                }
            }
        } catch (error) {
            clearInterval(interval);
        }
    }, vscode.workspace.getConfiguration('dft-lammps').get('pollInterval', 5) * 1000);

    panel.onDidDispose(() => clearInterval(interval));
    panel.webview.html = getCalculationHtml(calculation);
}

async function handleUploadStructure(item: ProjectTreeItem) {
    const uris = await vscode.window.showOpenDialog({
        canSelectFiles: true,
        canSelectFolders: false,
        canSelectMany: true,
        filters: {
            'Structure Files': ['cif', 'poscar', 'vasp', 'xyz', 'json'],
            'All Files': ['*']
        }
    });

    if (!uris) return;

    for (const uri of uris) {
        try {
            const content = await vscode.workspace.fs.readFile(uri);
            const filename = path.basename(uri.fsPath);
            
            await client.uploadStructure(item.project.id, {
                name: filename,
                format: path.extname(filename).slice(1) || 'poscar',
                data: Buffer.from(content).toString()
            });

            vscode.window.showInformationMessage(`Uploaded: ${filename}`);
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to upload ${uri.fsPath}: ${error}`);
        }
    }
}

async function handleOpenSettings() {
    vscode.commands.executeCommand('workbench.action.openSettings', 'dft-lammps');
}

function setupFileWatcher(context: vscode.ExtensionContext) {
    const pattern = '**/*.{cif,poscar,vasp,xyz}';
    const watcher = vscode.workspace.createFileSystemWatcher(pattern);

    watcher.onDidCreate(uri => {
        // Optional: Auto-upload on file creation
        // This could be controlled by a setting
    });

    context.subscriptions.push(watcher);
}

function monitorCalculation(calculationId: string) {
    // Background monitoring logic
}

// HTML generators for webviews
function getProjectHtml(project: any): string {
    return `
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>${project.name}</title>
            <style>
                body { font-family: var(--vscode-font-family); padding: 20px; }
                .header { border-bottom: 1px solid var(--vscode-panel-border); padding-bottom: 10px; margin-bottom: 20px; }
                .status { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; }
                .status.active { background: #2ea043; color: white; }
                .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0; }
                .stat-box { background: var(--vscode-editor-background); padding: 15px; border-radius: 8px; border: 1px solid var(--vscode-panel-border); }
                .stat-value { font-size: 24px; font-weight: bold; }
                .stat-label { font-size: 12px; color: var(--vscode-descriptionForeground); }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>${project.name}</h1>
                <span class="status ${project.status}">${project.status}</span>
                <p style="color: var(--vscode-descriptionForeground);">${project.description || 'No description'}</p>
            </div>
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-value">${project.total_structures || 0}</div>
                    <div class="stat-label">Structures</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">${project.completed_calculations || 0}</div>
                    <div class="stat-label">Completed</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">${project.failed_calculations || 0}</div>
                    <div class="stat-label">Failed</div>
                </div>
            </div>
            <p><strong>Project ID:</strong> <code>${project.id}</code></p>
            <p><strong>Created:</strong> ${new Date(project.created_at).toLocaleString()}</p>
            <p><strong>Type:</strong> ${project.project_type}</p>
        </body>
        </html>
    `;
}

function getCalculationHtml(calc: any): string {
    const statusColor = calc.status === 'completed' ? '#2ea043' : 
                        calc.status === 'failed' ? '#f85149' : '#d29922';
    
    return `
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Calculation ${calc.id}</title>
            <style>
                body { font-family: var(--vscode-font-family); padding: 20px; }
                .status { color: ${statusColor}; font-weight: bold; }
                pre { background: var(--vscode-editor-background); padding: 15px; border-radius: 8px; overflow-x: auto; }
            </style>
        </head>
        <body>
            <h1>Calculation</h1>
            <p><strong>Status:</strong> <span class="status">${calc.status}</span></p>
            <p><strong>Type:</strong> ${calc.calculation_type}</p>
            <p><strong>ID:</strong> <code>${calc.id}</code></p>
            ${calc.results ? `<h2>Results</h2><pre>${JSON.stringify(calc.results, null, 2)}</pre>` : ''}
            ${calc.error_message ? `<h2>Error</h2><pre style="color: #f85149;">${calc.error_message}</pre>` : ''}
        </body>
        </html>
    `;
}
