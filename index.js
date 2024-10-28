const express = require('express');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const app = express();

const port = 5100;

// Middleware to parse JSON bodies
app.use(express.json());

// Listening on the port
app.listen(port, () => {
    console.log(`API is now running on port ${port}`);
});

// Health check endpoint
app.get('/', (req, res) => res.json("API is running"));

// Creating the plan and save it into text
app.post('/plan', (req, res) => {
    // Get the input data from the request body
    const inputData = req.body;

    // Convert inputData to JSON string to pass to Python script
    const inputDataString = JSON.stringify(inputData);

    // Path to main.py (assuming it's in the same directory)
    const scriptPath = path.join(__dirname, 'main.py');

    const pythonPath = path.join(__dirname, 'venv', 'bin', 'python');

    // Spawn a child process to run the Python script
    const pythonProcess = spawn(pythonPath, [scriptPath, '--input_data', inputDataString]);

    let pythonOutput = '';
    let pythonError = '';

    // Collect data from script
    pythonProcess.stdout.on('data', (data) => {
        pythonOutput += data.toString();
    });

    // Collect error output
    pythonProcess.stderr.on('data', (data) => {
        pythonError += data.toString();
    });

    // Handle script completion
    pythonProcess.on('close', (code) => {
        if (code === 0) {
            console.log('Python script executed successfully.');
            res.json({ success: true, message: 'Care plan generated successfully.' });
        } else {
            console.error(`Python script exited with code ${code}`);
            console.error(`Error: ${pythonError}`);
            res.status(500).json({ success: false, error: pythonError });
        }
    });
});

// Get the txt plan
app.post('/getPlan', (req, res) => {
    // File location '/home/ubuntu/UserPlan'
    const uid = req.body.uid;

    if (!uid) {
        return res.status(400).json({ success: false, error: 'UID is required' });
    }

    const filePath = path.join('/home/ubuntu/UserPlan', `${uid}.txt`);

    // Check if the file exists
    fs.access(filePath, fs.constants.F_OK, (err) => {
        if (err) {
            // File does not exist
            return res.status(404).json({ success: false, error: 'Plan not found' });
        } else {
            // Read and return the file
            fs.readFile(filePath, 'utf8', (err, data) => {
                if (err) {
                    // Error reading the file
                    return res.status(500).json({ success: false, error: 'Error reading the plan' });
                } else {
                    // Return the file content
                    res.json({ success: true, plan: data });
                }
            });
        }
    });
});
