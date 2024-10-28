const express = require('express')

const app = express();

const port = 5100

//listening the port
app.listen(port, ()=> {
    console.log(`API is now running on port ${port}`);
})

//Here is the call back function
app.get('/', (req, res) => res.json("api is running"))

// Creating the plan and save it into text
app.post('/plan', (req, res) => {
    // Get the input

    // Run the main.py code and give it the input

    // Return success or error 

});

// Get the txt plan
app.post('/getPlan', (req, res) => {
    // file location '/home/ubuntu/UserPlan'
    // If the file exit, return it

    // If not, return error message

});