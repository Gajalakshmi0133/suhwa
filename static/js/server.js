const express = require("express");
const app = express();
const path = require("path");

app.use(express.static(__dirname + "/public"));
app.use(express.json());

app.post("/fcfs", (req, res) => {
    const processes = req.body.processes;

    let time = 0;
    let results = [];

    processes.forEach((p) => {
        let start = time;
        let completion = start + p.burst;
        let tat = completion - p.arrival;
        let wt = tat - p.burst;

        results.push({
            process: p.process,
            arrival: p.arrival,
            burst: p.burst,
            start,
            completion,
            tat,
            wt
        });

        time = completion;
    });

    res.json(results);
});

app.listen(3000, () => console.log("Server running on http://localhost:3000"));
