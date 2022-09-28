const express = require('express')
const cors = require('cors');
const bodyParser = require('body-parser');

const app = express();
app.use(bodyParser.json());
app.use(cors());

const port = 3000

app.use(express.json());

const url = '/api/v1'

app.get(url, (req, res) => res.json({ message: 'message' }))

app.listen(port, () => console.log(`Advanced Project app listening on port ${port}!`))