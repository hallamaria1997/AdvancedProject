const express = require('express')
const router = express.Router();
const cors = require('cors');
const bodyParser = require('body-parser');
const {spawn} = require('child_process');
const fs = require('fs')
const multer = require('multer');

const app = express();
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(cors());

const port = 3000

app.use(express.json());

const url = '/api/v1'

// how to save the received image
const storage = multer.diskStorage({
    destination: 'images\\',
	filename: function (req, file, cb) {
		cb(null, file.originalname)
	  }
});

function checkFileType(file, cb) {
    // Allowed ext
    const filetypes = /jpeg|jpg|png|gif/;
    // Check ext
    const extname = filetypes.test(path.extname(file.originalname).toLowerCase());
    // Check mime
    const mimetype = filetypes.test(file.mimetype);

    if (mimetype && extname) {
        return cb(null, true);
    } else {
        cb('Error: Images Only!');
    }
}

// file upload handler
const upload = multer({
    storage: storage,
    limits: { fileSize: 1000000 }
});

app.get(url, (req, res) => res.json({ message: 'message' }))

// the endpoint for uploading image and getting the beauty score
app.post(url + '/image', upload.single('file'), (req, res) => {
	console.log(req.file)
	console.log(req.file.filename)

	imgs_path = 'images\\'
	console.log(imgs_path + req.file.filename)
	const python = spawn('python', ['neural_net.py', imgs_path + req.file.filename]);
	python.stdout.on('data', function (data) {
		console.log('Pipe data from python script ...');
		retObj = {score: data.toString()};
	});
	python.on('exit', (code) => {
		console.log(`child process close all stdio with code ${code}`);
		// send data to browser
		res.status(200).json(retObj)
	});
})

app.listen(port, () => console.log(`Advanced Project app listening on port ${port}!`))