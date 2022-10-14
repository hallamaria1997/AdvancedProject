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

// a dummy endpoint to test running a python script
app.get(url + '/sum', (req, res) =>{
	const python = spawn('python', ['sum.py', req.body.num1, req.body.num2]);
	python.stdout.on('data', function (data) {
		console.log('Pipe data from python script ...');
		retObj = {sum: data.toString()};
	});
	// in close event we are sure that stream from child process is closed
	python.on('close', (code) => {
		console.log(`child process close all stdio with code ${code}`);
		// send data to browser
		res.status(200).json(retObj)
	});
})

// a dummy endpoint to test uploading an image
app.post(url + '/image', upload.single('file'), (req, res) => {
	console.log(req.file)
	console.log(req.body.new_filename)

	// have to change this line when we create the new network file
	const python = spawn('python', ['image.py', req.body.filename]);
	python.stdout.on('data', function (data) {
		console.log('Pipe data from python script ...');
		retObj = {score: data.toString()};
	});
	python.on('close', (code) => {
		console.log(`child process close all stdio with code ${code}`);
		// send data to browser
		res.status(200).json(retObj)
	});
	/*
	fs.rename(req.file.filename, req.file.originalname, (error) => {
		if (error) {
			return console.log(`Error: ${error}`);
		}
		else {
			console.log('renaming')
		}
	});
	*/
	// res.status(200).json(retObj);
})

app.listen(port, () => console.log(`Advanced Project app listening on port ${port}!`))