import { NONE_TYPE } from '@angular/compiler';
import { Component, OnInit } from '@angular/core';
import { ImageUploadService } from '../image-upload.service';

@Component({
  selector: 'app-image-upload',
  templateUrl: './image-upload.component.html',
  styleUrls: ['./image-upload.component.css'],
})
export class ImageUploadComponent implements OnInit {

	//myfile = new File(null, null)
	image_score = 0;
	fileToUpload = File;
	filename = '';
	uploaded = false;
	imgUrl: any;

	constructor(private service: ImageUploadService) {}

	ngOnInit(): void {}

	fileChange(event: any) {
		console.log(event.target.files[0]);
		this.fileToUpload = event.target.files[0];
		this.filename = this.fileToUpload.name;

		// to display the image right after upload
		const reader = new FileReader();
		reader.readAsDataURL(event.target.files[0]); 
		reader.onload = (_event) => { 
			this.imgUrl = reader.result; 
		}
	  
	}
	
	uploadFile() {
		this.service.printUrl();
		console.log('uploadFile()');
		this.service.postFile(this.fileToUpload).subscribe((data) => {this.image_score = data.score; this.uploaded=true; console.log(data)});
	}
}
