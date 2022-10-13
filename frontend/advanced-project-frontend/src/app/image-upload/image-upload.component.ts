import { Component, OnInit } from '@angular/core';
import { ImageUploadService } from '../image-upload.service';
import { img_score } from '../img_score';

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

	constructor(private service: ImageUploadService) {}

	ngOnInit(): void {}

	fileChange(event: any) {
		console.log(event.target.files[0]);
		this.fileToUpload = event.target.files[0];
		this.filename = this.fileToUpload.name;
		console.log('fine')
	}
	
	uploadFile() {
		this.service.printUrl();
		console.log('uploadFile()');
		this.service.postFile(this.fileToUpload).subscribe((data) => {this.image_score = data.score; this.uploaded=true; console.log(data)});
	}

	/*
	handleFileInput(event: any) {
		if (event.target.files.length > 0) {
			const file = event.target.files[0];
			this.formGroup.patchValue({
			fileSource: file
      });
		}
	}
	
	
	uploadFileToActivity() {
		this.service.postFile(this.formGroup.get('fileSource')?.value).subscribe(data => {
		  // do something, if upload success
		  console.log('success')
		  }, error => {
			console.log(error);
		  });
	  }

	submit(){
		const formData = new FormData();
		formData.append('file', this.formGroup.get('fileSource')?.value);
			
		this.http.post('http://localhost:8001/upload.php', formData)
			.subscribe(res => {
			console.log(res);
			alert('Uploaded Successfully.');
			})
	  }*/
}
