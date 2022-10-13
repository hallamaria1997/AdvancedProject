import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, of } from 'rxjs';
import { catchError, tap } from 'rxjs/operators';
import { img_score } from './img_score';

@Injectable({
  providedIn: 'root'
})
export class ImageUploadService {

	constructor(private http: HttpClient) { }

	private readonly api_url = 'http://localhost:3000/api/v1/image'

	handleError(e: string): void {
		console.error(e);
	}
	
	printUrl(): void {
		console.log(this.http)
	}

	postFile(fileToUpload: any): Observable<img_score> {
		console.log('postFile')
		const formData: FormData = new FormData();
		formData.append('file', fileToUpload, fileToUpload.name);
		return this.http
			.post<img_score>(this.api_url, formData);
	}
}
