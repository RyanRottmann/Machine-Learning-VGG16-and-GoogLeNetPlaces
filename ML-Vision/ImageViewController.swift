//
//  ImageViewController.swift
//  ML-Vision
//
//  Created by DALE MUSSER on 12/12/17.
//  Updated 10/26/18 for Xcode 10.0
//  Copyright © 2017 Tech Innovator. All rights reserved.
//
// http://www.wolfib.com/Image-Recognition-Intro-Part-1/
// https://developer.apple.com/machine-learning/

import UIKit
import CoreML
import Vision

class ImageViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var textView: UITextView!
    @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
    @IBOutlet weak var textView2: UITextView!
    @IBOutlet weak var activityIndicator2: UIActivityIndicatorView!
    
    let imagePicker = UIImagePickerController()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        imagePicker.delegate = self
        textView.text = ""
        textView2.text = ""
        activityIndicator.hidesWhenStopped = true
        activityIndicator2.hidesWhenStopped = true
        
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    @IBAction func cameraSelected(_ sender: Any) {
        takePhotoWithCamera()
    }
    

    @IBAction func photoLibrarySelected(_ sender: Any) {
        pickPhotoFromLibrary()
    }
    
    func takePhotoWithCamera() {
        if (!UIImagePickerController.isSourceTypeAvailable(UIImagePickerController.SourceType.camera)) {
            let alertController = UIAlertController(title: "No Camera", message: "The device has no camera.", preferredStyle: .alert)
            let okAction = UIAlertAction(title: "OK", style: .default, handler: nil)
            alertController.addAction(okAction)
            present(alertController, animated: true, completion: nil)
        } else {
            imagePicker.allowsEditing = false
            imagePicker.sourceType = .camera
            present(imagePicker, animated: true, completion: nil)
        }
    }
    
    func pickPhotoFromLibrary() {
        imagePicker.allowsEditing = false
        imagePicker.sourceType = .photoLibrary
        present(imagePicker, animated: true, completion: nil)
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        if let pickedImage = info[UIImagePickerController.InfoKey.originalImage] as? UIImage {
            imageView.contentMode = .scaleAspectFit
            imageView.image = pickedImage
            textView.text = ""
            textView2.text = ""
            
            guard let ciImage = CIImage(image: pickedImage) else {
                displayString(string: "Unable to convert image to CIImage.");
                return
            }
            
            detectScene(image: ciImage)
            detectScene2(image: ciImage)
        }
        
        dismiss(animated: true, completion: nil)
    }
    
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        dismiss(animated: true, completion: nil)
    }
    
    func displayString(string: String) {
        textView.text = textView.text + string + "\n";
    }
    
    func displayString2(string: String) {
        textView2.text = textView2.text + string + "\n";
    }
    func clearText(){
        textView.text = nil
    }
    func clearText2(){
        textView2.text = nil
    }
    func detectScene(image: CIImage) {
        displayString(string: "Detecting Scene Using VGG16...")
        
        // Load the ML model through its generated class
        /*
         guard let model = try? VNCoreMLModel(for: GoogLeNetPlaces().model) else {
         displayString(string: "Can't load ML model.")
         return
         }
         */
        guard let model = try? VNCoreMLModel(for: VGG16().model) else {
            displayString(string: "Can't load ML model.")
            return
        }
        
        
        // Create a Vision request with completion handler
        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            guard let results = request.results as? [VNClassificationObservation],
                let _ = results.first else {
                    self?.displayString(string: "Unexpected result type from VNCoreMLRequest")
                    return
            }
            
            // Update UI on main queue
            DispatchQueue.main.async { [weak self] in
                self?.activityIndicator.stopAnimating()
                self?.clearText()
                self?.displayString(string: "VGG16 Machine Learning Complete\n")
                for result in results {
                    self?.displayString(string: "\(Int(result.confidence * 100))% \(result.identifier)")
                }
            }
        }
        
        activityIndicator.startAnimating()
        
        // Run the Core ML GoogLeNetPlaces classifier on global dispatch queue
        let handler = VNImageRequestHandler(ciImage: image)
        DispatchQueue.global(qos: .userInteractive).async {
            do {
                try handler.perform([request])
            } catch {
                DispatchQueue.main.async { [weak self] in
                    self?.displayString(string: error.localizedDescription)
                    self?.activityIndicator.stopAnimating()
                }
            }
        }
    }

    func detectScene2(image: CIImage) {
        displayString2(string: "Detecting Scene Using GoogLeNetPlaces...")
        
        // Load the ML model through its generated class
        
         guard let model = try? VNCoreMLModel(for: GoogLeNetPlaces().model) else {
         displayString2(string: "Can't load ML model.")
         return
         }
         /*
        guard let model = try? VNCoreMLModel(for: VGG16().model) else {
            displayString(string: "Can't load ML model.")
            return
        }*/
        
        
        // Create a Vision request with completion handler
        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            guard let results = request.results as? [VNClassificationObservation],
                let _ = results.first else {
                    self?.displayString2(string: "Unexpected result type from VNCoreMLRequest")
                    return
            }
            
            // Update UI on main queue
            DispatchQueue.main.async { [weak self] in
                self?.activityIndicator2.stopAnimating()
                self?.clearText2()
                self?.displayString2(string: "GoogLeNetPlaces Machine Learning Complete\n")
                for result in results {
                    self?.displayString2(string: "\(Int(result.confidence * 100))% \(result.identifier)")
                }
            }
        }
        
        activityIndicator2.startAnimating()
        
        // Run the Core ML GoogLeNetPlaces classifier on global dispatch queue
        let handler = VNImageRequestHandler(ciImage: image)
        DispatchQueue.global(qos: .userInteractive).async {
            do {
                try handler.perform([request])
            } catch {
                DispatchQueue.main.async { [weak self] in
                    self?.displayString2(string: error.localizedDescription)
                    self?.activityIndicator2.stopAnimating()
                }
            }
        }
    }}



