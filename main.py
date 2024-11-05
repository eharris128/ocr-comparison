import pytesseract
from google.cloud import vision
from google.oauth2 import service_account
import time
import cv2

class OCRComparison:
    def __init__(self):
        # Initialize Google Cloud Vision client
        self.credentials = service_account.Credentials.from_service_account_file('./service-account.json')
        self.client = vision.ImageAnnotatorClient(credentials=self.credentials)
        
    def preprocess_image(self, image_path):
        """Common preprocessing for both OCR engines"""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return gray
        
    def tesseract_ocr(self, image_path):
        """Process image with Tesseract"""
        start_time = time.time()
        
        # Preprocess image
        processed_image = self.preprocess_image(image_path)
        
        # Configure Tesseract
        custom_config = r'--oem 3 --psm 6'
        
        # Perform OCR
        text = pytesseract.image_to_string(processed_image, config=custom_config)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        return {
            'text': text,
            'processing_time': processing_time,
            'confidence': pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT)
        }
        
    def google_cloud_vision(self, image_path):
        """Process image with Google Cloud Vision"""
        start_time = time.time()
        
        # Read image file
        with open(image_path, 'rb') as image_file:
            content = image_file.read()
        
        image = vision.Image(content=content)
        
        # Perform OCR
        response = self.client.document_text_detection(image=image)
        text = response.full_text_annotation.text
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Calculate average confidence
        confidence = 0
        num_blocks = 0
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                confidence += block.confidence
                num_blocks += 1
        
        avg_confidence = confidence / num_blocks if num_blocks > 0 else 0
        
        return {
            'text': text,
            'processing_time': processing_time,
            'confidence': avg_confidence
        }
        
    def compare_results(self, image_path):
        """Compare results from both OCR engines"""
        print("Starting OCR comparison...")
        print("-" * 50)
        
        # Run both OCR engines
        tesseract_result = self.tesseract_ocr(image_path)
        gcv_result = self.google_cloud_vision(image_path)
        
        # Print comparison
        print("Tesseract OCR Results:")
        print(f"Processing Time: {tesseract_result['processing_time']:.2f} seconds")
        print(f"Text Length: {len(tesseract_result['text'])} characters")
        print("\nExtracted Text:")
        print(tesseract_result['text'][:500] + "..." if len(tesseract_result['text']) > 500 else tesseract_result['text'])
        
        print("\nGoogle Cloud Vision Results:")
        print(f"Processing Time: {gcv_result['processing_time']:.2f} seconds")
        print(f"Confidence: {gcv_result['confidence']*100:.2f}%")
        print(f"Text Length: {len(gcv_result['text'])} characters")
        print("\nExtracted Text:")
        print(gcv_result['text'][:500] + "..." if len(gcv_result['text']) > 500 else gcv_result['text'])
        
        return {
            'tesseract': tesseract_result,
            'google_vision': gcv_result
        }

# Example usage
if __name__ == "__main__":
    comparison = OCRComparison()
    results = comparison.compare_results('./IMG_6353.jpg')